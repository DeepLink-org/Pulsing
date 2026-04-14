#!/usr/bin/env python3
"""Minimal DP x TP transfer-queue demo.

Different DP ranks read different shards. TP ranks in the same DP group read
the same shard.

Usage:
    python examples/python/transfer_queue_ray_rollout.py
"""

try:
    import ray
except ImportError as exc:
    raise ImportError(
        "This example requires Ray. Install with: pip install 'ray[default]'"
    ) from exc

import pulsing as pul
from pulsing.integrations.ray import init_in_ray

TOPIC = "rollout_demo"
DP_SIZE = 3
TP_SIZE = 2
WORLD_SIZE = DP_SIZE * TP_SIZE
NUM_ROLLOUTS = 6
READ_TIMEOUT_SECONDS = 5.0


def print_rollout_report(rollout_id: int, summaries: list[dict]) -> None:
    """Print all summary entries for one rollout in a compact table."""
    columns = [
        ("global_rank", "global"),
        ("dp_rank", "dp"),
        ("tp_rank", "tp"),
        ("bucket_id", "bucket"),
        ("sample_idx", "sample"),
        ("trajectory_id", "traj"),
        ("reward", "reward"),
        ("advantage", "adv"),
        ("prompt", "prompt"),
        ("response", "response"),
    ]
    ordered = sorted(
        summaries,
        key=lambda summary: (
            summary["dp_rank"],
            summary["tp_rank"],
            summary["global_rank"],
        ),
    )
    rendered_rows: list[dict[str, str]] = []
    widths: dict[str, int] = {key: len(label) for key, label in columns}

    for summary in ordered:
        row = {
            "global_rank": str(summary["global_rank"]),
            "dp_rank": str(summary["dp_rank"]),
            "tp_rank": str(summary["tp_rank"]),
            "bucket_id": str(summary["bucket_id"]),
            "sample_idx": str(summary["sample_idx"]),
            "trajectory_id": str(summary["trajectory_id"]),
            "reward": f"{summary['reward']:.3f}",
            "advantage": f"{summary['advantage']:.3f}",
            "prompt": str(summary["prompt"]),
            "response": str(summary["response"]),
        }
        rendered_rows.append(row)
        for key, _ in columns:
            widths[key] = max(widths[key], len(row[key]))

    def format_row(values: dict[str, str] | None = None) -> str:
        parts = []
        for key, label in columns:
            value = label if values is None else values[key]
            parts.append(f"{value:<{widths[key]}}")
        return " | ".join(parts)

    separator = "-+-".join("-" * widths[key] for key, _ in columns)

    print(f"\n=== Rollout {rollout_id} ===")
    print(format_row())
    print(separator)
    for row in rendered_rows:
        print(format_row(row))


class Box:
    def __init__(self, inner):
        self._inner = inner

    @property
    def inner(self):
        return self._inner


@ray.remote
class RolloutWorker:
    """Writes one shard per DP rank."""

    def __init__(self, dp_size):
        self.dp_size = dp_size
        self.client = pul.transfer_queue.get_client(
            topic=TOPIC, num_buckets=self.dp_size, bucket_capacity=3
        )

    def generate(self, rollout_id: int) -> list:
        data = [
            {
                "trajectory_id": rollout_id,
                "prompt": f"prompt-{i}",
                "response": f"response-{i}",
                "reward": round(0.25 + i * 0.1, 3),
                "advantage": round(0.25 + i * 0.1 - 0.75, 3),
            }
            for i in range(self.dp_size)
        ]
        rollout_data_refs = []
        for dp_rank in range(self.dp_size):
            self.client.put(
                sample_idx=rollout_id,
                data={"train_data": data[dp_rank]},
                bucket_id=dp_rank,
            )
            rollout_data_refs.append(Box(ray.put((rollout_id, dp_rank))))
        return rollout_data_refs


@ray.remote
class TrainerWorker:
    """Reads the shard for one global rank."""

    def __init__(
        self,
        global_rank,
        dp_size,
        tp_size,
    ):
        self.global_rank = global_rank
        self.dp_size = dp_size
        self.dp_rank, self.tp_rank = divmod(global_rank, tp_size)
        self.client = pul.transfer_queue.get_client(
            topic=TOPIC, num_buckets=self.dp_size, bucket_capacity=3
        )

    def run_training_step(self, rollout_data_refs: list):
        rollout_id, dp_rank = ray.get(rollout_data_refs[self.dp_rank].inner)
        data = self.client.get(
            sample_idx=rollout_id,
            data_fields=["train_data"],
            bucket_id=dp_rank,
            timeout=READ_TIMEOUT_SECONDS,
        )
        if data is None:
            raise TimeoutError(
                f"global_rank={self.global_rank} dp_rank={self.dp_rank} "
                f"did not receive rollout_id={rollout_id} from bucket_id={dp_rank}"
            )
        sample = data["train_data"]
        return {
            "global_rank": self.global_rank,
            "dp_rank": self.dp_rank,
            "tp_rank": self.tp_rank,
            "bucket_id": dp_rank,
            "sample_idx": rollout_id,
            "trajectory_id": sample["trajectory_id"],
            "prompt": sample["prompt"],
            "response": sample["response"],
            "reward": sample["reward"],
            "advantage": sample["advantage"],
        }


def main() -> None:
    print("=== Transfer Queue Ray Rollout Demo ===")
    print(f"topic={TOPIC} dp={DP_SIZE} tp={TP_SIZE} world={WORLD_SIZE}")
    ray.init(
        num_cpus=WORLD_SIZE + 1,
        runtime_env={"worker_process_setup_hook": init_in_ray},
    )
    start_rollout_id = 0
    num_rollout = NUM_ROLLOUTS
    try:
        rollout_manager = RolloutWorker.remote(DP_SIZE)
        trainer_workers = [
            TrainerWorker.remote(i, DP_SIZE, TP_SIZE) for i in range(WORLD_SIZE)
        ]

        rollout_data_next_future = rollout_manager.generate.remote(start_rollout_id)
        for rollout_id in range(start_rollout_id, num_rollout):
            if rollout_data_next_future is not None:
                rollout_data_curr_ref = ray.get(rollout_data_next_future)

            if rollout_id + 1 < num_rollout:
                rollout_data_next_future = rollout_manager.generate.remote(
                    rollout_id + 1
                )

            trainer_refs = [
                trainer_worker.run_training_step.remote(rollout_data_curr_ref)
                for trainer_worker in trainer_workers
            ]
            trainer_summaries = ray.get(trainer_refs)
            print_rollout_report(rollout_id, trainer_summaries)

    finally:
        pul.cleanup_ray()
        ray.shutdown()


if __name__ == "__main__":
    main()
