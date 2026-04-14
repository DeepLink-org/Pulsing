#!/usr/bin/env python3
"""Transfer Queue + Ray example for RL rollout/train handoff.

This example simulates a simple reinforcement-learning pipeline:
1. One Ray rollout actor generates ``train_datas``.
2. The rollout actor splits that list evenly into three shards.
3. The shards are stored in a transfer queue with ``num_buckets=3`` using
   ``sample_idx`` values ``0``, ``1``, and ``2`` so each shard lands in a
   different bucket.
4. Three Ray trainer actors with ranks ``0``, ``1``, and ``2`` each fetch the
   shard for their matching bucket.
5. The driver repeats rollout + training for multiple rounds.

Usage:
    python examples/python/transfer_queue_ray_rollout.py
"""

from __future__ import annotations

from typing import TypedDict

try:
    import ray
except ImportError as exc:
    raise ImportError(
        "This example requires Ray. Install with: pip install 'ray[default]'"
    ) from exc

import pulsing as pul

PARTITION_ID = "rl_rollout_demo"
NUM_BUCKETS = 3
RECORDS_PER_BUCKET = 4
NUM_ROUNDS = 3
ROLLOUT_BATCH_SIZE = 1
READ_TIMEOUT_SECONDS = 5.0


class TrainRecord(TypedDict):
    trajectory_id: int
    prompt: str
    response: str
    reward: float
    advantage: float
    rollout_round: int


def sample_idx_for_rank(rank: int) -> int:
    """Map each trainer rank to the shard it owns."""
    return rank


def build_train_datas(
    rollout_round: int,
    total_records: int = NUM_BUCKETS * RECORDS_PER_BUCKET,
) -> list[TrainRecord]:
    """Create a fake RL rollout batch."""
    train_datas: list[TrainRecord] = []
    trajectory_base = rollout_round * total_records
    for trajectory_id in range(total_records):
        global_trajectory_id = trajectory_base + trajectory_id
        reward = round(0.25 + trajectory_id * 0.1, 3)
        train_datas.append(
            {
                "trajectory_id": global_trajectory_id,
                "prompt": f"round-{rollout_round}-prompt-{global_trajectory_id}",
                "response": f"round-{rollout_round}-response-{global_trajectory_id}",
                "reward": reward,
                "advantage": round(reward - 0.75, 3),
                "rollout_round": rollout_round,
            }
        )
    return train_datas


def split_train_data_by_dp(
    records: list[TrainRecord], num_shards: int
) -> list[list[TrainRecord]]:
    """Split records into near-equal contiguous shards."""
    base_size, remainder = divmod(len(records), num_shards)
    shards: list[list[TrainRecord]] = []
    start = 0

    for shard_id in range(num_shards):
        shard_size = base_size + (1 if shard_id < remainder else 0)
        stop = start + shard_size
        shards.append(records[start:stop])
        start = stop

    return shards


@ray.remote
class RolloutWorker:
    """Produces rollout data and writes one shard per bucket."""

    def __init__(
        self, partition_id: str = PARTITION_ID, num_buckets: int = NUM_BUCKETS
    ):
        self.partition_id = partition_id
        self.num_buckets = num_buckets
        self.client = pul.transfer_queue.get_client(
            partition_id=self.partition_id,
            num_buckets=self.num_buckets,
            batch_size=ROLLOUT_BATCH_SIZE,
        )

    def generate(self, rollout_round: int) -> dict:
        train_datas = build_train_datas(rollout_round)
        shards = split_train_data_by_dp(train_datas, self.num_buckets)

        for rank, shard in enumerate(shards):
            sample_idx = sample_idx_for_rank(rank)
            self.client.put(
                sample_idx=sample_idx,
                data={
                    "train_data": shard,
                    "target_rank": rank,
                    "rollout_round": rollout_round,
                },
            )

        return {
            "rollout_round": rollout_round,
            "sample_idxs": [sample_idx_for_rank(rank) for rank in range(len(shards))],
            "shard_sizes": [len(shard) for shard in shards],
        }


@ray.remote
class TrainerWorker:
    """Runs one training step from the shard for its rank."""

    def __init__(
        self,
        rank: int,
        partition_id: str = PARTITION_ID,
        num_buckets: int = NUM_BUCKETS,
    ):
        self.rank = rank
        self.partition_id = partition_id
        self.num_buckets = num_buckets
        self.client = pul.transfer_queue.get_client(
            partition_id=self.partition_id,
            num_buckets=self.num_buckets,
            batch_size=ROLLOUT_BATCH_SIZE,
        )

    def run_training_step(self, rollout_round: int) -> dict:
        rows = self.client.get(
            data_fields=["train_data", "target_rank", "rollout_round"],
            sample_idxs=[sample_idx_for_rank(self.rank)],
            batch_size=1,
            task_name=f"trainer_rank_{self.rank}_round_{rollout_round}",
            timeout=READ_TIMEOUT_SECONDS,
        )
        if not rows:
            raise TimeoutError(
                f"rank={self.rank} did not receive round={rollout_round} shard in time"
            )

        row = rows[0]
        shard = row["train_data"]
        avg_reward = round(
            sum(item["reward"] for item in shard) / len(shard),
            3,
        )
        bucket_id = row["sample_idx"] % self.num_buckets
        trajectory_ids = [item["trajectory_id"] for item in shard]

        return {
            "rollout_round": rollout_round,
            "rank": self.rank,
            "sample_idx": row["sample_idx"],
            "bucket_id": bucket_id,
            "record_count": len(shard),
            "trajectory_ids": trajectory_ids,
            "avg_reward": avg_reward,
        }


def main() -> None:
    print("=== Transfer Queue Ray Rollout Example ===")
    ray.init(num_cpus=NUM_BUCKETS + 1)

    try:
        rollout_worker = RolloutWorker.remote()
        trainer_workers = [TrainerWorker.remote(rank=i) for i in range(NUM_BUCKETS)]

        for rollout_round in range(NUM_ROUNDS):
            produced = ray.get(rollout_worker.generate.remote(rollout_round))
            print(
                f"round={produced['rollout_round']} rollout stored shards: "
                f"sample_idxs={produced['sample_idxs']} "
                f"shard_sizes={produced['shard_sizes']}"
            )

            trainer_summaries = ray.get(
                [
                    trainer.run_training_step.remote(rollout_round)
                    for trainer in trainer_workers
                ]
            )

            for summary in sorted(trainer_summaries, key=lambda item: item["rank"]):
                print(
                    f"round={summary['rollout_round']} rank={summary['rank']} "
                    f"trained on sample_idx={summary['sample_idx']} "
                    f"(bucket={summary['bucket_id']}), records={summary['record_count']}, "
                    f"trajectory_ids={summary['trajectory_ids']}, "
                    f"avg_reward={summary['avg_reward']}"
                )

            print(f"Round {rollout_round} training finished for all trainer ranks.")
    finally:
        pul.cleanup_ray()
        ray.shutdown()


if __name__ == "__main__":
    main()
