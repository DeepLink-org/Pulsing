#!/usr/bin/env python3
"""Transfer Queue (sync client) example — using TransferQueueClient from plain threads.

Shows how to use the synchronous TransferQueueClient when the caller code is
plain synchronous Python (e.g. a training loop, a data-loading worker thread).
Do not call this sync client from the active Pulsing event loop thread; use
``await pul.transfer_queue.get_async_client()`` there, or move sync usage into
``asyncio.to_thread(...)``.

Usage:
    python examples/python/transfer_queue_sync.py
"""

import logging

import pulsing as pul

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Transfer Queue Sync Client Example ===\n")

    try:
        # 1) Get a sync client. transfer_queue bootstraps Pulsing internally.
        client = pul.transfer_queue.get_client(
            partition_id="demo_sync", num_buckets=2, batch_size=4
        )
        logger.info("Transfer queue client created (2 buckets, batch_size=4)\n")

        # --- Phase 1: Simulate inference producing prompts ---
        logger.info("--- Phase 1: Writing prompts ---")
        for i in range(6):
            meta = client.put(sample_idx=i, data={"prompt": f"Question {i}"})
            logger.info(f"  sample {i}: wrote prompt, fields={meta['fields']}")

        # Try to read — samples are incomplete (no response yet)
        incomplete = client.get(
            data_fields=["prompt", "response"], batch_size=10, task_name="train"
        )
        logger.info(
            f"\nAfter prompts only: {len(incomplete)} complete samples (expected 0)\n"
        )

        # --- Phase 2: Simulate inference producing responses ---
        logger.info("--- Phase 2: Writing responses ---")
        for i in range(6):
            meta = client.put(sample_idx=i, data={"response": f"Answer {i}"})
            logger.info(f"  sample {i}: wrote response, fields={meta['fields']}")

        # --- Phase 3: Consume complete samples in batches ---
        logger.info("\n--- Phase 3: Consuming complete samples ---")
        batch1 = client.get(
            data_fields=["prompt", "response"], batch_size=4, task_name="train"
        )
        logger.info(f"Batch 1 ({len(batch1)} samples):")
        for s in batch1:
            logger.info(f"  prompt={s['prompt']!r}, response={s['response']!r}")

        batch2 = client.get(
            data_fields=["prompt", "response"], batch_size=4, task_name="train"
        )
        logger.info(f"Batch 2 ({len(batch2)} samples):")
        for s in batch2:
            logger.info(f"  prompt={s['prompt']!r}, response={s['response']!r}")

        # --- Phase 4: Independent consumer reads the same data ---
        logger.info("\n--- Phase 4: Independent consumer (different task_name) ---")
        eval_batch = client.get(
            data_fields=["prompt", "response"], batch_size=10, task_name="eval"
        )
        logger.info(
            f"Eval consumer got {len(eval_batch)} samples "
            f"(same data, independent tracking)"
        )

        # --- Phase 5: Targeted fetch with sample_idxs + timeout ---
        logger.info("\n--- Phase 5: Targeted sample_idxs fetch ---")
        subset = client.get(
            data_fields=["prompt", "response"],
            sample_idxs=[3, 1],
            batch_size=2,
            task_name="debug_subset",
            timeout=0.5,
        )
        logger.info(
            "Subset fetch returned sample_idxs=%s",
            [row["sample_idx"] for row in subset],
        )

        # --- Cleanup ---
        client.clear()
        logger.info("\nCleared all data")
        logger.info("Example completed!")

    finally:
        logger.info("Example finished (transfer_queue runtime cleanup is automatic)")


if __name__ == "__main__":
    main()
