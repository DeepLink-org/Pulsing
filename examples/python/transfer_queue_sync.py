#!/usr/bin/env python3
"""Transfer Queue (sync client) example — using TransferQueueClient from plain threads.

Shows how to use the synchronous TransferQueueClient when the caller code is
plain synchronous Python (e.g. a training loop, a data-loading worker thread).

Usage:
    python examples/python/transfer_queue_sync.py
"""

import asyncio
import logging
import threading

import pulsing as pul

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    logger.info("=== Transfer Queue Sync Client Example ===\n")

    # 1) Create a background event loop and initialize pulsing on it
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True, name="pulsing-loop").start()
    asyncio.run_coroutine_threadsafe(pul.init(), loop).result(timeout=30)
    logger.info("Pulsing initialized\n")

    try:
        # 2) Get a sync client (using the background loop)
        client = pul.transfer_queue.get_client(
            partition_id="demo_sync", num_buckets=2, batch_size=4, loop=loop
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

        # --- Cleanup ---
        client.clear()
        logger.info("\nCleared all data")
        logger.info("Example completed!")

    finally:
        # 3) Shutdown pulsing and stop the background loop
        asyncio.run_coroutine_threadsafe(pul.shutdown(), loop).result(timeout=10)
        loop.call_soon_threadsafe(loop.stop)
        logger.info("System shutdown")


if __name__ == "__main__":
    main()
