"""
job_worker.py — 训练任务 worker（Step 3 假执行版）
=================================================

职责：
  1. 轮询 train_runs 表中的 queued 任务
  2. 将任务状态更新为 running
  3. 模拟训练执行一段时间
  4. 将任务状态更新为 completed

说明：
  - 当前版本不调用真实 Whisper-Finetune，只验证任务消费链路
  - 后续可直接把 `_simulate_train_run()` 替换成真实训练执行逻辑
  - 设计上它应作为独立进程运行，后续也适合单独封装为 worker 容器
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from typing import Optional

from database import SessionLocal, TrainRun


def _claim_next_train_run() -> Optional[int]:
    """
    取出最早的 queued 任务，并将其置为 running。

    当前默认只跑一个 worker 进程，所以这里先用最简单的串行消费方式。
    """
    db = SessionLocal()
    try:
        train_run = (
            db.query(TrainRun)
            .filter(TrainRun.status == "queued")
            .order_by(TrainRun.created_at.asc())
            .first()
        )
        if not train_run:
            return None

        train_run.status = "running"
        train_run.started_at = datetime.utcnow()
        train_run.error_message = None
        db.commit()
        db.refresh(train_run)

        print(f"[WORKER] Claimed train run #{train_run.id}: {train_run.name}")
        return train_run.id
    finally:
        db.close()


def _simulate_train_run(run_id: int, fake_duration: float) -> None:
    """
    模拟训练执行。

    这里先 sleep 一段时间，后续替换成真实 `finetune.py` 调用。
    """
    print(f"[WORKER] Train run #{run_id} is running (fake mode, {fake_duration:.1f}s)")
    time.sleep(fake_duration)


def _mark_completed(run_id: int) -> None:
    db = SessionLocal()
    try:
        train_run = db.query(TrainRun).filter(TrainRun.id == run_id).first()
        if not train_run:
            return

        train_run.status = "completed"
        train_run.completed_at = datetime.utcnow()
        db.commit()
        print(f"[WORKER] Train run #{run_id} completed")
    finally:
        db.close()


def _mark_failed(run_id: int, error_message: str) -> None:
    db = SessionLocal()
    try:
        train_run = db.query(TrainRun).filter(TrainRun.id == run_id).first()
        if not train_run:
            return

        train_run.status = "failed"
        train_run.error_message = error_message[:1000]
        train_run.completed_at = datetime.utcnow()
        db.commit()
        print(f"[WORKER] Train run #{run_id} failed: {error_message}")
    finally:
        db.close()


def process_one_train_run(fake_duration: float) -> bool:
    """
    处理一条任务。

    Returns:
        True 表示处理了一条任务，False 表示当前没有任务
    """
    run_id = _claim_next_train_run()
    if run_id is None:
        return False

    try:
        _simulate_train_run(run_id, fake_duration=fake_duration)
        _mark_completed(run_id)
    except Exception as exc:
        _mark_failed(run_id, str(exc))

    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="ASR Eval training worker (fake Step 3)")
    parser.add_argument("--poll-interval", type=float, default=2.0, help="轮询间隔（秒）")
    parser.add_argument("--fake-duration", type=float, default=8.0, help="模拟训练耗时（秒）")
    parser.add_argument("--once", action="store_true", help="只执行一次轮询并退出")
    args = parser.parse_args()

    print(
        f"[WORKER] Started. poll_interval={args.poll_interval:.1f}s, "
        f"fake_duration={args.fake_duration:.1f}s, once={args.once}"
    )

    while True:
        handled = process_one_train_run(fake_duration=args.fake_duration)

        if args.once:
            break

        if not handled:
            time.sleep(args.poll_interval)


if __name__ == "__main__":
    main()
