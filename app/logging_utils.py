from __future__ import annotations

import logging
import queue
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional


class QueueLogHandler(logging.Handler):
    def __init__(self, log_queue: Optional[queue.Queue] = None):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record: logging.LogRecord) -> None:
        if self.log_queue is None:
            return
        try:
            self.log_queue.put_nowait(self.format(record))
        except Exception:
            pass


def setup_logger(log_dir: Path, log_queue: Optional[queue.Queue] = None, name: str = 'id_ocr_gui') -> logging.Logger:
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = log_dir / f'run_{ts}.log'

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    fh = logging.FileHandler(log_path, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    qh = QueueLogHandler(log_queue)
    qh.setFormatter(fmt)
    logger.addHandler(qh)

    logger.info('日志初始化完成：%s', log_path)
    return logger
