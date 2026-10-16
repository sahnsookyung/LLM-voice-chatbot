"""Bounded timing records; never collect speech, prompts, vocabulary or raw audio."""

import time
from collections import deque

import psutil


def memory_snapshot():
    process = psutil.Process()
    workers = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            workers += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    native = renderer = 0
    try:
        parent = process.parent()
        if parent and parent.name() == "Companion":
            native = parent.memory_info().rss
            for child in parent.children(recursive=True):
                if "WebKit" in child.name():
                    renderer += child.memory_info().rss
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass
    return {
        "process_rss_bytes": workers,
        "native_rss_bytes": native,
        "renderer_rss_bytes": renderer,
        # macOS may host WebKit under launchd rather than this process tree.
        # A missing child is not evidence of zero renderer memory usage.
        "renderer_memory_available": renderer > 0,
        "whole_app_rss_complete": native > 0 and renderer > 0,
        "whole_app_rss_bytes": workers + native + renderer,
        "system_available_bytes": psutil.virtual_memory().available,
        "system_swap_bytes": psutil.swap_memory().used,
    }


class Timings:
    def __init__(self):
        self.records = deque(maxlen=3000)

    def record(self, stage, seconds, **configuration):
        self.records.append({"stage": stage, "seconds": max(0, seconds), **configuration})

    def since(self, stage, start, **configuration):
        self.record(stage, time.monotonic() - start, **configuration)

    def export(self):
        return {
            "version": 1,
            "clock": "process-local monotonic durations",
            "records": list(self.records),
        }
