"""Misc helper: timing context manager."""
import contextlib, time

@contextlib.contextmanager
def timer(msg: str):
    t0 = time.perf_counter()
    yield
    print(f"[{msg}] {time.perf_counter() - t0:.2f}s")