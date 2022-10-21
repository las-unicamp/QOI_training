from functools import wraps
import time


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        print(
            "Training complete in "
            + f"{elapsed_time // (60 * 60 * 24):.0f}d  "
            + f"{elapsed_time // (60 * 60):.0f}h  "
            + f"{elapsed_time // 60:.0f}m  "
            + f"{elapsed_time % 60:.0f}s"
        )
        return result

    return timeit_wrapper
