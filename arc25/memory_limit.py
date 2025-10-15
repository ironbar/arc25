import resource
from contextlib import contextmanager

class MemoryLimitExceeded(BaseException):
    pass

@contextmanager
def apply_memory_limit(memory_limit_mb: int):
    max_bytes = memory_limit_mb * 1024 * 1024
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    new_soft = min(max_bytes, old_hard if old_hard != resource.RLIM_INFINITY else max_bytes)
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_soft))
    try:
        yield
    finally:
        resource.setrlimit(resource.RLIMIT_AS, (old_soft, old_hard))
