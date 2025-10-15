import resource
from contextlib import contextmanager

class MemoryLimitExceeded(BaseException):
    pass

@contextmanager
def apply_memory_limit(memory_limit_mb: int):
    max_bytes = memory_limit_mb * 1024 * 1024
    old_soft, old_hard = resource.getrlimit(resource.RLIMIT_AS)
    # Compute the new soft cap but keep the hard cap unchanged
    new_soft = min(max_bytes, old_hard if old_hard != resource.RLIM_INFINITY else max_bytes)

    changed = False
    try:
        if old_soft == resource.RLIM_INFINITY or new_soft < old_soft:
            resource.setrlimit(resource.RLIMIT_AS, (new_soft, old_hard))
            changed = True
        yield
    finally:
        if changed:
            # Restore soft; never attempt to raise hard
            cur_soft, cur_hard = resource.getrlimit(resource.RLIMIT_AS)
            restore_soft = min(old_soft, cur_hard)  # cap to current hard if it tightened
            try:
                resource.setrlimit(resource.RLIMIT_AS, (restore_soft, cur_hard))
            except ValueError:
                # If even that fails (rare), fall back to the maximum allowed soft
                resource.setrlimit(resource.RLIMIT_AS, (cur_hard, cur_hard))


def preexec_memory_limit(memory_limit_mb):
    import resource
    limit = int(memory_limit_mb) * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if hard == resource.RLIM_INFINITY or limit <= hard:
        new_soft, new_hard = limit, limit
    else:
        new_soft, new_hard = hard, hard
    resource.setrlimit(resource.RLIMIT_AS, (new_soft, new_hard))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))
