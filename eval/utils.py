
def get_batch_ranges(n, batch_size):
    batch_starts = list(range(0, n, batch_size))
    batch_ranges = []
    for batch_start in batch_starts:
        batch_end = min(n, batch_start + batch_size)
        batch_ranges.append((batch_start, batch_end))
    
    return batch_ranges
