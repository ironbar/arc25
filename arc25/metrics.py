import numpy as np


def pixel_similarity_score(ground_truth: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute the average pixel-wise match between two integer matrices.

    - If the matrices have the same shape, returns the fraction of pixels that are equal.
    - If shapes differ, slides the smaller matrix over the larger one to find the region
      with the maximum number of matching pixels, then counts all other pixels as mismatches.

    Returns a score in [0.0, 1.0].
    """
    gt = np.asarray(ground_truth)
    ref = np.asarray(reference)

    # same shape: global pixel accuracy
    if gt.shape == ref.shape:
        return float(np.mean(gt == ref))

    # identify which is smaller by area
    if gt.size <= ref.size:
        small, large = gt, ref
    else:
        small, large = ref, gt

    hs, ws = small.shape
    hl, wl = large.shape
    best_matches = 0

    # slide small over large
    for i in range(hl - hs + 1):
        for j in range(wl - ws + 1):
            region = large[i : i + hs, j : j + ws]
            matches = np.count_nonzero(region == small)
            if matches > best_matches:
                best_matches = matches

    # total pixels is size of larger matrix (all other pixels counted as mismatches)
    return best_matches / float(large.size)
