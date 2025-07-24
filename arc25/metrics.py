import numpy as np


def pixel_similarity_score(ground_truth: np.ndarray, reference: np.ndarray) -> float:
    """
    Compute a pixel-wise similarity score between two 2D integer matrices.

    - If the matrices have the same shape, returns the fraction of pixels that match.
    - If the shapes differ, finds the best overlap alignment and divides
      the number of matching pixels by the area of the smallest bounding box
      that contains both matrices.

    Returns a float in [0.0, 1.0].
    """
    gt = np.asarray(ground_truth)
    ref = np.asarray(reference)
    # Dimension checks
    if gt.ndim > 2 or ref.ndim > 2:
        raise ValueError("Only 1D or 2D arrays are supported")
    if gt.ndim == 1:
        gt = gt[np.newaxis, :]
    elif gt.ndim == 0:
        gt = np.array([[gt]], dtype=int)
    if ref.ndim == 1:
        ref = ref[np.newaxis, :]
    elif ref.ndim == 0:
        ref = np.array([[ref]], dtype=int)

    h1, w1 = gt.shape
    h2, w2 = ref.shape

    # same shape: simple pixel accuracy
    if h1 == h2 and w1 == w2:
        return float(np.mean(gt == ref))

    # different shapes: slide one over the other to maximize matches
    bbox_h = max(h1, h2)
    bbox_w = max(w1, w2)
    denom = bbox_h * bbox_w
    best_matches = 0

    # dx, dy are offsets from ref to gt
    for dx in range(-(h2 - 1), h1):
        for dy in range(-(w2 - 1), w1):
            # overlap region in gt
            i1_start = max(0, dx)
            i1_end   = min(h1, h2 + dx)
            j1_start = max(0, dy)
            j1_end   = min(w1, w2 + dy)
            if i1_end <= i1_start or j1_end <= j1_start:
                continue

            # corresponding region in ref
            i2_start = i1_start - dx
            i2_end   = i1_end   - dx
            j2_start = j1_start - dy
            j2_end   = j1_end   - dy

            region_gt  = gt[i1_start:i1_end, j1_start:j1_end]
            region_ref = ref[i2_start:i2_end, j2_start:j2_end]
            matches = np.count_nonzero(region_gt == region_ref)

            if matches > best_matches:
                best_matches = matches

    return best_matches / float(denom)
