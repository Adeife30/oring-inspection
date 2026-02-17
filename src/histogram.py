import numpy as np


def compute_histogram(img: np.ndarray) -> np.ndarray:
    """
    Compute 256-bin histogram for a uint8 grayscale image using NumPy only.
    Returns: hist shape (256,)
    """
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    hist = np.bincount(img.ravel(), minlength=256).astype(np.int64)
    return hist


def otsu_threshold(hist: np.ndarray) -> int:
    """
    Otsu threshold computed from a histogram (NumPy only).
    Returns threshold in [0, 255].
    """
    hist = hist.astype(np.float64)
    total = hist.sum()
    if total == 0:
        return 128

    # probabilities
    p = hist / total

    omega = np.cumsum(p)                      # cumulative class probabilities
    mu = np.cumsum(p * np.arange(256))        # cumulative class means
    mu_t = mu[-1]                             # total mean

    # Between-class variance: (mu_t*omega - mu)^2 / (omega*(1-omega))
    denom = omega * (1.0 - omega)
    # avoid division by zero
    denom[denom == 0] = np.nan

    sigma_b2 = ((mu_t * omega - mu) ** 2) / denom

    t = int(np.nanargmax(sigma_b2))
    return t
