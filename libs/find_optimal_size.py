import math


def find_optimal_rectangle(feature_num):
    """Find rectangle closest to square that can fit all features."""
    sqrt_n = int(math.sqrt(feature_num))

    # Find the best dimensions around the square root
    best_w, best_h = sqrt_n, math.ceil(feature_num / sqrt_n)
    best_diff = abs(best_w - best_h)

    # Check nearby factors for better square-like ratios
    for w in range(max(1, sqrt_n - 2), sqrt_n + 3):
        h = math.ceil(feature_num / w)
        diff = abs(w - h)

        if diff < best_diff:
            best_w, best_h = w, h
            best_diff = diff

    return best_w, best_h