"""
fusion.py - Prediction fusion utilities
"""


def normalize(pred_dict):
    """Normalize a dict of {label: score} so values sum to 1."""
    total = sum(pred_dict.values())
    if total == 0:
        n = len(pred_dict)
        return {k: 1.0 / n for k in pred_dict}
    return {k: v / total for k, v in pred_dict.items()}


def get_top_prediction(pred_dict):
    """Return the label with the highest score."""
    return max(pred_dict, key=pred_dict.get)


def get_top_k(pred_dict, k=3):
    """Return top-k predictions as list of (label, score) tuples, sorted descending."""
    sorted_preds = sorted(pred_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_preds[:k]


def fuse_predictions(pred1, pred2, weight1=0.5, weight2=0.5):
    """
    Weighted fusion of two prediction dicts.
    Only fuses keys present in BOTH dicts.
    Falls back to whichever dict has more keys if no overlap.
    """
    keys = set(pred1.keys()) | set(pred2.keys())
    fused = {}
    for k in keys:
        v1 = pred1.get(k, 0.0)
        v2 = pred2.get(k, 0.0)
        fused[k] = weight1 * v1 + weight2 * v2
    return fused
