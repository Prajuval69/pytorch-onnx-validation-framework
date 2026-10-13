"""NumPy-only numerical semantics. Cosine is diagnostic, never the verdict."""

import math

import numpy as np


def _finite(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


def _sample(value: np.generic) -> float | str:
    result = float(value)
    return result if math.isfinite(result) else str(result)


def compare(
    reference: dict[str, np.ndarray],
    candidate: dict[str, np.ndarray],
    atol: float = 1e-5,
    rtol: float = 1e-4,
    nan_policy: str = "fail",
) -> dict:
    if not math.isfinite(atol) or not math.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("Tolerances must be finite and nonnegative")
    if nan_policy not in ("fail", "equal"):
        raise ValueError("Unsupported nonfinite policy")
    if not reference or set(reference) != set(candidate):
        return {"status": "FAIL", "metrics": {}, "reason": "Output names/count mismatch"}
    metrics = {}
    reasons = []
    for name, expected in reference.items():
        actual = candidate[name]
        if expected.shape != actual.shape or expected.dtype != actual.dtype:
            reasons.append(f"{name}: output shape/dtype mismatch")
            metrics[name] = {
                "reference_shape": list(expected.shape),
                "candidate_shape": list(actual.shape),
                "reference_dtype": str(expected.dtype),
                "candidate_dtype": str(actual.dtype),
            }
            continue
        if expected.dtype.kind not in "fiu" or expected.size == 0:
            reasons.append(f"{name}: unsupported or empty output")
            metrics[name] = {}
            continue
        ref = expected.astype(np.float64)
        cand = actual.astype(np.float64)
        finite = np.isfinite(ref) & np.isfinite(cand)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            difference = np.abs(ref - cand)
            threshold = atol + rtol * np.abs(ref)
            close = finite & (difference <= threshold)
            if nan_policy == "equal":
                close |= (np.isnan(ref) & np.isnan(cand)) | (
                    np.isinf(ref) & np.isinf(cand) & (ref == cand)
                )
            mismatch = ~close
            count = int(np.count_nonzero(mismatch))
            if np.all(finite):
                max_abs = _finite(float(np.max(difference)))
                max_rel = _finite(
                    float(
                        np.max(
                            difference
                            / np.maximum(np.abs(ref), max(atol, np.finfo(np.float64).eps))
                        )
                    )
                )
                mean_abs = _finite(float(np.mean(difference)))
                scale = float(np.max(difference))
                rmse = (
                    0.0
                    if scale == 0
                    else _finite(scale * float(np.sqrt(np.mean((difference / scale) ** 2))))
                )
                a_scale, b_scale = float(np.max(np.abs(ref))), float(np.max(np.abs(cand)))
                cosine = None
                if a_scale > 0 and b_scale > 0:
                    a, b = (ref / a_scale).ravel(), (cand / b_scale).ravel()
                    cosine = _finite(
                        float(
                            np.clip(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)), -1, 1)
                        )
                    )
            else:
                max_abs = max_rel = mean_abs = rmse = cosine = None
            worst = None
            if count:
                scores = np.where(
                    mismatch, np.where(np.isfinite(difference), difference, np.inf), -1
                )
                index = np.unravel_index(int(np.argmax(scores)), ref.shape)
                worst = {
                    "index": [int(v) for v in index],
                    "reference": _sample(ref[index]),
                    "candidate": _sample(cand[index]),
                    "threshold": _finite(threshold[index]),
                    "abs_error": _finite(difference[index]),
                }
                reasons.append(f"{name}: {count} elements exceed tolerance/nonfinite policy")
        metrics[name] = {
            "max_abs": max_abs,
            "max_rel": max_rel,
            "mean_abs": mean_abs,
            "rmse": rmse,
            "cosine_similarity": cosine,
            "total_elements": int(ref.size),
            "mismatched_elements": count,
            "nonfinite_elements": int(np.count_nonzero(~finite)),
            "worst": worst,
        }
    return {
        "status": "FAIL" if reasons else "PASS",
        "metrics": metrics,
        "reason": "; ".join(reasons) if reasons else None,
    }
