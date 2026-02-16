"""Minimal smoke script with synthetic data only (no private inputs)."""

import pandas as pd

from analyse import compute_zone_summary


if __name__ == "__main__":
    toy_pairs = pd.DataFrame(
        {
            "method": ["dero", "dero", "amsa", "amsa", "ajuste", "ajuste"],
            "zone": ["avant", "arriere", "avant", "arriere", "avant", "arriere"],
            "dose_manuel": [1.0, 1.2, 0.9, 1.5, 1.1, 1.4],
            "dose_method": [1.1, 1.0, 0.95, 1.35, 1.15, 1.5],
        }
    )
    toy_pairs["diff"] = toy_pairs["dose_method"] - toy_pairs["dose_manuel"]
    toy_pairs["abs_error"] = toy_pairs["diff"].abs()
    toy_pairs["rel_error"] = toy_pairs["abs_error"] / toy_pairs["dose_manuel"].abs()

    summary = compute_zone_summary(toy_pairs, n_bootstrap=100)

    print("Smoke run OK (synthetic data)")
    print(summary[["method", "zone", "n_pairs", "median_abs_error", "spearman_r"]])
