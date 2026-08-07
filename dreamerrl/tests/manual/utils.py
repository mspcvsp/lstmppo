import numpy as np


def safe_cv(mean, std):
    if np.allclose(mean, 0):
        return 0.0
    return std.mean() / abs(mean.mean())


def summarize_curve(curve_list):
    arr = np.stack(curve_list)
    mean = arr.mean(axis=0)
    std = arr.std(axis=0)
    cv = safe_cv(mean, std)
    return mean, std, cv


def summarize_metrics(results_per_seed):
    summary = {}

    keys = ["total_loss", "recon_loss", "reward_loss", "cont_loss", "kl_dyn", "kl_rep"]

    for key in keys:
        # Extract per-seed curves
        curves = []
        for seed_metrics in results_per_seed:
            curve = np.array([getattr(m, key).item() for m in seed_metrics])
            curves.append(curve)

        curves = np.stack(curves)  # shape: (num_seeds, steps)

        mean = curves.mean(axis=0)
        std = curves.std(axis=0)

        # CV across seeds
        cv = std.mean() / abs(mean.mean()) if not np.allclose(mean, 0) else 0.0

        summary[key] = (mean, std, cv)

    return summary
