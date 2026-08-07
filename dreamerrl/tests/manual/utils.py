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


def summarize_metrics(metrics_list):
    curves = {
        "total_loss": np.stack([m.total_loss.item() for m in metrics_list]),
        "recon_loss": np.stack([m.recon_loss.item() for m in metrics_list]),
        "reward_loss": np.stack([m.reward_loss.item() for m in metrics_list]),
        "cont_loss": np.stack([m.cont_loss.item() for m in metrics_list]),
        "kl_dyn": np.stack([m.kl_dyn.item() for m in metrics_list]),
        "kl_rep": np.stack([m.kl_rep.item() for m in metrics_list]),
    }

    summary = {}
    for key, arr in curves.items():
        mean = arr.mean()
        std = arr.std()
        cv = safe_cv(mean, std)
        summary[key] = (mean, std, cv)

    return summary
