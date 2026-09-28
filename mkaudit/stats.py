"""Statistics helpers: bootstrap confidence intervals, exact McNemar test, Benjamini-Hochberg FDR and
age-partial Spearman correlation (paper Secs. 1.3 and 1.6)."""
import numpy as np
from scipy.stats import spearmanr, binomtest, t as tdist
from sklearn.linear_model import LinearRegression


def bootstrap_ci(fn, a, b, n=1000, seed=0):
    """95% percentile CI of fn(a, b) over n paired bootstrap resamples (segment-level Spearman CI of the skill models)."""
    rng = np.random.default_rng(seed); vals = []
    a = np.asarray(a); b = np.asarray(b)
    for _ in range(n):
        i = rng.integers(0, len(a), len(a))
        try:
            vals.append(fn(a[i], b[i]))
        except Exception:
            pass
    return float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))


def boot_ci(sub, fn, rng, B=2000):
    """95% percentile CI of fn(rows) over B bootstrap resamples of the rows of DataFrame `sub` (video-level
    resampling for the audit metrics). `rng` is a numpy Generator shared across calls, as in the paper scripts."""
    vals = [fn(sub.iloc[rng.integers(0, len(sub), len(sub))]) for _ in range(B)]
    return [float(np.nanpercentile(vals, 2.5)), float(np.nanpercentile(vals, 97.5))]


def mcnemar_exact(a, b):
    """Exact McNemar on paired booleans a (zero-shot pass) vs b (grounded pass).
    Returns (n01 = only b correct, n10 = only a correct, two-sided p)."""
    n01 = int(np.sum(~a & b)); n10 = int(np.sum(a & ~b))
    return n01, n10, (1.0 if n01 + n10 == 0 else float(binomtest(min(n01, n10), n01 + n10, 0.5).pvalue))


def bh_fdr(p):
    """Benjamini-Hochberg adjusted q-values."""
    p = np.asarray(p, float); n = len(p); o = np.argsort(p); q = np.empty(n)
    run = 1.0
    for rank, i in list(enumerate(o, 1))[::-1]:
        run = min(run, p[i] * n / rank); q[i] = run
    return q


def partial_spearman(x, y, z):
    """Spearman between residuals of x and y after linear regression on z."""
    z = np.asarray(z, float).reshape(-1, 1)
    rx = np.asarray(x, float) - LinearRegression().fit(z, x).predict(z)
    ry = np.asarray(y, float) - LinearRegression().fit(z, y).predict(z)
    return float(spearmanr(rx, ry).correlation)


def partial_spearman_test(x, yv, z):
    """Partial Spearman of x and yv controlling for z, with a two-sided t-approximation p-value (df = n - 3)."""
    z = np.asarray(z, float).reshape(-1, 1)
    rx = np.asarray(x, float) - LinearRegression().fit(z, x).predict(z)
    ry = np.asarray(yv, float) - LinearRegression().fit(z, yv).predict(z)
    r = spearmanr(rx, ry).correlation
    n = len(rx); tt = r * np.sqrt((n - 3) / max(1e-12, 1 - r ** 2))
    return float(r), float(2 * tdist.sf(abs(tt), n - 3))


def partial_p_value(r, n):
    """Two-sided t-approximation p-value of a first-order partial correlation r from n observations."""
    return float(2 * tdist.sf(abs(r * np.sqrt((n - 3) / max(1e-12, 1 - r ** 2))), n - 3))
