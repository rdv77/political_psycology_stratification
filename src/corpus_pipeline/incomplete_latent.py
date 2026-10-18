"""Observed-entry Gaussian FA (Gibbs) and ridge matrix factorization.

No API, imputation library, or corpus access. NaN is unobserved, zero is data.
Models use the same original rating units, not sparse-data standard deviations.
The Bayesian model has proper, explicit priors; it is not a MAP estimate called
Bayesian. Rotation is a reporting convention, not identification by the data.
"""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.linalg import orthogonal_procrustes
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist
from scipy.stats import norm, rankdata, spearmanr


def validate(x, q):
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or np.isinf(x).any() or not 1 <= q < min(x.shape):
        raise ValueError("Need a finite/NaN matrix and 1 <= q < min(n, p)")
    if not np.isfinite(x).any():
        raise ValueError("No observed data")
    return x


def gaussian_precision_sample(precision, h, rng):
    """Batched N(P^-1 h, P^-1); noise uses the TRANSPOSED Cholesky."""
    chol = np.linalg.cholesky(precision)
    mean = np.linalg.solve(precision, h[..., None])[..., 0]
    noise = np.linalg.solve(np.swapaxes(chol, -1, -2),
                            rng.normal(size=h.shape)[..., None])[..., 0]
    return mean + noise


def ridge_factorization(x, q, penalty=1., seed=0, max_iter=1500, tolerance=1e-8):
    """Minimize .5 SSE_observed + .5 lambda(||U||²+||L||²)
    + .5/2.25 ||mu-2.5||² by exact alternating ridge updates.

    Working zeros are always multiplied by the observation mask. An empty row
    has zero internal U (penalty optimum), but is NaN in published coordinates.
    """
    x = validate(x, q)
    if penalty <= 0:
        raise ValueError("penalty must be positive")
    mask = np.isfinite(x)
    values = np.where(mask, x, 0.)
    rng = np.random.default_rng(seed)
    u = rng.normal(0, .15, (len(x), q))
    load = rng.normal(0, .15, (x.shape[1], q))
    mu = np.full(x.shape[1], 2.5)
    prior_precision = np.diag([1 / 2.25] + [penalty] * q)
    prior_h = np.array([2.5 / 2.25] + [0.] * q)
    history = []
    for iteration in range(max_iter):
        precision = np.einsum('ij,jk,jl->ikl', mask, load, load) + penalty * np.eye(q)
        h = np.where(mask, values - mu, 0.) @ load
        u = np.linalg.solve(precision, h[..., None])[..., 0]
        design = np.column_stack([np.ones(len(x)), u])
        precision = np.einsum('ij,ik,il->jkl', mask, design, design) + prior_precision
        h = values.T @ design + prior_h
        coefficients = np.linalg.solve(precision, h[..., None])[..., 0]
        mu, load = coefficients[:, 0], coefficients[:, 1:]
        residual = np.where(mask, values - mu - u @ load.T, 0.)
        objective = .5 * (residual ** 2).sum() + .5 * penalty * (
            (u ** 2).sum() + (load ** 2).sum()) + .5 / 2.25 * ((mu - 2.5) ** 2).sum()
        history.append(float(objective))
        if len(history) > 1 and abs(history[-2] - objective) < tolerance * (1 + abs(history[-2])):
            break
    return {"prediction": mu + u @ load.T, "scores": u, "loadings": load,
            "mean": mu, "objective": float(objective), "history": history,
            "iterations": iteration + 1, "converged": iteration + 1 < max_iter,
            "penalty": penalty}


def best_ridge(x, q, penalty=1., starts=4, seed=0):
    fits = [ridge_factorization(x, q, penalty, seed + k) for k in range(starts)]
    best = min(fits, key=lambda fit: fit['objective'])
    best['start_objectives'] = [fit['objective'] for fit in fits]
    best['all_starts_converged'] = all(fit['converged'] for fit in fits)
    return best


@dataclass(frozen=True)
class Priors:
    mean: float = 2.5
    mean_sd: float = 1.5
    loading_sd: float = .75
    noise_shape: float = 3.
    noise_scale: float = 1.


def gibbs_factor_analysis(x, q, *, warmup=1200, draws=2000, chains=4,
                          seed=20261018, priors=Priors()):
    """y_ij ~ N(mu_j + f_i'L_j, psi_j), ONLY for observed ij.

    f_i ~ N(0,I); mu_j ~ N(2.5,1.5²); L_jk ~ N(0,loading_sd²/q);
    psi_j ~ InverseGamma(shape=3, scale=1). Conditional posteriors are
    Gaussian (scores, intercept+loadings jointly) and inverse gamma (noise).
    No draws of absent ratings enter any likelihood or parameter update.
    """
    x = validate(x, q)
    if warmup < 0 or draws < 4 or chains < 1 or min(priors.mean_sd, priors.loading_sd,
                                                   priors.noise_shape, priors.noise_scale) <= 0:
        raise ValueError("Invalid sampling settings or proper priors")
    n, p = x.shape
    mask = np.isfinite(x)
    values = np.where(mask, x, 0.)
    result = {key: [] for key in ('scores', 'loadings', 'mean', 'noise')}
    prior_precision = np.diag([1 / priors.mean_sd ** 2] + [q / priors.loading_sd ** 2] * q)
    prior_h = np.array([priors.mean / priors.mean_sd ** 2] + [0.] * q)
    for chain in range(chains):
        rng = np.random.default_rng(seed + 100003 * chain)
        u = rng.normal(size=(n, q))
        load = rng.normal(0, priors.loading_sd / np.sqrt(q), (p, q))
        mu = rng.normal(priors.mean, priors.mean_sd, p)
        initial_noise = priors.noise_scale / (priors.noise_shape - 1) if priors.noise_shape > 1 else priors.noise_scale / (priors.noise_shape + 1)
        psi = np.full(p, initial_noise)
        saved = {key: [] for key in result}
        for iteration in range(warmup + draws):
            precision = np.einsum('ij,jk,jl->ikl', mask / psi, load, load) + np.eye(q)
            h = (np.where(mask, values - mu, 0.) / psi) @ load
            u = gaussian_precision_sample(precision, h, rng)
            design = np.column_stack([np.ones(n), u])
            precision = np.einsum('ij,ik,il->jkl', mask, design, design) / psi[:, None, None]
            precision += prior_precision
            h = (values.T @ design) / psi[:, None] + prior_h
            coefficients = gaussian_precision_sample(precision, h, rng)
            mu, load = coefficients[:, 0], coefficients[:, 1:]
            residual = np.where(mask, values - mu - u @ load.T, 0.)
            shape = priors.noise_shape + mask.sum(axis=0) / 2
            scale = priors.noise_scale + (residual ** 2).sum(axis=0) / 2
            psi = 1 / rng.gamma(shape, 1 / scale)
            if iteration >= warmup:
                for key, value in (('scores', u), ('loadings', load), ('mean', mu), ('noise', psi)):
                    saved[key].append(value.copy())
        for key in result:
            result[key].append(np.asarray(saved[key]))
    return {key: np.asarray(value) for key, value in result.items()}


def _split(a):
    half = a.shape[1] // 2
    return np.concatenate([a[:, :half], a[:, -half:]], axis=0)


def _rhat(a):
    within = np.var(a, axis=1, ddof=1).mean(axis=0)
    between = a.shape[1] * np.var(a.mean(axis=1), axis=0, ddof=1)
    variance = (a.shape[1] - 1) / a.shape[1] * within + between / a.shape[1]
    ratio = np.divide(variance, within, out=np.full_like(variance, np.inf), where=within > 0)
    ratio[(within == 0) & (variance == 0)] = 1.
    return np.sqrt(ratio)


def chain_diagnostics(a):
    """Rank-normalized split/folded Rhat and initial-positive-monotone bulk ESS.

    Input chains x draws x scalar monitors. Monitor invariant covariance and
    predictions, NOT unidentified raw factor labels. ESS is capped at total N.
    """
    if a.shape[0] < 2:
        return {"max_rank_split_rhat": None, "min_bulk_ess": None}
    s = _split(a)
    c, n, k = s.shape
    flat = s.reshape(c * n, k)
    z = norm.ppf((rankdata(flat, axis=0) - .375) / (c * n + .25)).reshape(s.shape)
    folded = abs(flat - np.median(flat, axis=0))
    fz = norm.ppf((rankdata(folded, axis=0) - .375) / (c * n + .25)).reshape(s.shape)
    rhats = np.maximum(_rhat(z), _rhat(fz))
    centered = z - z.mean(axis=1, keepdims=True)
    fft = np.fft.rfft(centered, n=2 * n, axis=1)
    acov = np.fft.irfft(fft * fft.conj(), n=2 * n, axis=1)[:, :n].real / n
    w = acov[:, 0].mean(axis=0) * n / (n - 1)
    var_plus = (n - 1) / n * w + np.var(z.mean(axis=1), axis=0, ddof=1)
    rho = 1 - (w - acov.mean(axis=0)) / np.maximum(var_plus, 1e-30)
    rho[0] = 1
    pairs = rho[:2 * (n // 2)].reshape(n // 2, 2, k).sum(axis=1)
    positive = np.cumprod(pairs > 0, axis=0).astype(bool)
    monotone = np.minimum.accumulate(pairs, axis=0)
    tau = np.maximum(1., -1 + 2 * np.where(positive, monotone, 0.).sum(axis=0))
    ess = np.minimum(c * n, c * n / tau)
    return {"max_rank_split_rhat": float(np.max(rhats)),
            "rhat_over_1_01": int((rhats > 1.01).sum()), "monitor_count": k,
            "min_bulk_ess": float(ess.min()), "median_bulk_ess": float(np.median(ess)),
            "draws_per_chain": a.shape[1], "chains": a.shape[0]}


def canonical_axes(scores, loadings):
    """Orthogonal varimax; sort by loading norm, sign by largest indicator."""
    from scripts.latent_space_benchmark import varimax_rotation
    rotation = varimax_rotation(loadings)
    load = loadings @ rotation
    order = np.argsort(-np.sum(load ** 2, axis=0))
    rotation = rotation[:, order]
    load = loadings @ rotation
    signs = np.sign(load[np.argmax(abs(load), axis=0), np.arange(load.shape[1])])
    signs[signs == 0] = 1
    rotation *= signs
    return scores @ rotation, loadings @ rotation, rotation


def summarize_bayesian(draws, mask):
    fs, ls, ms, ps = (draws[k] for k in ('scores', 'loadings', 'mean', 'noise'))
    shared = np.einsum('ctik,ctjk->ctij', fs, ls)
    covariance = np.einsum('ctjk,ctlk->ctjl', ls, ls)
    prediction = (shared + ms[:, :, None, :]).mean(axis=(0, 1))
    # Diagnostics retain invariant scales and cross-feature covariance.
    tri = np.triu_indices(ls.shape[2])
    monitors = np.concatenate([ms, ps, covariance[:, :, tri[0], tri[1]],
                               shared[:, :, mask]], axis=-1)
    diagnostics = chain_diagnostics(monitors)
    # Orientation is post-processing of the full posterior, never a likelihood
    # constraint. Generalized Procrustes only removes orthogonal gauge freedom.
    l = ls.reshape(-1, *ls.shape[2:])
    f = fs.reshape(-1, *fs.shape[2:])
    ev, vectors = np.linalg.eigh(covariance.mean(axis=(0, 1)))
    q = l.shape[-1]
    reference = vectors[:, -q:] * np.sqrt(np.maximum(ev[-q:], 0))
    aligned_l = np.empty_like(l)
    aligned_f = np.empty_like(f)
    for _ in range(3):
        for i in range(len(l)):
            rotation, _ = orthogonal_procrustes(l[i], reference)
            aligned_l[i] = l[i] @ rotation
            aligned_f[i] = f[i] @ rotation
        reference = aligned_l.mean(axis=0)
    score, load, rotation = canonical_axes(aligned_f.mean(axis=0), reference)
    aligned_f = aligned_f @ rotation
    aligned_l = aligned_l @ rotation
    return {"prediction": prediction, "scores": score, "loadings": load,
            "mean": ms.mean(axis=(0, 1)), "noise": ps.mean(axis=(0, 1)),
            "score_lower": np.quantile(aligned_f, .025, axis=0),
            "score_upper": np.quantile(aligned_f, .975, axis=0),
            "loading_lower": np.quantile(aligned_l, .025, axis=0),
            "loading_upper": np.quantile(aligned_l, .975, axis=0),
            "covariance_lower": np.quantile(covariance.reshape(-1, *covariance.shape[2:]), .025, axis=0),
            "covariance_upper": np.quantile(covariance.reshape(-1, *covariance.shape[2:]), .975, axis=0),
            "diagnostics": diagnostics,
            "aligned_intervals_are_orientation_conditional": True}


def prediction_mean(draws):
    return (np.einsum('ctik,ctjk->ctij', draws['scores'], draws['loadings'])
            + draws['mean'][:, :, None, :]).mean(axis=(0, 1))


def holdout_mask(x, target_mask, seed, fraction=.2):
    """Hide exact scores, preserving >=1 row and >=2 column observations.
    Ranges remain training assumptions, never validation targets.
    """
    rng = np.random.default_rng(seed)
    available = np.isfinite(x)
    train = available.copy()
    target = np.zeros(x.shape, dtype=bool)
    candidates = np.argwhere(target_mask & available)
    rng.shuffle(candidates)
    desired = max(1, round(len(candidates) * fraction))
    for i, j in candidates:
        if train[i].sum() > 1 and train[:, j].sum() > 2:
            target[i, j] = True
            train[i, j] = False
        if target.sum() >= desired:
            break
    if not target.any():
        raise ValueError("No identifiable validation cells under overlap constraints")
    return target


def validation_folds(x, target_mask, seed, folds=5):
    """Disjoint exact-cell folds, each eligible target tested exactly once.

    Singleton rows cannot be tested without turning them into entirely new
    unmeasured actors. Greedy capacity allocation preserves >=1 row / >=2 col
    training entries; unallocatable cells are returned explicitly.
    """
    available = np.isfinite(x)
    row_n, col_n = available.sum(1), available.sum(0)
    eligible = target_mask & available & (row_n[:, None] > 1) & (col_n[None, :] > 2)
    rng = np.random.default_rng(seed)
    candidates = np.argwhere(eligible)
    rng.shuffle(candidates)
    targets = [np.zeros(x.shape, bool) for _ in range(folds)]
    for i, j in candidates:
        options = [k for k in range(folds) if targets[k][i].sum() < row_n[i] - 1
                   and targets[k][:, j].sum() < col_n[j] - 2]
        if options:
            k = min(options, key=lambda k: (targets[k].sum(), k))
            targets[k][i, j] = True
    covered = np.logical_or.reduce(targets)
    return targets, {'eligible': int(eligible.sum()), 'tested_once': int(covered.sum()),
                     'eligible_not_allocated': int((eligible & ~covered).sum()),
                     'ineligible_exact_cells': int((target_mask & available & ~eligible).sum())}


def prediction_metrics(truth, prediction):
    difference = np.asarray(prediction) - np.asarray(truth)
    return {"rmse": float(np.sqrt(np.mean(difference ** 2))),
            "mae": float(np.mean(abs(difference))), "n": len(difference)}


def compare_geometry(reference, candidate, row_mask=None, feature_mask=None):
    """Signed/permuted axis match and rotation-invariant pairwise distances.
    No unrestricted Procrustes in axis-stability metric (would inflate it).
    """
    # Relative geometry of a numerically collapsed ridge solution is arbitrary
    # floating-point residue; correlations of those residues are meaningless.
    for fitted in (reference, candidate):
        if fitted.get('effective_rank_threshold_0_01') == 0:
            return {'status': 'undefined_collapsed_latent_signal',
                    'worst_axis_congruence': None, 'mean_axis_congruence': None,
                    'distance_spearman': None}
    a, b = reference['loadings'], candidate['loadings']
    if feature_mask is not None:
        a, b = a[feature_mask], b[feature_mask]
    denom = np.linalg.norm(a, axis=0)[:, None] * np.linalg.norm(b, axis=0)[None, :]
    congruence = np.divide(a.T @ b, denom, out=np.zeros_like(denom), where=denom > 1e-10)
    rows, cols = linear_sum_assignment(-abs(congruence))
    sa = reference['scores'] if row_mask is None else reference['scores'][row_mask]
    sb = candidate['scores'] if row_mask is None else candidate['scores'][row_mask]
    d1, d2 = pdist(sa), pdist(sb)
    correlation = float(spearmanr(d1, d2).statistic) if np.std(d1) > 1e-10 and np.std(d2) > 1e-10 else None
    return {"status": "descriptive_geometry_comparison",
            "worst_axis_congruence": float(np.min(abs(congruence[rows, cols]))),
            "mean_axis_congruence": float(np.mean(abs(congruence[rows, cols]))),
            "distance_spearman": correlation}
