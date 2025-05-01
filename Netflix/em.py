"""Mixture model for matrix completion"""
from typing import Tuple
import numpy as np
from scipy.special import logsumexp
from common import GaussianMixture
import common

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment

    """

    n, d = X.shape
    K = mixture.mu.shape[0]
    
    post = np.zeros((n, K))
    mask = X != 0  # shape (n, d) — 1 for observed, 0 for missing

    for k in range(K):
        mu_k = mixture.mu[k]        # (d,)
        var_k = mixture.var[k]      # scalar
        p_k = mixture.p[k]          # scalar

        # Compute squared distance only on observed features
        diff = (X - mu_k) * mask    # broadcasted difference with masking
        squared_diff = np.sum(diff**2, axis=1)  # (n,)
        num_obs = np.sum(mask, axis=1)          # number of observed dims per example

        # Gaussian log-likelihood for observed features only
        log_prob = -0.5 * (num_obs * np.log(2 * np.pi * var_k) + squared_diff / var_k)
        post[:, k] = np.exp(log_prob) * p_k

    # Normalize posteriors
    total = np.sum(post, axis=1, keepdims=True)
    total[total == 0] = 1e-16  # avoid division by zero
    post /= total

    # Log-likelihood
    log_likelihood = np.sum(np.log(total))

    return post, log_likelihood



def mstep(X: np.ndarray, post: np.ndarray, mixture: GaussianMixture,
          min_variance: float = .25) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data, with incomplete entries (set to 0)
        post: (n, K) array holding the soft counts
            for all components for all examples
        mixture: the current gaussian mixture
        min_variance: the minimum variance for each gaussian

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    n, d = X.shape
    K = mixture.mu.shape[0]
    mask = (X != 0).astype(float)             # (n, d) indicator of observed entries

    # Prepare new parameters
    mu_new = np.zeros_like(mixture.mu)        # (K, d)
    var_new = np.zeros_like(mixture.var)      # (K,)
    p_new  = np.zeros_like(mixture.p)         # (K,)

    for k in range(K):
        post_k = post[:, k]                   # (n,)
        N_k = post_k.sum()                    # effective weight of component k

        # 1) Mixing weight
        p_new[k] = N_k / n

        # 2) Means: weighted sums & weighted observation‐counts per dim
        #    S_k[i] = Σ_j post[j,k] * X[j,i] * mask[j,i]
        #    W_k[i] = Σ_j post[j,k] * mask[j,i]
        S_k = (post_k[:, None] * X * mask).sum(axis=0)        # shape (d,)
        W_k = (post_k[:, None] * mask).sum(axis=0)           # shape (d,)

        # start with old mu, then overwrite dims with enough support
        mu_k = mixture.mu[k].copy()
        support = (W_k >= 1.0)        # only update coords with ≥1 total weight
        mu_k[support] = S_k[support] / W_k[support]
        mu_new[k] = mu_k

        # 3) Variance: spherical, using all observed entries
        #    Σ = Σ_{j,i obs} post[j,k] * (X[j,i] - mu_k[i])^2 
        diff = (X - mu_k) * mask                            # zero out missing
        sq = diff**2
        weighted_sq = (post_k[:, None] * sq).sum()          # scalar
        total_obs = (post_k[:, None] * mask).sum()          # total “obs‐weight”

        if total_obs > 0:
            var_k = weighted_sq / total_obs
        else:
            var_k = mixture.var[k]

        # floor the variance
        var_new[k] = max(var_k, min_variance)

    return GaussianMixture(mu=mu_new, var=var_new, p=p_new)
def run1(X: np.ndarray, mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """Runs the mixture model

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the current assignment
    """
    X_copy = X.copy()

    prev_ll = -np.inf
    ll = None

    # safety cap on iterations
    max_iter = 100

    for _ in range(max_iter):
    
        # E‑step: recompute posteriors and get new log‑likelihood
        post, ll = estep(X_copy, mixture)

        # M‑step: fit new parameters to current posteriors
        mixture = mstep(X_copy, post, mixture)

        # check convergence: Δℓ ≤ tol * |ℓ|
        if prev_ll != -np.inf and (ll - prev_ll) <= 1e-6 * abs(ll):
            break

        prev_ll = ll

    return mixture, post, ll


def run(X: np.ndarray,
        mixture: GaussianMixture,
        post: np.ndarray) -> Tuple[GaussianMixture, np.ndarray, float]:
    """
    Runs EM until convergence.

    Convergence when improvement in log‑likelihood ≤ 1e‑6 * |new_ll|.
    Does not mutate the caller’s X.
    """
    # copy so we never overwrite X in place
    X_copy = X.copy()

    prev_ll = -np.inf

    # do up to max_iter EM iterations
    for _ in range(100):
        # E‑step
        post, ll = estep(X_copy, mixture)

        # check convergence: Δℓ ≤ tol * |ℓ|
        if prev_ll != -np.inf and (ll - prev_ll) <= 1e-6 * abs(ll):
            break

        prev_ll = ll

        # M‑step
        mixture = mstep(X_copy, post, mixture)

    # one final E‑step on the converged mixture
    post, ll = estep(X_copy, mixture)
    return mixture, post, ll

def fill_matrix(X: np.ndarray, mixture: GaussianMixture) -> np.ndarray:
    """Fills an incomplete matrix according to a mixture model

    Args:
        X: (n, d) array of incomplete data (incomplete entries =0)
        mixture: a mixture of gaussians

    Returns
        np.ndarray: a (n, d) array with completed data
    """
    
    
    n, d = X.shape
    K    = mixture.mu.shape[0]
    mask = (X != 0).astype(float)

    log_w = np.zeros((n, K))
    for k in range(K):
        mu_k  = mixture.mu[k]
        var_k = mixture.var[k]

        # avoid log(0) on zero‐weight clusters
        if mixture.p[k] > 0:
            log_pk = np.log(mixture.p[k])
        else:
            log_pk = -np.inf

        diff    = (X - mu_k) * mask
        sq_dist = (diff**2).sum(axis=1)
        n_obs   = mask.sum(axis=1)

        log_gauss = -0.5*(n_obs*np.log(2*np.pi*var_k) + sq_dist/var_k)
        log_w[:, k] = log_pk + log_gauss

    # 2) stable normalization via log‐sum‐exp
    max_log_w = log_w.max(axis=1, keepdims=True)  # (n,1)
    W = np.exp(log_w - max_log_w)                 # still (n,K)
    post = W / W.sum(axis=1, keepdims=True)        # (n,K) = p(k|x_obs)

    # 3) reconstruct every row by the posterior‐weighted mean
    #    X_recon[n,i] = sum_k post[n,k] * mu[k,i]
    X_recon = post.dot(mixture.mu)                 # (n,d)

    # 4) fill only the missing entries
    X_pred = X.copy()
    missing = (X_pred == 0)
    X_pred[missing] = X_recon[missing]

    return X_pred



X = np.loadtxt("netflix_incomplete.txt")

#K = [1, 2, 3, 4,12]
K=[12]
seed = [0, 1, 2, 3,4]
mx=None
for k in K:

    best_cost = -np.inf
    best_seed = None
    for s in seed:
        mixture, post = common.init(X, k, s)
        # Run EM
        mixture, post, cost = run(X, mixture, post)

        if cost > best_cost:
            best_cost = cost
            best_seed = s
            mx=mixture

    print(f"EM>>> Best for K={k}: seed={best_seed}, cost={best_cost:.4f}")

X_gold = np.loadtxt('netflix_complete.txt') 
X_pred = fill_matrix(X_gold, mx)
print(common.rmse(X_gold, X_pred))