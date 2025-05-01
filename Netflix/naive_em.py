"""Mixture model using EM"""
from typing import Tuple
import numpy as np
from common import GaussianMixture
import common

def estep(X: np.ndarray, mixture: GaussianMixture) -> Tuple[np.ndarray, float]:
    """E-step: Softly assigns each datapoint to a gaussian component

    Args:
        X: (n, d) array holding the data
        mixture: the current gaussian mixture

    Returns:
        np.ndarray: (n, K) array holding the soft counts
            for all components for all examples
        float: log-likelihood of the assignment
    """
    n, d = X.shape
    K = mixture.mu.shape[0]
    post = np.zeros((n, K))

    for k in range(K):
        mu_k = mixture.mu[k]
        var_k = mixture.var[k]
        p_k = mixture.p[k]

        # Mask: 1 if value is observed, 0 if missing (assuming 0 means missing)
        mask = X != 0
        diff = (X - mu_k) * mask  # broadcast subtraction and apply mask
        n_obs = mask.sum(axis=1)  # number of observed features for each point

        # Compute log-likelihood per point for this component
        log_prob = -0.5 * (n_obs * np.log(2 * np.pi * var_k) + (diff ** 2).sum(axis=1) / var_k)
        log_prob += np.log(p_k)

        post[:, k] = np.exp(log_prob)

    # Normalize posterior
    total = np.sum(post, axis=1, keepdims=True)
    total[total == 0] = 1e-16  # to prevent division by zero
    post /= total

    # Compute total log-likelihood
    log_likelihood = np.sum(np.log(total))

    return post, log_likelihood


def mstep(X: np.ndarray, post: np.ndarray) -> GaussianMixture:
    """M-step: Updates the gaussian mixture by maximizing the log-likelihood
    of the weighted dataset

    Args:
        X: (n, d) array holding the data
        post: (n, K) array holding the soft counts
            for all components for all examples

    Returns:
        GaussianMixture: the new gaussian mixture
    """
    
    min_variance = 1e-6
    n, d = X.shape
    K = post.shape[1]
    mask = X != 0  # shape (n, d)

    n_obs = post.sum(axis=0)  # (K,)
    p = n_obs / n

    mu = np.zeros((K, d))
    var = np.zeros(K)

    for k in range(K):
        weight = post[:, k].reshape(-1, 1)  # shape (n, 1)
        weighted_X = weight * X * mask  # element-wise mask
        mask_sum = (weight * mask).sum(axis=0)  # sum of weights for observed entries

        # Avoid division by zero
        mask_sum[mask_sum == 0] = 1e-16
        mu[k] = weighted_X.sum(axis=0) / mask_sum

        # Variance calculation
        diff = (X - mu[k]) * mask
        var[k] = np.sum(post[:, k] * (diff ** 2).sum(axis=1)) / np.sum(post[:, k] * mask.sum(axis=1))
        var[k] = max(var[k], min_variance)

    return GaussianMixture(mu=mu, var=var, p=p)


def run(X_1: np.ndarray, mixture: GaussianMixture,
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
    X= X_1.copy()
    prev_likelihood = -np.inf
    for _ in range(100):
        post, log_likelihood = estep(X, mixture)

        # M-step
        mixture = mstep(X, post)

        # Check for convergence
        if np.abs(log_likelihood - prev_likelihood) <= 1e-6 * np.abs(log_likelihood):
           break

        prev_likelihood = log_likelihood

    return mixture, post, log_likelihood




X = np.loadtxt("toy_data.txt")

K = [1, 2, 3, 4]
seed = [0, 1, 2, 3,4]
for k in K:

    best_cost = -np.inf
    best_seed = None
    for s in seed:
        mixture, post = common.init(X, k, s)
        # Run K-means
        mixture, post, cost = run(X, mixture, post)
        #print(f"K={k}, seed={s}, cost={cost}")
        # Plot the results
        #common.plot(X, mixture, post, f"K={k}, seed={s}")

        if cost > best_cost:
            best_cost = cost
            best_seed = s
        
    print(f"NAIVE EM>>> Best for K={k}: seed={best_seed}, cost={best_cost:.4f}")
