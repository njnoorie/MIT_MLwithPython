import numpy as np
import kmeans
import common
import naive_em
import em

X = np.loadtxt("toy_data.txt")

"""
Read a 2D toy dataset using X = np.loadtxt('toy_data.txt'). 
Your task is to run the K-means algorithm on this data using the implementation we have provided in kmeans.py 
Initialize K-means using common.init(X, K, seed), where  is the number of clusters 
and seed is the random seed used to randomly initialize the parameters.
Try K=[1,2,3,4] on this data, plotting each solution using our common.plot function. 
Since the initialization is random, please use seeds  to and select the one that minimizes the total cost. 
Save the associated plots (best solution for each K)

"""


K = [1, 2, 3, 4]
seed = [0, 1, 2, 3,4]
for k in K:

    best_cost = -np.inf
    best_seed = None
    for s in seed:
        mixture, post = common.init(X, k, s)
        # Run K-means
        mixture, post, cost = kmeans.run(X, mixture, post)
        #print(f"K={k}, seed={s}, cost={cost}")
        # Plot the results
        #common.plot(X, mixture, post, f"K={k}, seed={s}")

        if cost > best_cost:
            best_cost = cost
            best_seed = s
        
    print(f"KMEANS>>> Best for K={k}: seed={best_seed}, cost={best_cost:.4f}")


# K = [1, 2, 3, 4]
# seed = [0, 1, 2, 3,4]
# arr=[]
# for k in K:
#     for s in seed:
#         mixture, post = common.init(X, k, s)
#         #run em
#         mixture, post, cost = naive_em.run(X, mixture, post)
#         print(f"K={k}, seed={s}, cost={cost}")
#         arr.append((cost,(k,s)))


K = [1, 2, 3, 4]
seed = [0, 1, 2, 3, 4]
arr = []

# Optional: Fill missing data if allowed
X_filled = X.copy()
col_means = X_filled.sum(axis=0) / (X_filled != 0).sum(axis=0)
X_filled[X_filled == 0] = np.take(col_means, np.where(X_filled == 0)[1])

for k in K:
    best_cost = -np.inf
    best_seed = None

    for s in seed:
        mixture, post = common.init(X_filled, k, s)
        mixture, post, cost = naive_em.run(X, mixture, post)
        #print(f"K={k}, seed={s}, cost={cost:.4f}")
        arr.append((cost, (k, s)))

        if cost > best_cost:
            best_cost = cost
            best_seed = s

    print(f"NAIVE>>> Best for K={k}: seed={best_seed}, cost={best_cost:.4f}")
    print(common.bic(X_filled, mixture, best_cost))




