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


# K = [1, 2, 3, 4]
# seed = [0, 1, 2, 3,4]
# for k in K:
#     for s in seed:
#         mixture, post = common.init(X, k, s)
#         # Run K-means
#         mixture, post, cost = kmeans.run(X, mixture, post)
#         print(f"K={k}, seed={s}, cost={cost}")
#         # Plot the results
#         #common.plot(X, mixture, post, f"K={k}, seed={s}")



K = [1, 2, 3, 4]
seed = [0, 1, 2, 3,4]
arr=[]
for k in K:
    for s in seed:
        mixture, post = common.init(X, k, s)
        #run em
        mixture, post, cost = em.run(X, mixture, post)
        print(f"K={k}, seed={s}, cost={cost}")
        arr.append((cost,(k,s)))




