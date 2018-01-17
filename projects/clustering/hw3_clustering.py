import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal

mean1 = np.array([0, 0])
cov1 = np.matrix([[10, 5], [5, 5]])
mean2 = np.array([7, 9])
cov2 = np.matrix([[10, 0], [0, 10]])
mean3 = np.array([5, -4])
cov3 = np.matrix([[5, 2], [2, 5]])
X = np.random.multivariate_normal(mean1, cov1, 1000)
Y = np.random.multivariate_normal(mean2, cov2, 800)
Z = np.random.multivariate_normal(mean3, cov3, 500)
X = np.concatenate((X, Y, Z))

red = np.array([1, 0, 0])
green = np.array([0, 1, 0])
blue = np.array([0, 0, 1])
COLOR_MAP = {0 : red, 1: green, 2: blue}

for n in range(X.shape[0]):
    plt.plot(X[n, 0], X[n, 1], 'ro', color='k', markersize=3)
plt.show()
K = 3 #number of clusters


def kmeans(X, iterations=10):
    col = X.shape[1] #number of coordinates of each vector
    row = X.shape[0] #number of given vectors
    mu = np.random.rand(K, col) #initialize centroids randomly
    c = np.zeros(row) #assigments to clusters
    dist = np.zeros(K) #auxiliary vector of distances to centroids

    for i in range(iterations):
        print('Iteration number ',i + 1,' / ',iterations)

        ##update cluster assigments
        for n in range(row):
            for k in range(K):
                dist[k] = np.linalg.norm(X[n, :] - mu[k, :])
            c[n] = np.argmin(dist)

        ##update centroids
        for k in range(K):
            sum_k = np.zeros(col)
            n_k = 0
            for n in range(row):
                if c[n] == k:
                    sum_k += X[n, :]
                    n_k += 1
            if n_k != 0:
                mu[k, :] = sum_k/float(n_k)
            else:
                mu[k, :] = np.random.rand(col)

    return mu, c

mu, c = kmeans(X)

# plot results
for n in range(X.shape[0]):
    plt.plot(X[n, 0], X[n, 1], 'ro', color=COLOR_MAP[c[n]], markersize=3)

for k in range(K):
    plt.plot(mu[k, 0], mu[k, 1], 'x', color='k', markersize=7)
plt.show()

def gmm(X, iterations=10):
    col = X.shape[1] #number of coordinates of each vector
    row = X.shape[0] #number of given vectors
    mu = np.random.rand(K, col) #initialize centroids randomly
    sigma = np.zeros((K, col, col))
    phi = np.zeros((row, K)) #assigments to clusters

    ## normalized pi
    pi = np.ones(K)/float(K)

    ## make random sigmas identity
    for k in range(K):
        sigma[k, :, :] = np.identity(col)

    for i in range(iterations):
        print('Iteration number ', i + 1, ' / ', iterations)

        ## E-step
        for n in range(row):
            sum_aux = 0
            for k in range(K):
                sum_aux += pi[k]*multivariate_normal.pdf(X[n, :], mu[k, :], sigma[k, :, :])
            for k in range(K):
                if sum_aux == 0:
                    phi[n, k] = pi[k]/float(K)
                else:
                    phi[n, k] = pi[k]*multivariate_normal.pdf(X[n, :], mu[k, :], sigma[k, :, :])/sum_aux

        ## M-step
        for k in range(K):
            n_k = 0
            mu_sum = np.zeros(col)
            sigma_sum = np.zeros((col, col))
            for n in range(row):
                n_k += phi[n, k]
                mu_sum += phi[n, k]*X[n, :]
                vec = X[n, :] - mu[k, :]
                sigma_sum += phi[n, k]*np.outer(vec, vec)
            pi[k] = float(n_k)/float(row)
            if n_k == 0:
                mu[k, :] = np.random.rand(col)
                sigma[k, :, :] = np.identity(col)
            else:
                mu[k, :] = mu_sum/float(n_k)
                sigma[k, :, :] = sigma_sum/float(n_k)


    return pi, mu, sigma, phi

pi, mu, sigma, phi = gmm(X)

# plot results
    for n in range(X.shape[0]):
        colour = phi[n, 0] * COLOR_MAP[0] + phi[n, 1] * COLOR_MAP[1] + phi[n, 2] * COLOR_MAP[2]
        plt.plot(X[n, 0], X[n, 1], 'ro', color=colour, markersize=2)
    plt.show()




