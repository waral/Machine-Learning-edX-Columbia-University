from __future__ import division
import numpy as np
import matplotlib.pyplot as plt

data = np.genfromtxt('ratings.csv', delimiter=",")

# not using timestamp
X = np.delete(data[1:], 3, axis=1)
print(X.shape)
print(X)
print(np.amax(X[:, 1]))
print('max min rating: ', np.amax(X[:, 2]), np.amin(X[:, 2]))


sort_objects = np.unique(X[:, 1])
sort_users = np.unique(X[:, 0])

for i in range(X.shape[0]):
    X[i, 0] = np.where(sort_users == int(X[i, 0]))[0][0]
    X[i, 1] = np.where(sort_objects == int(X[i, 1]))[0][0]

print(X.shape)
print(X)
# user and object max ids
user_max = int(np.amax(X[:, 0]))  # user max index
object_max = int(np.amax(X[:, 1]))  # object max index
print('Max user id: ', user_max)
print('Max movie id: ', object_max)


#shuffling data
np.random.shuffle(X)
print(X)
print(X.shape)

#setting train, val, test sets
X_train = X[:60000]
X_val = X[60000:80000]
X_test = X[80000:]
print(X_train.shape, X_val.shape, X_test.shape)

def check_error(X, u, v):

    err = 0
    N = X.shape[0]
    for k in range(N):
        user_id = int(X[k, 0])
        object_id = int(X[k, 1])
        rating = int(X[k, 2])
        predict_rating = np.dot(u[user_id, :], v[object_id, :])
        err += np.absolute(predict_rating - rating)

    av_err = err / float(N)
    print('Average error: ', av_err)

    return av_err

def PMF(train_data, val_data, user_max_id, object_max_id, iterations=2, lam=2, sigma2=0.1, d=10):

    length = train_data.shape[0]
    mean = np.zeros(d)
    cov = (1/float(lam))*np.identity(d)
    L = np.zeros(iterations) #objective function
    Nu = user_max_id + 1 #int(np.amax(train_data[:, 0])) #user max index
    Nv = object_max_id + 1#int(np.amax(train_data[:, 1])) #object max index
    Mes = np.zeros((Nu, Nv)) #measured
    M = np.zeros((Nu, Nv)) #matrix of ratings
    train_err_list = []
    val_err_list = []

    for k in range(length):
        i = int(train_data[k, 0]) #- 1 maybe not needed if we index starting from 0
        j = int(train_data[k, 1]) #- 1
        Mes[i, j] = 1 #user ui rated movie vj
        M[i, j] = train_data[k, 2] #setting rating

    ##initialize locations and users
    u = np.zeros((iterations, Nu, d))
    v = np.zeros((iterations, Nv, d))
    v[0, :, :] = np.random.multivariate_normal(mean, cov, Nv) #initialize v as multivariate normal

    for k in range(iterations):
        print('Iteration: ', k+1, ' / ', iterations)

        ##update user location
        if k == 0:
            l = 0
        else:
            l = k-1

        for i in range(Nu):
            A = lam * sigma2 * np.identity(d)
            vec = np.zeros(d)
            for j in range(Nv):
                if Mes[i, j] == 1:
                    A += np.outer(v[l, j, :], v[l, j, :])
                    vec += M[i, j]*v[l, j, :]
            u[k, i, :] = np.dot(np.linalg.inv(A), vec)

        ##update object location
        for j in range(Nv):
            A = lam * sigma2 * np.identity(d)
            vec = np.zeros(d)
            for i in range(Nu):
                if Mes[i, j] == 1:
                    A += np.outer(u[k, i, :], u[k, i, :])
                    vec += M[i, j]*u[k, i, :]
            v[k, j, :] = np.dot(np.linalg.inv(A), vec)

        ##update objective function
        for i in range(Nu):
            for j in range(Nv):
                if Mes[i, j] == 1:
                    L[k] -= np.square(M[i, j] - np.dot(u[k, i, :].T, v[k, j, :]))
        L[k] = (1/(2*sigma2))*L[k]
        L[k] -= (lam/float(2))*(np.square(np.linalg.norm(u[k, :, :])) + np.square(np.linalg.norm(v[k, :, :])))
        print('Loss: ', L[k])

        print('Training set:')
        train_err_list.append(check_error(train_data, u[k, :, :], v[k, :, :]))
        print('Validation set:')
        val_err_list.append(check_error(val_data, u[k, :, :], v[k, :, :]))

    return L, u, v, train_err_list, val_err_list

iterations = 1
count = 1
best_lam = -1
best_sigma2 = -1
best_av_err_val = 100
best_train_err_list = None
best_val_err_list = None

for i in range(count):
    print('Parameter iteration: ', i+1, ' / ', count)
    lam = 10**np.random.uniform(-1.5, 1.5)
    sigma2 = 10**np.random.uniform(-1.5, 0.5)
    print('lam:',lam, ' sigma2:', sigma2)
    L, u_matrices, v_matrices, train_err_list, val_err_list = PMF(X_train, X_val, user_max, object_max,
                                                                  iterations=iterations, lam=lam, sigma2=sigma2, d=10)
    u = u_matrices[iterations-1, :, :]
    v = v_matrices[iterations-1, :, :]
    correct_train = 0

    # append training set error

    av_err_train = train_err_list[iterations-1]

    # append validation set error

    av_err_val = val_err_list[iterations-1]

    if av_err_val < best_av_err_val:
        best_av_err_val = av_err_val
        best_lam = lam
        best_sigma2 = sigma2
        best_train_err_list = train_err_list
        best_val_err_list = val_err_list

# best_lam = 4.34
# best_sigma2 = 0.8
iterations = 3
print('best_lam:', best_lam, ' best_sigma2:', best_sigma2)
print('Best validation set error: ', best_av_err_val)
L, u_matrices, v_matrices, train_err_list, val_err_list = PMF(X_train, X_val, user_max, object_max,
                                                                  iterations=iterations, lam=lam, sigma2=sigma2, d=10)

L = -L
plt.subplot(2, 1, 1)
plt.title('Training loss')
plt.xlabel('Iteration')
plt.plot(L, '-o')

plt.subplot(2, 1, 2)
plt.title('Training and validation error')
plt.xlabel('Iteration')
plt.plot(train_err_list, '-o', label='training error')
plt.plot(val_err_list, '-o', label='validation error')

plt.gcf().set_size_inches(12, 12)
plt.show()

u = u_matrices[iterations-1, :, :]
v = v_matrices[iterations-1, :, :]
print('Test set: ')
check_error(X_test, u, v)


