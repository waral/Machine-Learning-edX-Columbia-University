from __future__ import division
import numpy as np
import pandas as pd

## data preprocessing
data = pd.read_csv('iris.csv', sep=",", header=None)

for n in range(data.shape[0]):
    if data.get_value(n, 4) == 'Iris-setosa':
        data.set_value(n, 4, 0)
    if data.get_value(n, 4) == 'Iris-versicolor':
        data.set_value(n, 4, 1)
    if data.get_value(n, 4) == 'Iris-virginica':
        data.set_value(n, 4, 2)

data = data.as_matrix()
print(data)
print(data.shape)

def pluginClassifier(X_train, y_train, X_test, classes=3):
    rows_test = X_test.shape[0]
    rows = X_train.shape[0]
    columns = X_train.shape[1] # aka number of classes
    proby = np.zeros((rows_test, classes)) # probabilities
    piy= np.zeros(classes) # mle for prior
    ny = np.zeros(classes) # number of elements in class
    muy = np.zeros((classes, columns)) # mle for average
    sigmay = np.zeros((columns, columns, classes)) # mle for covariance
    invy = np.zeros((columns, columns, classes)) # inverses of sigmay
    sqrty = np.zeros(classes) # sqrt of determinant of invy

    ## compute mle(maximum likelihood estimate) for prior and average
    for n in range(classes):
        sumx = np.zeros(X_train.shape[1])
        for k in range(rows):
            if y_train[k] == n: # or y_train[k] == n+1 depending on indexing
                ny[n] += 1
                sumx = sumx + X_train[k, :]
        piy[n] = ny[n]/rows
        muy[n, :] = (1/ny[n])*sumx


    ## compute mle for covariance
    for n in range(classes):
        for k in range(rows):
            if y_train[k] == n:
                aux = X_train[k, :] - muy[n]
                sigmay[:, :, n] = sigmay[:, :, n] + np.outer(aux, aux)
        sigmay[:, :, n] = (1/ny[n])*sigmay[:, :, n]
        invy[:, :, n] = np.linalg.inv(sigmay[:, :, n])
        sqrty[n] = np.sqrt(np.absolute(np.linalg.det(invy[:, :, n])))

    ## plug-in classifier: computing not rescale p(x, y)
    for k in range(rows_test):
        for n in range(classes):
            vec = X_test[k, :] - muy[n, :]
            aux = (-0.5)*np.dot(vec.T, np.dot(invy[:, :, n], vec))
            proby[k, n] = piy[n]*sqrty[n]*np.exp(aux)

    aux = np.sum(proby, axis=1)

    # rescaling probabilities p(x,y)
    for k in range(rows_test):
        for n in range(classes):
            proby[k, n] = proby[k, n]/aux[k]

    return proby

## classifying according to output probabilities several times for shuffled data
test_acc = 0
iterations = 100
for l in range(iterations):
    # print(l+1)
    np.random.shuffle(data)
    X_train = data[0:120, 0:4]
    y_train = data[0:120, 4]
    X_test = data[120:150, 0:4]
    y_test = data[120:150, 4]
    probabilities = pluginClassifier(X_train, y_train, X_test)
    rows_test = X_test.shape[0]
    y_out = np.zeros(rows_test)
    ## classifying according to the highest probability
    for k in range(rows_test):
        vec = probabilities[k, :]
        y_out[k] = np.argmax(vec)
        test_acc += np.sum(y_out[k]==y_test[k] )/30

print("Average accuracy: ", test_acc/iterations)