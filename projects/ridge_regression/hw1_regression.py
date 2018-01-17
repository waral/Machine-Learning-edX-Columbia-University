import numpy as np

data = np.genfromtxt('wine_white.csv', delimiter=";")
print(data[1:])
print(data[1:].shape)

X_train = data[1:3200,0:11]
y_train = data[1:3200, 11]
X_val = data[3200:4000,0:11]
y_val = data[3200:4000, 11]
X_test = data[4000:, 0:11]
y_test = data[4000:, 11]

##Preprocessing for ridge regression
dev = np.std(X_train, axis=0)
mean = np.mean(X_train, axis=0)
y_mean = np.mean(y_train)

def preprocess(x):
    x_new = (x - mean)/dev
    return x_new

X_train = preprocess(X_train)
y_train = y_train - y_mean
X_val = preprocess(X_val)
X_test = preprocess(X_test)

# sanity check
print(np.mean(preprocess(X_train), axis=0))
print(np.std(preprocess(X_train), axis=0))

def rregression(x, y, lambda_input):
    aux = np.dot(x.T,x)
    n = aux.shape[0]
    return np.dot(np.dot(np.linalg.inv(lambda_input*np.identity(n) + aux), x.T), y)

# select best lambda
best_lam = 0
best_val_error = 1e10
iterations = 1000
for i in range(iterations):
    #print('Iteration ',i+1, ' / ', iterations)
    lam = 10**np.random.uniform(-5,5)
    wRR = rregression(X_train, y_train, lam)
    y_ans = np.dot(X_val, wRR) + y_mean
    val_error = np.mean(np.absolute(y_ans - y_val))
    #print('lambda: ', lam, ' validation accuracy: ', val_error)
    if val_error < best_val_error:
        best_lam = lam
        best_val_error = val_error

print('Best lambda: ', best_lam, ' validation accuracy: ', best_val_error)
wRR = rregression(X_train, y_train, best_lam)
y_ans = np.dot(X_test, wRR) + y_mean
test_error = np.mean(np.absolute(y_ans - y_test))
print('Test accuracy: ', test_error)



