import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

data = pd.read_csv('house_data.csv', index_col = 0)

x = data['sqft_living'].values.reshape([-1,1])
y = data['price'].values.reshape([-1,1])

X_train = x[:17292]
Y_train = y[:17292]

X_test = x[17292:]
Y_test = y[17292:]

c1 = np.random.normal()
c2 = np.random.normal()
LR = 0.000000001
m = 1000
print(c1, c2)

for i in range(m):
    Y_pred = getY_pred(c1, c2, X_train)
    c1 = c1 - LR * getC0(Y_pred, Y_train)
    c2 = c2 - LR * getC1(Y_pred, X_train, Y_train)
    mse = getMse(Y_pred, Y_train)
    print(mse)
    
print(c1, c2)

Y_pred = c1 + c2 * X_test 
mse = getMse(Y_pred, Y_test)
print(mse)  
print(c1, c2)
plt.xlabel("X")
plt.ylabel("Y")
plt.title("Real vs predicted values")
plt.scatter(X_test,Y_test)
plt.plot(mse)


def getY_pred(c1, c2, x):
    mul = np.dot(c2, x)
    summ = np.add(c1, mul)
    return summ

def getC0(y_pred, y):
    sub = y_pred - y
    div = (np.sum(sub))
    return np.divide(np.sum(sub), len(y))

def getC1(y_pred, x, y):
    sub = np.subtract(y_pred,y)
    mul = np.multiply(sub, x)
    return np.divide(np.sum(mul), len(y))

def getMse(y_pred, y):
    sub = np.subtract(y_pred,y)
    power = np.square(sub)
    summ = np.sum(power)
    return np.divide(summ , 2 * len(y))
