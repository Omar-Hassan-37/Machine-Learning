import pandas as  pd
import numpy as np
import matplotlib.pyplot as plot
import random


def predict(x, w, b):
    classification = np.dot(x, w) + b
    return classification

def cost_func(w, b, x, y):
    cf = y *(np.dot(x, w) + b)
    return cf

def fit(x, y, b):
    w = np.zeros(len(x[0])).reshape((1,2))
    wt = np.transpose(w)
    LR = 0.1
    lambda_df = 1 / 1000
    for i in range(1000):
        for j, x_term in enumerate(x):
            x_term1 = x_term.reshape((1,2))
            cost_f = cost_func(wt, b, x_term1, y[j])
            
            if cost_f >= 1:
                w -= LR * (2 * np.dot(lambda_df, w))
            else:
                w += LR * (np.dot(x_term1, y[j]) - 2 * np.dot(lambda_df, w))
                b += LR * y[j]
                
    return w, b


data = pd.read_csv('heart.csv', index_col = 0)

data1 = data.sample(frac = 1)
x = data1[['oldpeak','thal']].values.reshape([-1,2])
y = data1['target'].to_numpy()
x_train = x[:272]
y_train = y[:272]
x_test = x[272:]
y_test = y[272:]

color = data['target'].apply(lambda x: 'red' if x == 0 else 'green')
data.plot(kind='scatter', x='fbs', y='restecg', c=color)
plot.show()
np.array(x_train)
np.array(y_train)


b = 0
Y_pred = np.zeros(31)
w , b= fit(x_train, y_train, b)
wt = np.transpose(w)
for j, x_term in enumerate(x_test):
    x_term1 = x_term.reshape((1,2))
    Y_pred[j] = np.round(predict(x_term1, wt, b))

error = 0
for i in range(31):
    if(Y_pred[i] != y_test[i]):
        error += 1

print((31-error)/31)
