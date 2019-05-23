import random
import numpy as np

def Split_dataset(x,y,n_label):
    n_each = n_label / 10
    isRun = True
    x_train = []
    y_train = []
    index = np.zeros(10)
    while(isRun):
        a = random.randint(0, np.shape(x)[0])
        x1 = x[a]
        y1 = y[a]
        if index[y1] < 10:
            x_train.append(x1)
            y_train.append(y1)
            index[y1] = index[y1]+1
        isOk1 = True
        for i in range(10):
            if index[i] < 10:
                isOk1 = False
        if isOk1:
            break
    for p in range(28):
        x_train.append(x_train[p])
        y_train.append(y_train[p])
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train,y_train