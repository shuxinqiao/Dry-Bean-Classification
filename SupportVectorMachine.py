from re import X
from tkinter import W
import numpy as np

N_class = 7
alpha = 0.1      # learning rate
batch_size = 1100    # batch size
MaxEpoch = 100        # Maximum epoch
reg_strength = 1000


def train(X_train, t_train, X_val, t_val):
    
    N_train = X_train.shape[0]
    N_feature = X_train.shape[1]

    w = np.zeros(N_feature)

    for epoch in range(MaxEpoch):

        dist = 1 - t_train * (np.dot(X_train, w))
        dist[dist < 0] = 0
        h_loss = reg_strength * (np.sum(dist) / N_train)

        loss_grad_w = np.zeros(len(w))
        for ind, d in enumerate(dist):
            if max(0,d) == 0:
                loss_grad_w += w
            else:
                loss_grad_w -= (reg_strength * t_train[ind] * X_train[ind])

        loss_grad_w = loss_grad_w / len(t_train)
        w -= alpha * loss_grad_w

        loss = (0.5) * np.dot(w,w) + h_loss

        print('----Epoch {epoch}----\nLoss = {loss}'.format(epoch=epoch, loss=loss))




