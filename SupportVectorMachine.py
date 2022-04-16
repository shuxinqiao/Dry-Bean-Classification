import numpy as np

N_class = 7
reg_strength = 0.1      # regulization loss weight
delta = 1       # right classify score

alpha = 0.1      # learning rate
MaxEpoch = 100        # Maximum epoch

def train(X_train, t_train, X_val, t_val):
    # general parameter
    N_train = X_train.shape[0]
    N_feature = X_train.shape[1]

    train_losses = []
    valid_accs = []
    epoch_best = 0
    acc_best = 0

    w = np.zeros((N_feature, N_class))

    # training epoch loop
    for epoch in range(MaxEpoch):
        # loss gradient w.r.t weight init.
        loss_grad_w = np.zeros((N_feature, N_class))

        # y prediction based on current w
        y_hat = np.dot(X_train, w)

        # margin calculation
        margin = y_hat - y_hat[range(N_train), t_train.astype(int)].reshape((-1,1)) + delta
        margin[range(N_train), t_train.astype(int)] = 0     
        margin = (margin > 0) * margin      # max(0,d)

        # loss calculation
        # data loss + regulazation loss
        loss = 0
        loss += np.mean(margin) + reg_strength * np.sum(np.dot(w, w.T))

        # record loss
        train_losses.append(loss)
        
        count = (margin > 0).astype(int)
        count[range(N_train), t_train.astype(int)] = -np.sum(count, axis=1)

        # loss gradient plus hinge loss (regression regulazation)
        loss_grad_w += (np.dot(count.T, X_train) / N_train + 2 * reg_strength * w.T).T

        # weight update
        w -= alpha * loss_grad_w

        print('----Epoch {epoch}----\nLoss = {loss}'.format(epoch=epoch, loss=loss))
        

        # validation prediction
        acc = predict(X_val, w, t_val)

        # record accuracy
        valid_accs.append(acc)

        print('acc  = {acc}'.format(acc=acc))


        # find best epoch and return
        if acc > acc_best:
            acc_best = acc
            epoch_best = epoch
            W_best = w

    return epoch_best, acc_best, W_best, train_losses, valid_accs


# prediction function
def predict(X, W, t):
    y_hat = np.dot(X, W)
    y_pred = np.argmax(y_hat, axis=1)
    acc = np.mean(y_pred == t)

    return acc