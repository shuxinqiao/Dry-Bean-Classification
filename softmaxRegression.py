import numpy as np

# Global parameters
N_class = 7
alpha = 0.1      # learning rate
batch_size = 1100    # batch size
MaxEpoch = 100        # Maximum epoch


# prediction function
def predict(X, W, t):
    # X_new: Nsample x (d+1)
    # W: (d+1) x K

    #w = W[:-1,:]
    #b = W[-1,:].reshape(N_class,)

    count = 0
    step = 1
    for split in range(0,X.shape[0],step):
        z = np.dot(X[split:split+step], W) #+ b
        y = softmax(z)

        count += np.sum(t[split:split+step] == np.argmax(y, axis=1).reshape(step,1))

    acc = count / X.shape[0]

    return acc


# Transform one int to row one hot vector(matrix)
def onehot(x):
    onehot_matrix = np.zeros((len(x),N_class))
    for i in range(len(x)):
        onehot_matrix[i,x[i]] = 1
        
    return onehot_matrix


# Softmax calculation
def softmax(z):
    exp = np.exp(z - np.max(z, axis=1).reshape(z.shape[0],1))

    for i in range(len(z)):
        exp[i] /= np.sum(exp[i])

    return exp


# Training function (including validation)
def train(X_train, y_train, X_val, t_val):
    # general parameters
    N_train = X_train.shape[0]
    N_val = X_val.shape[0]

    N_feature = X_train.shape[1]


    # variables
    train_losses = []
    valid_accs = []
    epoch_best = 0
    acc_best = 0
    W_best = 0

    w = np.ones((N_feature,N_class))
    b = np.ones(N_class)


    # Training epoch loop
    for epoch in range(MaxEpoch):
        
        # Training - gradient descent
        step = batch_size
        for split in range(0,N_train,step):
            z = np.dot(X_train[split:split+step], w)# + b
            y = softmax(z)
            y_hot = onehot(y_train[split:split+step])#.T

            loss_grad_w = np.dot(X_train[split:split+step].T,(y - y_hot)) * (1/step)
            loss_grad_b = np.mean(y - y_hot, axis=0)#.reshape(N_class,1)

            w = w - alpha * loss_grad_w
            #b = b - alpha * loss_grad_b


        # loss calculation (prevent memory problem by editing step)
        loss = 0
        step = N_train
        for split in range(0,N_train,step):
            z = np.dot(X_train[split:split+step], w)# + b
            y = softmax(z)
            y_hot = onehot(y_train[split:split+step])

            loss += -np.sum(np.log(np.diag(y.dot(y_hot.T)) + 1e-16))
        
        loss = (loss / N_train)
        train_losses.append(loss)

        print('----Epoch {epoch}----\nLoss = {loss}'.format(epoch=epoch, loss=loss))
        

        # validation
        count = 0
        step = 1
        for split in range(0,N_val,step):
            z = np.dot(X_val[split:split+step], w)# + b
            y = softmax(z)

            count += np.sum(t_val[split:split+step] == np.argmax(y, axis=1).reshape(step,1))

        acc = count / N_val
        valid_accs.append(acc)
        
        print('acc  = {acc}'.format(acc=acc))

        
        # find best epoch and return
        if acc > acc_best:
            acc_best = acc
            epoch_best = epoch
            W_best = w#np.append(w,b.reshape(1,N_class),axis = 0)
        
    return epoch_best, acc_best, W_best, train_losses, valid_accs
