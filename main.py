import numpy as np
import pandas as pd
import utils
import SoftmaxRegression
import SupportVectorMachine
from NeuralNetwork import Network, Layer, FCLayer, ActivationLayer, tanh, tanh_prime, mse, mse_prime


#-------------------------
# Process Data
#-------------------------
def processData():
    # Read data
    panda_excel = pd.read_excel('Dry_Bean_Dataset.xlsx', engine='openpyxl')
    dataset = np.array(panda_excel)

    # shuffle dataset
    random_indices = np.arange(len(dataset))
    np.random.shuffle(random_indices)
    dataset = dataset[random_indices]

    # Split features and type
    data_features = dataset[:,:-1]
    data_type = dataset[:,-1]#.reshape(dataset.shape[0],1)

    type_names = ['Seker', 'Barbunya', 'Bombay', 'Cali', 'Horoz', 'Sira', 'Dermason']

    for bean_type in type_names:
        data_type = np.where(data_type == bean_type.upper(), type_names.index(bean_type), data_type)


    # numpy float type
    data_features = data_features.astype(np.float64)

    # Normalization 
    data_features = (data_features - data_features.mean(axis=0)) / data_features.std(axis=0)

    # adding all one bias feature
    data_features = np.concatenate(
            (data_features, np.ones((data_features.shape[0], 1))), axis=1)



    print("---- Data Process ----")
    print("RAW data shape:", dataset.shape)
    print("Feature data shape:", data_features.shape)
    print("Type data shape:", data_type.shape)


    # split data
    # Train : Val : Test approx.= 11 : 1 : 1.3
    X_train = data_features[:11000]
    t_train = data_type[:11000]

    X_val = data_features[11000:12000]
    t_val = data_type[11000:12000]

    X_test = data_features[12000:]
    t_test = data_type[12000:]

    return X_train, t_train, X_val, t_val, X_test, t_test


#-------------------------
# Softmax Regression
#-------------------------
def softmax(X_train, t_train, X_val, t_val, X_test, t_test):

    print('\n---- Softmax Regression ----\n')

    softmax_epoch_best, softmax_acc_best,  softmax_W_best, softmax_train_losses, softmax_valid_accs =\
         SoftmaxRegression.train(X_train, t_train, X_val, t_val)

    utils.plot(softmax_train_losses, "Softmax-TrainLoss", "Softmax - Train Loss", "Epoch", "Loss")
    utils.plot(softmax_valid_accs, "Softmax-ValAcc", "Softmax - Validatino Accuracy", "Epoch", "Accuracy")

    softmax_acc_test = SoftmaxRegression.predict(X_test, softmax_W_best, t_test)

    print('\nTest accuracy: {}'.format(softmax_acc_test))

    return 0


#-------------------------
# Neural Network
#-------------------------
def ANN(X_train, t_train, X_val, t_val, X_test, t_test):
    
    print('\n---- Neural Network ----')

    X_train = X_train.reshape(X_train.shape[0],1,16)
    t_train = t_train.flatten()
    print(X_train.shape)

    net = Network()
    net.add(FCLayer(16, 100))               
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(100, 50))                   
    net.add(ActivationLayer(tanh, tanh_prime))
    net.add(FCLayer(50, 7))                    
    net.add(ActivationLayer(tanh, tanh_prime))

    net.use(mse, mse_prime)
    net.fit(X_train, t_train, epochs=35, learning_rate=0.3)

    out = net.predict(X_test[:10])
    print(out)
    print(t_test[:10].flatten())
    

#-------------------------
# Support Vector Machine
#-------------------------
def SVM(X_train, t_train, X_val, t_val, X_test, t_test):
    print('\n---- Support Vector Machine ----\n')

    SVM_epoch_best, SVM_acc_best, SVM_W_best, SVM_train_losses, SVM_valid_accs = \
        SupportVectorMachine.train(X_train, t_train, X_val, t_val)

    utils.plot(SVM_train_losses, "SVM-TrainLoss", "SVM - Train Loss", "Epoch", "Loss")
    utils.plot(SVM_valid_accs, "SVM-ValAcc", "SVM - Validatino Accuracy", "Epoch", "Accuracy")

    SVM_acc_test = SupportVectorMachine.predict(X_test, SVM_W_best, t_test)

    print('\nTest accuracy: {}'.format(SVM_acc_test))

    return 0


#-------------------------
# Main Function
#-------------------------
def main():
    # process data

    X_train, t_train, X_val, t_val, X_test, t_test = processData()

    # train models
    #softmax(X_train, t_train, X_val, t_val, X_test, t_test)

    SVM(X_train, t_train, X_val, t_val, X_test, t_test)

    #ANN(X_train, t_train, X_val, t_val, X_test, t_test)
    




if __name__ == '__main__':
    main()
