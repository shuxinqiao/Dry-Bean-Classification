from msilib import type_valid
from matplotlib.pyplot import axis
import numpy as np
import pandas as pd
import SoftmaxRegression
import SupportVectorMachine
from NeuralNetwork import Network, Layer, FCLayer, ActivationLayer, tanh, tanh_prime, mse, mse_prime


def processData():
    #-------------------------
    # Process Data
    #-------------------------
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
    print("type data shape:", data_type.shape)


    # split data
    # Train : Val : Test approx.= 11 : 1 : 1.3
    X_train = data_features[:11000]
    t_train = data_type[:11000]

    X_val = data_features[11000:12000]
    t_val = data_type[11000:12000]

    X_test = data_features[12000:]
    t_test = data_type[12000:]

    return X_train, t_train, X_val, t_val, X_test, t_test


def softmax(X_train, t_train, X_val, t_val, X_test, t_test):
    #-------------------------
    # Softmax Regression
    #-------------------------
    print('\n---- Softmax Regression ----')

    epoch_best, acc_best,  W_best, train_losses, valid_accs = SoftmaxRegression.train(X_train, t_train, X_val, t_val)

    acc_test = SoftmaxRegression.predict(X_test, W_best, t_test)

    print('\nTest accuracy: {}'.format(acc_test))



def ANN(X_train, t_train, X_val, t_val):
    #-------------------------
    # Neural Network
    #-------------------------
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
    


def SVM(X_train, t_train, X_val, t_val):
    #-------------------------
    # Support Vector Machine
    #-------------------------
    print('\n---- Support Vector Machine ----')
    SupportVectorMachine.train(X_train, t_train, X_val, t_val)

    return 0


def main():
    X_train, t_train, X_val, t_val, X_test, t_test = processData()
    #softmax(X_train, t_train, X_val, t_val, X_test, t_test)
    #ANN(X_train, t_train, X_val, t_val)
    SVM(X_train, t_train, X_val, t_val)




if __name__ == '__main__':
    main()
