import numpy as np
import pandas as pd
import utils
import SoftmaxRegression
import SupportVectorMachine
import NeuralNetwork


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
    data_features = np.concatenate((data_features, np.ones((data_features.shape[0], 1))), axis=1)



    print("---- Data Process ----\n")
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
# Majority Guess
#-------------------------
def majorGuess(t_test):
    print('\n---- Majority Guess ----\n')

    # Dermason - Class 7 - order 6
    print('Test Accuracy: {}'.format(np.mean(t_test == 6)))


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
# Neural Network
#-------------------------
def NN(X_train, t_train, X_val, t_val, X_test, t_test):
    print('\n---- Neural Network ----\n')
    

    X_train = np.concatenate((X_train, X_val), axis=0)
    t_train = np.concatenate((t_train, t_val), axis=0)
    
    t_train = utils.onehot(t_train)
    t_test = utils.onehot(t_test)


    history = NeuralNetwork.train(X_train, t_train, X_test, t_test)

    NN_train_losses = history.history['loss']
    NN_valid_accs = history.history['val_accuracy']

    utils.plot(NN_train_losses, "NN-TrainLoss", "NN - Train Loss", "Epoch", "Loss")
    utils.plot(NN_valid_accs, "NN-ValAcc", "NN - Validatino Accuracy", "Epoch", "Accuracy")




    
#-------------------------
# Main Function
#-------------------------
def main():
    # process data

    X_train, t_train, X_val, t_val, X_test, t_test = processData()

    # Majority Guess
    majorGuess(t_test)

    # train models
    softmax(X_train, t_train, X_val, t_val, X_test, t_test)

    SVM(X_train, t_train, X_val, t_val, X_test, t_test)

    NN(X_train, t_train, X_val, t_val, X_test, t_test)
    




if __name__ == '__main__':
    main()
