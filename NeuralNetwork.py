from keras.layers import Dense
from keras.models import Sequential


def train(X_train, t_train, X_test, t_test):
    

    N_feature = X_train.shape[1]
    N_class = t_train.shape[1]

    model = Sequential()

    model.add(Dense(units=64, activation='sigmoid'))
    model.add(Dense(units=N_class, activation='softmax'))
    model.build(input_shape=X_train.shape)
    model.summary()

    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics='accuracy')
    history = model.fit(X_train, t_train, batch_size=100, epochs=100, validation_split=0.1)
    loss, accuracy = model.evaluate(X_test, t_test)

    print('loss: {}'.format(loss))
    print('accuracy: {}'.format(accuracy))

    return history