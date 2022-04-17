import matplotlib.pyplot as plt
import numpy as np
import os


# plot directory
pyfile_dir = os.path.dirname(__file__)
plot_dir = os.path.join(pyfile_dir, 'plots/')


# regular time series plot
def plot(list, filename, title, x_axis, y_axis):
    plt.figure()
    plt.plot(list)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.savefig(plot_dir + filename)

    return 0


# Transform one int to row one hot vector(matrix)
def onehot(x):
    N_class = np.max(x) + 1
    onehot_matrix = np.zeros((len(x),N_class))
    for i in range(len(x)):
        onehot_matrix[i,x[i]] = 1
        
    return onehot_matrix