import matplotlib.pyplot as plt
import numpy as np
import os


pyfile_dir = os.path.dirname(__file__)
plot_dir = os.path.join(pyfile_dir, 'plots/')


def plot(list, filename, title, x_axis, y_axis):
    plt.figure()
    plt.plot(list)
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.savefig(plot_dir + filename)

    return 0