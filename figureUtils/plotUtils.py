import matplotlib.pyplot as plt
import numpy as np

def distribution_plot(data):
    uniques, counts = np.unique(data, return_counts=True)

    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(uniques, counts, '.')
    ax1.set_xlim([-10, data.max()])
    ax2.plot(uniques, np.log(counts), '*')
    ax2.set_xlim([-10, data.max()])
    plt.show()