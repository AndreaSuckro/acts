import matplotlib.pyplot as plt
import random
import numpy as np


def plot_loss(epochs, losses):
    """
    Plots the development of the loss value over time.
    
    :param epochs: the epoch value for the corresponding loss
    :param losses: the loss values
    :return: a pretty plot
    """
    fig, ax = plt.subplots()

    ax.plot(epochs, losses, '--', linewidth=2)
    plt.xlabel('# of epoch')
    plt.ylabel('loss')
    plt.title('Loss over the training')

    plt.show()


def plot_sample(train_data, train_label, number=1):
    """
    Plots a subplot with a random positive and negative case sample.
    
    :param train_data: training lung ct patch
    :param train_label: the label per patch
    :param number: the number of samples that should be plotted
    :return: a plot with a sample from both classes
    """

    no_tumor_idx = random.choice([i for i, x in enumerate(train_label) if x == 0])
    tumor_idx = random.choice([i for i, x in enumerate(train_label) if x == 1])

    plt.figure(1)
    plt.title('Tumor an non-Tumor Patch')
    plt.subplot(211)
    plt.imshow(np.array(train_data[tumor_idx]))

    plt.subplot(212)
    plt.imshow(np.array(train_data[no_tumor_idx]))
    plt.show()


if __name__ == "__main__":
    losses = [0.39044588804244995, 0.4199160933494568, 0.9145151972770691, 0.48902201652526855, 0.744167685508728,
              0.3442825675010681, 0.2639879584312439, 0.2950464189052582, 0.24532786011695862, 0.18398214876651764,
              0.13264437019824982, 0.10774071514606476, 0.2505808174610138, 0.0962160974740982, 0.20678400993347168,
              0.14474785327911377, 0.159656822681427, 0.20220847427845, 0.1157199889421463, 0.11056768894195557]
    epochs = []
    for i in range(0, 100, 5):
        epochs.append(i)
    plot_loss(epochs, losses)
