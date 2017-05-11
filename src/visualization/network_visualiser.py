import matplotlib.pyplot as plt
import random


def plot_loss(epochs, losses):
    """
    Plots the development of the loss value over time.
    
    :param epochs: the epoch value for the corresponding loss
    :param losses: the loss values
    :return: a pretty plot
    """
    fig, ax = plt.subplots()

    fig.plot(epochs, losses, '--', linewidth=2)
    plt.xlabel('# of epoch')
    plt.ylabel('loss')
    plt.title('Loss over the training')

    plt.show()


if __name__ == "__main__":
    losses = []
    epochs = []
    for i in range(20):
        epochs.append(i)
        losses.append(random.random())
    plot_loss(epochs, losses)
