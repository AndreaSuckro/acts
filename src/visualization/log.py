import datetime


def log_results(epochs, losses, *, log_path='.'):
    """
    Logs the epochs and their loss values to a text file.

    """
    file = open(log_path + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')+'_perf.txt', 'w+')
    for ep, loss in zip(epochs, losses):
        file.write(f'Epoch: {ep} has loss {loss} \n')
    file.close()
