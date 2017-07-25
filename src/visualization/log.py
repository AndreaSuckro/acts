import datetime
import logging

def log_results(epochs, losses, *, log_path='.'):
    """
    Logs the epochs and their loss values to a text file.

    """
    file = open(log_path + datetime.datetime.today().strftime('%Y-%m-%d-%H-%M')+'_perf.txt', 'w+')
    for ep, loss in zip(epochs, losses):
        file.write(f'Epoch: {ep} has loss {loss} \n')
    file.close()


def log_args(func):
    """
    This decorator dumps out the keyword-arguments passed to a function before calling it
    :param func: the function to log 
    """
    logger = logging.getLogger()
    fname = func.__name__

    def echo_func(*args, **kwargs):
        logger.info(f'Calling `{fname}` with args: {kwargs}')
        return func(*args, **kwargs)

    return echo_func
