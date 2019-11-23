from CADETProcess.common import settings

from functools import wraps
import os
import logging

LOG_FORMAT = logging.Formatter('%(asctime)s:%(levelname)s:%(name)s:%(message)s')

def get_logger(name, level=None, log_directory=None):
    if level is None:
        level = settings.LOG_LEVEL

    logger = logging.getLogger(name)

    if log_directory is None:
        log_directory = os.path.join(settings.project_directory, 'log')
    else:
        log_directory = os.path.join(settings.project_directory, log_directory)

    if not os.path.exists(log_directory):
            os.makedirs(log_directory)

    file_handler = logging.FileHandler(os.path.join(log_directory, name + '.log'))
    file_handler.setFormatter(LOG_FORMAT)
    file_handler.setLevel(level)

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)
    logger.addHandler(file_handler)

    logger.setLevel(logging.DEBUG)

    return logger

import time
def log_time(logger_name, level=None):
    """Log execution time of function.

    Parameters
    ----------
    logger_name : str
        name of the logger
    """
    def log_time_decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            start = time.time()
            result = function(*args, **kwargs)
            elapsed = time.time() - start
            logger = get_logger(logger_name, level=None)
            logger.info('Execution of {} took {} s'.format(
                    str(function), elapsed))
            return result
        return wrapper
    return log_time_decorator

def log_exceptions(logger_name, level=None):
    """Log exceptions.

    Parameters
    ----------
    logger_name : str
        name of the logger
    """
    def log_exception_decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name, level=None)
            try:
                return function(*args, **kwargs)
            except:
                # log the exception
                err = "There was an exception in  "
                err += function.__name__
                logger.exception(err)

                # re-raise the exception
                raise
        return wrapper
    return log_exception_decorator

def log_results(logger_name, level=None):
    """Log results.

    Parameters
    ----------
    logger_name : str
        name of the logger
    """
    def log_results_decorator(function):
        @wraps(function)
        def wrapper(*args, **kwargs):
            logger = get_logger(logger_name, level=None)

            logger.info('{} was called with {}, {}'.format(
                    function, *args, **kwargs))
            results = function(*args, **kwargs)
            logger.info('Results: {}'.format(results))

            return results
        return wrapper
    return log_results_decorator
