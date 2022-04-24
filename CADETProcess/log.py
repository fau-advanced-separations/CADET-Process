import time

from functools import wraps
import os
import logging

import multiprocess


from CADETProcess import settings

LOG_FORMAT = logging.Formatter(
    '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

loggers = {}
log_to_file = {}


def get_logger(name, level=None, save_log=True):
    if level is None:
        level = settings.LOG_LEVEL

    try:
        logger = loggers[name]
    except KeyError:
        logger = multiprocess.get_logger()

    logger.setLevel(level)

    loggers[name] = logger

    if save_log:
        log_to_file[name] = logger

    return logger


def add_file_handler(logger, name, level, overwrite=False):
    if overwrite:
        mode = 'w'
    else:
        mode = 'a'

    file_handler = logging.FileHandler(
        os.path.join(settings.log_directory, name + '.log'),
        mode=mode
    )
    file_handler.setFormatter(LOG_FORMAT)
    file_handler.setLevel(level)

    logger.addHandler(file_handler)


def update_loggers():
    for name, logger in log_to_file.items():
        try:
            level = logger.handlers[0].level
        except IndexError:
            level = settings.LOG_LEVEL

        for hdlr in logger.handlers:
            logger.removeHandler(hdlr)

        add_file_handler(logger, name, level)


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
            logger.info(f'Execution of {str(function)} took {elapsed} s')
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
                err = "There was an exception in "
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
            logger.info(f'Results: {results}')

            return results
        return wrapper
    return log_results_decorator
