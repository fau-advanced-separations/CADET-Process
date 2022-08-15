import time

from functools import wraps
import os
import logging

import multiprocess

import pathos


LOG_FORMAT = logging.Formatter(
    '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

loggers = {}


def get_logger(name, level=None):
    try:
        logger = loggers[name]
    except KeyError:
        logger = pathos.logger()
        loggers[name] = logger

    if level is not None:
        level = getattr(logging, level)
        logger.setLevel(level)

    return logger


def update_loggers(log_directory, save_log):
    for name, logger in loggers.items():
        update_file_handlers(log_directory, logger, name, save_log)


def update_file_handlers(log_directory, logger, name, save_log):
    try:
        level = logger.handlers[0].level
    except IndexError:
        level = logger.level

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    if save_log:
        add_file_handler(log_directory, logger, name, level)


def add_file_handler(log_directory, logger, name, level, overwrite=False):
    log_directory = Path(log_directory)
    log_directory.mkdir(exist_ok=True)

    if overwrite:
        mode = 'w'
    else:
        mode = 'a'

    file_handler = logging.FileHandler(
        log_directory / f'{name}.log',
        mode=mode
    )
    file_handler.setFormatter(LOG_FORMAT)
    file_handler.setLevel(level)

    logger.addHandler(file_handler)


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
            logger.debug(f'Execution of {str(function)} took {elapsed} s')
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
            except Exception as e:
                # log the exception
                err = "There was an exception in "
                err += function.__name__
                logger.exception(err)

                # re-raise the exception
                raise e

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

            logger.debug('{} was called with {}, {}'.format(
                    function, *args, **kwargs))
            results = function(*args, **kwargs)
            logger.debug(f'Results: {results}')

            return results
        return wrapper
    return log_results_decorator
