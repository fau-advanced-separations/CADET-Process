"""
=================================
Logging (:mod:`CADETProcess.log`)
=================================

.. currentmodule:: CADETProcess.log

The CADETProcess.log module provides functionality for logging events in CADET-Process.

.. autosummary::
    :toctree: generated/

    loggers
    get_logger

"""

from functools import wraps
import logging
from pathlib import Path
import time

import pathos


__all__ = ['loggers', 'get_logger']


LOG_FORMAT = logging.Formatter(
    '%(asctime)s:%(levelname)s:%(name)s:%(message)s'
)

loggers = {}


def get_logger(name, level=None):
    """Retrieve logger from loggers dictionary. Create new one if it does not already exist.

    Parameters
    ----------
    name : str
        The name of the logger.
    level : str, optional
        The logging level to set on the logger.

    Returns
    -------
    logging.Logger
        The logger object.
    """
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
    """Update the file handlers of all logger objects in the loggers dictionary.

    Parameters
    ----------
    log_directory : str
        The directory to store the log files.
    save_log : bool
        If True, log files are saved. Otherwise, no files are saved.
    """
    for name, logger in loggers.items():
        update_file_handlers(log_directory, logger, name, save_log)


def update_file_handlers(log_directory, logger, name, save_log):
    """Update the file handlers of a logger object.

    Parameters
    ----------
    log_directory : str
        The directory to store the log files.
    logger : logging.Logger
        The logger object to update.
    name : str
        The name of the logger.
    save_log : bool
        If True, log files are saved. Otherwise, no files are saved.
    """
    try:
        level = logger.handlers[0].level
    except IndexError:
        level = logger.level

    for hdlr in logger.handlers:
        logger.removeHandler(hdlr)

    if save_log:
        add_file_handler(log_directory, logger, name, level)


def add_file_handler(log_directory, logger, name, level, overwrite=False):
    """Add a file handler to a logger object.

    Parameters
    ----------
    log_directory : str
        The directory to store the log files.
    logger : logging.Logger
        The logger object to update.
    name : str
        The name of the logger.
    level : str
        The logging level to set on the logger.
    overwrite : bool, optional
        If True, the log file is overwritten. Otherwise, logs are appended to the file.
    """
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
