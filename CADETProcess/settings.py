import logging
from pathlib import Path

from CADETProcess import CADETProcessError


LOG_LEVEL = getattr(logging, 'INFO')

working_directory = Path('./')
log_directory = working_directory / 'log'


def set_working_directory(directory='./', update_log=True, overwrite=True):
    global working_directory
    working_directory = Path(directory)

    global log_directory
    log_directory = working_directory / 'log'

    try:
        working_directory.mkdir(exist_ok=overwrite, parents=True)
        log_directory.mkdir(exist_ok=True, parents=True)
    except FileExistsError:
        raise CADETProcessError("Working directory already exists.")

    if update_log:

        from CADETProcess import log
        log.update_loggers()
