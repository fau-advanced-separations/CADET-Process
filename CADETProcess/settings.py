"""
=======================================
Settings (:mod:`CADETProcess.settings`)
=======================================

.. currentmodule:: CADETProcess.settings

This module provides functionality for general settings.

.. autosummary::
    :toctree: generated/

    Settings

"""  # noqa

import os
import shutil
import tempfile
from pathlib import Path
from warnings import warn

from CADETProcess.dataStructure import Bool, Structure, Switch

__all__ = ["Settings"]


class Settings(Structure):
    """
    A class for managing general settings.

    Attributes
    ----------
    working_directory : str or None
        The path of the working directory. If None, the current directory is used.
    save_log : bool
        Whether to save log files or not.
    temp_dir : str or None
        The path of the temporary directory.
        If None, a directory named "tmp" is created in the working directory.
    debug_mode : bool
        Whether to enable debug mode or not.
    LOG_LEVEL : str
        The log level to use.
        Must be one of 'DEBUG', 'INFO', 'WARNING', 'ERROR', or 'CRITICAL'.

    Methods
    -------
    delete_temporary_files()
        Deletes the temporary simulation files.
    """

    _save_log = Bool(default=False)
    debug_mode = Bool(default=False)
    LOG_LEVEL = Switch(
        valid=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
    )

    def __init__(self) -> None:
        """Initialize Settings Object."""
        self._temp_dir = None
        self.working_directory = None

    @property
    def working_directory(self) -> str:
        """
        The path of the working directory.

        If the working directory is not set, the current directory is used.

        Raises
        ------
        TypeError
            If the working directory is not a string or None.

        Returns
        -------
        pathlib.Path
            The absolute path of the working directory.
        """
        if self._working_directory is None:
            _working_directory = Path("./")
        else:
            _working_directory = Path(self._working_directory)

        _working_directory = _working_directory.absolute()

        _working_directory.mkdir(exist_ok=True, parents=True)

        return _working_directory

    @working_directory.setter
    def working_directory(self, working_directory: str) -> None:
        self._working_directory = working_directory

    def set_working_directory(self, working_directory: str) -> None:
        """Set working directory."""
        warn(
            "This function is deprecated, use working_directory property.",
            DeprecationWarning,
            stacklevel=2,
        )
        self.working_directory = working_directory

    @property
    def save_log(self) -> bool:
        """bool: If True, save log files."""
        return self._save_log

    @save_log.setter
    def save_log(self, save_log: bool) -> None:
        from CADETProcess import log

        log.update_loggers(self.log_directory, save_log)

        self._save_log = save_log

    @property
    def log_directory(self) -> str:
        """pathlib.Path: Log directory."""
        return self.working_directory / "log"

    @property
    def temp_dir(self) -> str:
        """pathlib.Path: Directory for temporary files."""
        if self._temp_dir is None:
            if "XDG_RUNTIME_DIR" in os.environ:
                _temp_dir = Path(os.environ["XDG_RUNTIME_DIR"]) / "CADET-Process"
            else:
                _temp_dir = self.working_directory / "tmp"
        else:
            _temp_dir = Path(self._temp_dir).absolute()

        _temp_dir.mkdir(exist_ok=True, parents=True)
        tempfile.tempdir = _temp_dir.as_posix()

        return Path(tempfile.gettempdir())

    @temp_dir.setter
    def temp_dir(self, temp_dir: str) -> None:
        self._temp_dir = temp_dir

    def delete_temporary_files(self) -> None:
        """Delete the temporary files directory."""
        shutil.rmtree(self.temp_dir / "simulation_files", ignore_errors=True)
        self.temp_dir = self._temp_dir
