from pathlib import Path
import shutil
import tempfile
from warnings import warn

from CADETProcess.dataStructure import StructMeta
from CADETProcess.dataStructure import Bool, Switch


class Settings(metaclass=StructMeta):
    _save_log = Bool(default=False)
    debug_mode = Bool(default=False)
    LOG_LEVEL = Switch(
        valid=['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL'],
        default='INFO'
    )

    def __init__(self):
        self._temp_dir = None
        self.working_directory = None

    @property
    def working_directory(self):
        if self._working_directory is None:
            _working_directory = Path('./')
        else:
            _working_directory = Path(self._working_directory)

        _working_directory = _working_directory.absolute()

        _working_directory.mkdir(exist_ok=True, parents=True)

        return _working_directory

    @working_directory.setter
    def working_directory(self, working_directory):
        self._working_directory = working_directory

    def set_working_directory(self, working_directory):
        warn(
            'This function is deprecated, use working_directory property.',
            DeprecationWarning, stacklevel=2
        )
        self.working_directory = working_directory

    @property
    def save_log(self):
        return self._save_log

    @save_log.setter
    def save_log(self, save_log):
        from CADETProcess import log
        log.update_loggers(self.log_directory, save_log)

        self._save_log = save_log

    @property
    def log_directory(self):
        return self.working_directory / 'log'

    @property
    def temp_dir(self):
        if self._temp_dir is None:
            _temp_dir = self.working_directory / 'tmp'
        else:
            _temp_dir = Path(self._temp_dir).absolute()

        _temp_dir.mkdir(exist_ok=True, parents=True)
        tempfile.tempdir = _temp_dir.as_posix()

        return Path(tempfile.gettempdir())

    @temp_dir.setter
    def temp_dir(self, temp_dir):
        self._temp_dir = temp_dir

    def delete_temporary_files(self):
        shutil.rmtree(self.temp_dir / "simulation_files", ignore_errors=True)
        self.temp_dir = self._temp_dir
