from pathlib import Path
import tempfile
from warnings import warn

from CADETProcess import CADETProcessError
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
        self.temp_dir = None
        self.working_directory = './'

    @property
    def working_directory(self):
        return self._working_directory

    @working_directory.setter
    def working_directory(self, working_directory):
        working_directory = Path(working_directory).absolute()
        working_directory.mkdir(exist_ok=True, parents=True)
        self._working_directory = Path(working_directory)

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
        return tempfile.gettempdir()

    @temp_dir.setter
    def temp_dir(self, temp_dir=None):
        if temp_dir is not None:
            try:
                exists = Path(temp_dir).exists()
            except TypeError:
                raise CADETProcessError('Not a valid path')
            if not exists:
                raise CADETProcessError('Not a valid path')

        tempfile.tempdir = temp_dir
