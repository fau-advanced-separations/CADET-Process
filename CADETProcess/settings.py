import logging
import os

import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.parent

project_directory = ('./')


LOG_PATH = os.path.join('logs/')
LOG_LEVEL = getattr(logging, 'INFO')
