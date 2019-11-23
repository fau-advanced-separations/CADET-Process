import os

import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.parent

import json
with open(os.path.join(ROOT_DIR, 'settings.json')) as json_data:
    settings = json.load(json_data)

project_directory = settings['project_directory']
cadet_path = settings['cadet_path']

import tempfile
tempfile.tempdir = settings['tempdir']

import logging
import os
LOG_PATH = os.path.join(settings['project_directory'], 'logs/')
LOG_LEVEL = getattr(logging, settings['LOG_LEVEL'])

import git
repo = git.Repo(ROOT_DIR.parent)
sha = repo.head.object.hexsha