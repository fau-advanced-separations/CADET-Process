import pathlib
ROOT_DIR = pathlib.Path(__file__).parent.parent

project_directory = ('./')

import logging
import os
LOG_PATH = os.path.join('logs/')
LOG_LEVEL = getattr(logging, 'INFO')

import git
try:
    repo = git.Repo(ROOT_DIR.parent)
    active_branch = str(repo.active_branch)
    sha = repo.head.object.hexsha
except git.InvalidGitRepositoryError:
    active_branch = ''
    sha = ''