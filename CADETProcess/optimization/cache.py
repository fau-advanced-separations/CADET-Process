from collections import defaultdict
from pathlib import Path
import shutil
import tempfile

from diskcache import Cache

from CADETProcess import CADETProcessError
from CADETProcess.dataStructure import DillDisk


class ResultsCache():
    """Cache to store (intermediate) results.

    Optinally uses diskcache library to store large objects in sqlite database.

    Internal structure:
    [evaluation_object][step][x]

    For example:
    [EvaluationObject 1][Evaluator 1][x] -> IntermediateResults 1
    [EvaluationObject 1][Evaluator 2][x] -> IntermediateResults 2
    [EvaluationObject 1][Objective 1][x] -> f1.1
    [EvaluationObject 1][Objective 2][x] -> f1.2
    [EvaluationObject 1][Constraint 1][x] -> g1.1

    [EvaluationObject 2][Evaluator 1][x] -> IntermediateResults 1
    [EvaluationObject 2][Evaluator 2][x] -> IntermediateResults 2
    [EvaluationObject 2][Objective 1][x] -> f2.1
    [EvaluationObject 2][Objective 2][x] -> f2.2
    [EvaluationObject 2][Constraint 1][x] -> g2.1

    [None][Evaluator 1][x] -> IntermediateResults 1
    [Objective 3][x] -> f3
    [Constraint 2][x] -> g2

    See Also
    --------
    CADETProcess.optimization.OptimizationProblem.add_evaluator
    """

    def __init__(self, use_diskcache=False, directory=None):
        self.use_diskcache = use_diskcache
        self.directory = directory
        self.init_cache()

        self.tags = defaultdict(list)

    def init_cache(self):
        """Initialize ResultsCache."""
        if self.use_diskcache:
            if self.directory is None:
                self.directory = Path(tempfile.mkdtemp(prefix='diskcache-'))

            if self.directory.exists():
                shutil.rmtree(self.directory, ignore_errors=True)

            self.directory.mkdir(exist_ok=True, parents=True)

            self.cache = Cache(
               self.directory.as_posix(),
               disk=DillDisk,
               disk_min_file_size=2**18,    # 256 kb
               size_limit=2**36,            # 64 GB
            )
            self.directory = self.cache.directory
        else:
            self.cache = {}

    def set(self, key, value, tag=None, close=True):
        """Add entry to cache.

        Parameters
        ----------
        key : hashable
            The key to retrieve the results for.
        value : object
            The value corresponding to the key.
        tag : str, optional
            Tag to associate with result. The default is None.
        """
        if self.use_diskcache:
            self.cache.set(key, value, tag=tag, expire=None)
        else:
            if tag is not None:
                self.tags[tag].append(key)
            self.cache[key] = value

        if close:
            self.close()

    def get(self, key, close=True):
        """Get entry from cache.

        Parameters
        ----------
        key : hashable
            The key to retrieve the results for.

        Returns
        -------
        value : object
            The value corresponding to the key.
        """
        value = self.cache[key]

        if close:
            self.close()

        return value

    def delete(self, key, close=True):
        """Remove entry from cache.

        Parameters
        ----------
        key : hashable
            The key to retrieve the results for.
        value : object
            The value corresponding to the key.

        """
        if self.use_diskcache:
            found = self.cache.delete(key)
            if not found:
                raise KeyError(key)
        else:
            self.cache.pop(key)

        if close:
            self.close()

    def prune(self, tag='temp'):
        """Remove tagged entries from cache.

        Parameters
        ----------
        tag : str, optional
            Tag to be removed. The default is 'temp'.
        """
        if self.use_diskcache:
            self.cache.evict(tag)
        else:
            keys = self.tags.pop(tag, [])

            for key in keys:
                try:
                    self.delete(key, close=False)
                except KeyError:
                    pass

        self.close()

    def close(self):
        """Close cache."""
        if self.use_diskcache:
            self.cache.close()

    def delete_database(self, reinit=False):
        """Delte database.

        Parameters
        ----------
        reinit : bool, optional
            If True, reinitialize cache. The default is False.
        """
        self.close()
        if self.use_diskcache:
            try:
                shutil.rmtree(self.directory, ignore_errors=True)
            except FileNotFoundError:
                pass

        if reinit:
            self.init_cache()
