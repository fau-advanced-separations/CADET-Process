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

    def set(self, eval_obj, step, x, result, tag=None, close=True):
        """Add entry to cache.

        Parameters
        ----------
        eval_obj : EvaluationObject
            Corresponding evaluation object for evaluation. Can be None.
        step : Evaluator
            (Intermediate) evaluator.
        x : list
            Value of optimization variables.
        result : object
            (Intermediate) result of evaluation.
        tag : str, optional
            Tag to associate with result. The default is None.
        """
        key = (eval_obj, step, str(x))
        if tag is not None:
            self.tags[tag].append(key)

        if self.use_diskcache:
            self.cache.set(key, result, expire=None)
        else:
            self.cache[key] = result

        if close:
            self.close()

    def get(self, eval_obj, step, x, close=True):
        """Get entry from cache.

        Parameters
        ----------
        eval_obj : EvaluationObject
            Corresponding evaluation object for evaluation. Can be None.
        step : Evaluator
            (Intermediate) evaluator.
        x : list
            Value of optimization variables.
        result : object
            (Intermediate) result of evaluation.

        Returns
        -------
        result : object
            (Intermediate) result of evaluation.
        """
        key = (eval_obj, step, str(x))

        result = self.cache[key]

        if close:
            self.close()

        return result

    def delete(self, eval_obj, step, x, close=True):
        """Remove entry from cache.

        Parameters
        ----------
        eval_obj : EvaluationObject
            Corresponding evaluation object for evaluation. Can be None.
        step : Evaluator
            (Intermediate) evaluator.
        x : list
            Value of optimization variables.
        """
        key = (eval_obj, step, str(x))

        if self.use_diskcache:
            self.cache.delete(key)
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
        try:
            keys = self.tags.pop(tag)
            for key in keys:
                eval_obj, step, x = key
                self.delete(eval_obj, step, x, close=False)
            self.close()
        except KeyError:
            pass

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
