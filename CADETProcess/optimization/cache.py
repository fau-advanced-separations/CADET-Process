from collections import defaultdict
import shutil
import tempfile

from diskcache import Cache

from CADETProcess.dataStructure import DillDisk


class ResultsCache():
    """
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

    """

    def __init__(self, use_diskcache=True, directory=None):
        self.init_cache(use_diskcache, directory)
        self.tags = defaultdict(list)

    def init_cache(self, use_diskcache, directory):
        if use_diskcache:
            if directory is None:
                directory = tempfile.mkdtemp(prefix='diskcache-')
            self.directory = directory

            self.cache = Cache(
               directory,
               disk=DillDisk,
               disk_min_file_size=2**18,    # 256 kb
               size_limit=2**36,            # 64 GB
            )
            self.directory = self.cache.directory
        else:
            self.cache = {}
            self.directory = None

        self.use_diskcache = use_diskcache

    def set(self, eval_obj, step, x, result, tag=None):
        key = (eval_obj, step, str(x))
        if tag is not None:
            self.tags[tag].append(key)

        if self.use_diskcache:
            self.cache.set(key, result, expire=None)
        else:
            self.cache[key] = result

    def get(self, eval_obj, step, x):
        key = (eval_obj, step, str(x))

        result = self.cache[key]

        return result

    def delete(self, eval_obj, step, x):
        key = (eval_obj, step, str(x))

        if self.use_diskcache:
            self.cache.delete(key)
        else:
            self.cache.pop(key)

    def prune(self, tag='temp'):
        try:
            keys = self.tags.pop(tag)
            for key in keys:
                eval_obj, step, x = key
                self.delete(eval_obj, step, x)
        except KeyError:
            pass

    def close(self):
        if self.use_diskcache:
            self.cache.close()

    def delete_database(self, reinit=False):
        if self.use_diskcache:
            self.close()
            try:
                shutil.rmtree(self.directory, ignore_errors=True)
            except FileNotFoundError:
                pass

        self.cache = None

        if reinit:
            self.init_cache(self.use_diskcache, self.directory)
