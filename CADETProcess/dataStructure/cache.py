from .dataStructure import Structure
from .parameter import Bool


class cached_property_if_locked(property):
    def __get__(self, instance, cls=None):
        if instance.lock:
            try:
                return instance.cached_properties[self.name]
            except KeyError:
                pass

        value = super().__get__(instance, cls)

        if instance.lock:
            instance.cached_properties[self.name] = value

        return value

    @property
    def name(self):
        return self.fget.__name__


class CachedPropertiesMixin(Structure):
    """
    Mixin class for caching properties in a structured object.

    This class is designed to be used as a mixin in conjunction with other classes
    inheriting from `Structure`. It provides functionality for caching properties and
    managing a lock state to control the caching behavior.

    Notes
    -----
    - To prevent the return of outdated state, the cache is cleared whenever the `lock`
      state is changed.
    """

    _lock = Bool(default=False)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_properties = {}

    @property
    def lock(self):
        """bool: If True, properties are cached. False otherwise."""
        return self._lock

    @lock.setter
    def lock(self, lock):
        self._lock = lock
        self.cached_properties = {}
