from typing import Any, Optional

from .dataStructure import Structure
from .parameter import Bool


class cached_property_if_locked(property):
    """
    A property that caches its value if the instance is locked.

    This property extends the built-in property to cache its value when the instance
    is locked. The cached value is stored in the instance's `cached_properties` dictionary.
    """

    def __get__(self, instance: Any, cls: Optional[type] = None) -> Any:
        """
        Get the value of the property, using the cache if the instance is locked.

        Parameters
        ----------
        instance : Any
            The instance from which to retrieve the property value.
        cls : Optional[type], optional
            The class of the instance, by default None.

        Returns
        -------
        Any
            The value of the property.
        """
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
    def name(self) -> str:
        """str: name of the property."""
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

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize Cached Properties Mixin Object."""
        super().__init__(*args, **kwargs)
        self.cached_properties = {}

    @property
    def lock(self) -> bool:
        """bool: If True, properties are cached. False otherwise."""
        return self._lock

    @lock.setter
    def lock(self, lock: bool) -> None:
        self._lock = lock
        self.cached_properties = {}
