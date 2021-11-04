from .dataStructure import StructMeta
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


class CachedPropertiesMixin(metaclass=StructMeta):
    lock = Bool(default=False)
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cached_properties = {}
