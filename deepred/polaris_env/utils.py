
class cproperty:
    def __init__(self, func):
        self.func = func
        self.cache_name = f"_{func.__name__}_cache"

    def __get__(self, instance, owner):
        if instance is None:
            return self
        # Check if the value is already cached
        if not hasattr(instance, self.cache_name):
            # Compute and store the value in the cache
            value = self.func(instance)
            setattr(instance, self.cache_name, value)
        return getattr(instance, self.cache_name)

    def __set__(self, instance, value):
        setattr(instance, self.cache_name, value)

    def __delete__(self, instance):
        if hasattr(instance, self.cache_name):
            delattr(instance, self.cache_name)