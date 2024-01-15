class ExportFlag(object):
    _current = None

    @classmethod
    def current(cls):
        return cls._current
    
    def __init__(self, is_export=False) -> None:
        self._is_export = is_export
    
    @property
    def is_export(self):
        return self._is_export
    
    def __enter__(self):
        self._last = self.current
        self.__class__._current = self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.__class__._current = self._last

ExportFlag._current = ExportFlag()