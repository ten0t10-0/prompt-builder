class B_Value:
    __slots__ = ('_default', '_current')
    def __init__(self, default):
        self._default = default
        self._current = default
    @property
    def current(self):
        return self._current
    @current.setter
    def current(self, new):
        self._current = new
    @property
    def default(self):
        return self._default
    @default.setter
    def default(self, new):
        self._default = new
    def update(self, new_value, default_if_none):
        if new_value is not None:
            self.current = new_value
            return True
        elif default_if_none:
            self.reset()
        return False
    def reset(self):
        self.current = self.default
