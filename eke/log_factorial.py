import numpy as _numpy


class LogFactorialTable:
    def __init__(self, max=100):
        self._table = _numpy.zeros(max, dtype=_numpy.float64)
        self._table[0] = 0
        self._max = max
        self._fill_table_from(0)

    def _fill_table_from(self, index):
        for this_index in range(index+1, len(self._table)):
            self._table[this_index] = (self._table[this_index-1]
                                       + _numpy.log(this_index))

    def _expand_table(self, new_length):
        new_length = int(new_length)
        if new_length < len(self._table):
            return
        old_table = self._table
        self._table = _numpy.zeros(new_length, dtype=_numpy.float64)
        self._table[:len(old_table)] = old_table
        self._max = new_length
        self._fill_table_from(len(old_table)-1)

    def _get_single(self, value):
        if not isinstance(value, int):
            raise ValueError("Value must be an integer")
        if value < 0:
            raise ValueError("Value must be positive")
        if value >= self._max:
            self._expand_table(int(_numpy.ceil(value/100.)*100))
        return self._table[value]

    def _get_array(self, value_array):
        if not issubclass(value_array.dtype.type, _numpy.integer):
            raise ValueError("Value array must be integers")
        if value_array.min() < 0:
            raise ValueError("Value array must be all positive")
        if value_array.max() >= self._max:
            self._expand_table(int(_numpy.ceil(value_array.max()/100.)*100))
        return self._table[value_array]

    def __call__(self, value):
        if isinstance(value, _numpy.ndarray):
            return self._get_array(value)
        else:
            return self._get_single(value)
