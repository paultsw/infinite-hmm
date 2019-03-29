# distutils: language=c++
"""
Cython-based implementation of an expanding matrix table (i.e., the number
of rows and columns in the matrix are not fixed).
"""
# numerics:
import numpy as np
cimport numpy as cnp

cdef cppclass Matrix:
    cdef __cinit__(self, limit):
        # [-[-[ TODO ]-]-]
        # _data hold the raw counts:
        cdef vector[int] _data = new vector
        # _total contains the running total number of observations;
        # _nrows and _ncols are the number of rows and columns, respectively.
        cdef unsigned int self._total=0, self._nrows=0, self._ncols=0
        if self._data == NULL:
            raise MemoryError("[MatrixTable] could not allocate memory.")

    cdef __dealloc__(self):
        """C-level teardown: delete data table."""
        if self._data != NULL:
            del self._data

    def __init__(self, **kwargs):
        """Python-level initialization."""
        # [-[-[ TODO ]-]-]
        pass

    def __getitem__(self, pair):
        """Getter-overloading for the bracket-indexing operators `[,]`."""
        # [-[-[ TODO ]-]-]
        row, col = pair

    def __setitem__(self, pair, val):
        """Setter-overloading for the bracket-indexing operators `[,]`."""
        # check dimensions of pair, val:
        pass # [-[-[ TODO ]-]-]

    def to_numpy(self):
        """Return a c-level numpy array."""
        pass # [-[-[ TODO ]-]-]

    def shape(self):
        """Return current shape of the underlying matrix."""
        return (self._nrows, self._ncols)

    def __repr__(self):
        return repr(self.to_numpy())
