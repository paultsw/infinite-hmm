# distutils: language=c++
"""
Cython-based implementation of an expanding matrix table (i.e., the number
of rows and columns in the matrix are not fixed).

Makes use of the std::map class from the C++ STL.
"""
# numerics:
import numpy as npy
cimport numpy as npc
from libcpp.map cimport cmap
from libcpp.pair cimport pair

cdef cppclass MatrixTable:
    cdef __cinit__(self):
        # _data hold the raw counts:
        cdef cmap[pair[int, int], int] self._data
        # _entry is a placeholder tuple for (row,col):
        cdef pair[int, int] self._entry
        # _total contains the running total number of observations:
        cdef unsigned int self._total=0, self._nrows=0, self._ncols=0
        if self._data == NULL:
            raise MemoryError("[MatrixTable] could not allocate memory.")

    def __dealloc__(self):
        """C-level teardown: delete data table."""
        if self._data != NULL:
            del self._data

    def __init__(self, **kwargs):
        """Python-level initialization."""
        pass

    def __getitem__(self, pair):
        """Getter-overloading for the bracket-indexing operators `[,]`."""
        row, col = pair
        # TODO

    def __setitem__(self, pair, val):
        """Setter-overloading for the bracket-indexing operators `[,]`."""
        # check dimensions of pair, val:
        pass # TODO

    def to_numpy(self):
        """Return a c-level numpy array."""
        pass # TODO

    def shape(self):
        """Return current shape of the underlying matrix."""
        return (self._nrows, self._ncols)

    def __repr__(self):
        return repr(self.to_numpy())
