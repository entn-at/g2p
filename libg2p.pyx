
from libcpp.string cimport string

cdef extern from "g2p.h":
    cdef cppclass G2P:
        G2P(string, string, string)
        void Phonetisize(string)

cdef class PyG2P:
    cdef G2P *thisptr
    def __cinit__(self, nnpath, nnmeta, fstpath):
        cdef string nnpath_c = nnpath.encode()
        cdef string nnmeta_c = nnmeta.encode()
        cdef string fstpath_c = fstpath.encode()
        self.thisptr = new G2P(nnpath_c, nnmeta_c, fstpath_c)
    def __dealloc__(self):
        del self.thisptr
    def Phonetisize(self, word):
        cdef string word_c = word.encode()
        self.thisptr.Phonetisize(word_c)
