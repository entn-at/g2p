
from libcpp.string cimport string
from libcpp.vector cimport vector

cdef extern from "g2p.h":
    cdef cppclass G2P:
        G2P(string, string, string, string)
        vector[string]& Phonetisize(string)

cdef class PyG2P:
    cdef G2P *thisptr
    def __cinit__(self, nnpath, nnmeta, fstpath, dictpath):
        cdef string nnpath_c = nnpath.encode()
        cdef string nnmeta_c = nnmeta.encode()
        cdef string fstpath_c = fstpath.encode()
        cdef string dictpath_c = dictpath.encode()
        self.thisptr = new G2P(nnpath_c, nnmeta_c, fstpath_c, dictpath_c)
    def __dealloc__(self):
        del self.thisptr
    def Phonetisize(self, word):
        cdef string word_c = word.encode()
        res = self.thisptr.Phonetisize(word_c)
        return [x.decode('utf-8') for x in res]
