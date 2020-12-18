import numpy as np
from ctypes import *


class GoSlice(Structure):
    _fields_ = [("data", POINTER(c_double)), ("len", c_longlong), ("cap", c_longlong)]


class ReturnStruct(Structure):
    _fields_ = [("r0", c_double), ("r1", c_double)]

def extractFromGoSlice(slice):
    tmp = [slice.data[i] for i in range(slice.len)]
    return np.array(tmp)

def convertToGoSlice(npArray):
    data = (c_double * len(npArray))(0)
    for i in range(len(npArray)):
        data[i] = float(npArray[i])
    return GoSlice(data, len(data), len(data))

def secureAggregtion(inputs, logN, threshold):
    lib = cdll.LoadLibrary('./SecureAggregation/secagg.so')
    lib.SecureAggregation.argtypes = [GoSlice, c_longlong, c_longlong, c_ulonglong, GoSlice]
    lib.SecureAggregation.restype = ReturnStruct
    numUsers = len(inputs)
    N = 2 ** logN
    length = len(inputs[0])
    numPieces = int(np.ceil(length/N))
    segments = [np.array_split(i, numPieces) for i in inputs]
    aggInputs = [np.concatenate([segments[i][j] for i in range(numUsers)]) for j in range(numPieces)]
    secRes = []
    cloudTime = []
    partyTime = []
    for i in range(numPieces):
        res = GoSlice((c_double * len(segments[0][i]))(0), len(segments[0][i]), len(segments[0][i]))
        result = lib.SecureAggregation(convertToGoSlice(aggInputs[i]), numUsers, threshold, logN+1, res)
        secRes.append(extractFromGoSlice(res))
        cloudTime.append(result.r0)
        partyTime.append(result.r1)
    return np.concatenate(secRes), np.sum(cloudTime), np.sum(partyTime)





