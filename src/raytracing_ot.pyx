#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:40:09 2023

@author: fabrizio
"""

# cython: language_level=3
# cython: embedsignature=True
# cython: profile=False
# cython: nonecheck=False
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False


cimport cython
from cython.parallel import prange
from cython cimport cdivision, boundscheck, wraparound
from libc.math cimport sqrt
from libcpp cimport bool as bool_cpp
from libcpp.vector cimport vector as cpp_vector
import numpy as np
cimport numpy as np


cdef double SQRT2 = sqrt(2)


@cdivision(True)
@boundscheck(False)
@wraparound(False) 
cdef Py_ssize_t ravel_index((Py_ssize_t, Py_ssize_t) index_2d, 
                            (Py_ssize_t, Py_ssize_t) shape_2d):
    
    # row, col = index_2d
    return index_2d[0] * shape_2d[1] + index_2d[1]



@cdivision(True)
@boundscheck(False)
@wraparound(False) 
cdef (Py_ssize_t, Py_ssize_t) unravel_index(Py_ssize_t idx, 
                                            (Py_ssize_t, Py_ssize_t) shape):
    return (idx // shape[1], idx % shape[1])


@boundscheck(False)
@wraparound(False) 
cdef void fill_diagonal(double[:, ::1] A, double value):
    cdef size_t i
    for i in range(A.shape[0]):
        A[i, i] = value



@cython.embedsignature(True)
@cdivision(True)
@boundscheck(False)
@wraparound(False) 
cpdef get_cost_matrix((Py_ssize_t, Py_ssize_t) mesh_shape, 
                      double[:] slowness=None, 
                      double fill_unallowed=1e10):

    cdef Py_ssize_t mesh_size = mesh_shape[0] * mesh_shape[1]
    cdef Py_ssize_t i, j
    cdef Py_ssize_t irow, icol
    cdef bool_cpp allowed_left, allowed_right, allowed_below, allowed_above
    
    cdef double[:, ::1] cost_matrix = np.full((mesh_size, mesh_size), 
                                              fill_unallowed,
                                              dtype=np.double)
    
    fill_diagonal(cost_matrix, 0)
    if slowness is None:
        slowness = np.ones(mesh_size, dtype=np.double)
        
    for i in range(mesh_size):
        irow, icol = unravel_index(i, mesh_shape)
        allowed_left = icol - 1 >= 0
        allowed_right = icol + 1 < mesh_shape[1]
        allowed_below = irow - 1 >= 0
        allowed_above = irow + 1 < mesh_shape[0]
        if allowed_left:
            j = i - 1
            cost_matrix[i, j] = (slowness[i] + slowness[j]) / 2
        if allowed_right:
            j = i + 1
            cost_matrix[i, j] = (slowness[i] + slowness[j]) / 2
        if allowed_below:
            j = i - mesh_shape[1]
            cost_matrix[i, j] = (slowness[i] + slowness[j]) / 2
        if allowed_above:
            j = i + mesh_shape[1]
            cost_matrix[i, j] = (slowness[i] + slowness[j]) / 2
        if allowed_left and allowed_below:
            j = i - mesh_shape[1] - 1
            cost_matrix[i, j] = SQRT2 * (slowness[i] + slowness[j]) / 2
        if allowed_left and allowed_above:
            j = i + mesh_shape[1] - 1
            cost_matrix[i, j] = SQRT2 * (slowness[i] + slowness[j]) / 2
        if allowed_right and allowed_below:
            j = i - mesh_shape[1] + 1
            cost_matrix[i, j] = SQRT2 * (slowness[i] + slowness[j]) / 2
        if allowed_right and allowed_above:
            j = i + mesh_shape[1] + 1
            cost_matrix[i, j] = SQRT2 * (slowness[i] + slowness[j]) / 2
    return np.asarray(cost_matrix)
    


@cython.embedsignature(True)
@cdivision(True)
@boundscheck(False)
@wraparound(False) 
cpdef get_finite_idx_and_cost((Py_ssize_t, Py_ssize_t) mesh_shape, 
                               double[:] slowness=None):

    cdef Py_ssize_t mesh_size = mesh_shape[0] * mesh_shape[1]
    cdef (Py_ssize_t, Py_ssize_t) cost_matrix_shape = (mesh_size, mesh_size)
    cdef Py_ssize_t i, j
    cdef Py_ssize_t irow, icol
    cdef bool_cpp allowed_left, allowed_right, allowed_below, allowed_above
    cdef cpp_vector[long] idx
    cdef cpp_vector[double] cost
    
    if slowness is None:
        slowness = np.ones(mesh_size, dtype=np.double)
    
    for i in range(mesh_size):
        idx.push_back(ravel_index((i, i), cost_matrix_shape))
        cost.push_back(0)
        irow, icol = unravel_index(i, mesh_shape)
        allowed_left = icol - 1 >= 0
        allowed_right = icol + 1 < mesh_shape[1]
        allowed_below = irow - 1 >= 0
        allowed_above = irow + 1 < mesh_shape[0]
        if allowed_left:
            j = i - 1
            idx.push_back(ravel_index((i, j), cost_matrix_shape))
            cost.push_back((slowness[i] + slowness[j]) / 2)
        if allowed_right:
            j = i + 1
            idx.push_back(ravel_index((i, j), cost_matrix_shape))
            cost.push_back((slowness[i] + slowness[j]) / 2)
        if allowed_below:
            j = i - mesh_shape[1]
            idx.push_back(ravel_index((i, j), cost_matrix_shape))
            cost.push_back((slowness[i] + slowness[j]) / 2)
        if allowed_above:
            j = i + mesh_shape[1]
            idx.push_back(ravel_index((i, j), cost_matrix_shape))
            cost.push_back((slowness[i] + slowness[j]) / 2)
        if allowed_left and allowed_below:
            j = i - mesh_shape[1] - 1
            idx.push_back(ravel_index((i, j), cost_matrix_shape))
            cost.push_back(SQRT2 * (slowness[i] + slowness[j]) / 2)
        if allowed_left and allowed_above:
            j = i + mesh_shape[1] - 1
            idx.push_back(ravel_index((i, j), cost_matrix_shape))
            cost.push_back(SQRT2 * (slowness[i] + slowness[j]) / 2)
        if allowed_right and allowed_below:
            j = i - mesh_shape[1] + 1
            idx.push_back(ravel_index((i, j), cost_matrix_shape))
            cost.push_back(SQRT2 * (slowness[i] + slowness[j]) / 2)
        if allowed_right and allowed_above:
            j = i + mesh_shape[1] + 1
            idx.push_back(ravel_index((i, j), cost_matrix_shape))
            cost.push_back(SQRT2 * (slowness[i] + slowness[j]) / 2)
    return np.asarray(idx), np.asarray(cost)


@cython.embedsignature(True)
@cdivision(True)
@boundscheck(False)
@wraparound(False) 
cpdef get_A_eq((ssize_t, ssize_t) cost_matrix_shape,
               long[:] valid_indexes,):
    
    cdef ssize_t rows = cost_matrix_shape[0]
    cdef ssize_t cols = cost_matrix_shape[1]
    cdef ssize_t size_indexes = valid_indexes.shape[0]
    cdef ssize_t i, j
    cdef long idx
    cdef char[:, ::1] A_eq = np.zeros((rows+cols, size_indexes), 
                                       dtype=np.int8)
    for i in prange(rows, nogil=True):
        for j in range(size_indexes):
            idx = valid_indexes[j]
            if idx % cols == i:
                A_eq[i, j] = 1
            if i*cols <= idx < cols * (i+1):
                A_eq[i + rows, j] = 1
                
    return np.asarray(A_eq)
                



