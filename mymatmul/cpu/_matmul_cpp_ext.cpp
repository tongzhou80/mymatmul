#define PY_ARRAY_UNIQUE_SYMBOL MYMATMUL_CPP_ARRAY_API
#include <Python.h>
#include <numpy/arrayobject.h>
#include <omp.h>

/* Both functions assume contiguous float32 row-major (C-order) arrays. */

static PyObject* py_matmul_cpp_ijk(PyObject* self, PyObject* args) {
    PyArrayObject *A, *B;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &A, &PyArray_Type, &B))
        return NULL;

    int M = (int)PyArray_DIM(A, 0);
    int K = (int)PyArray_DIM(A, 1);
    int N = (int)PyArray_DIM(B, 1);

    npy_intp dims[2] = {M, N};
    PyArrayObject* C = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_FLOAT, 0);
    if (!C) return NULL;

    float* __restrict__ pA = (float*)PyArray_DATA(A);
    float* __restrict__ pB = (float*)PyArray_DATA(B);
    float* __restrict__ pC = (float*)PyArray_DATA(C);

    for (int i = 0; i < M; i++)
        for (int j = 0; j < N; j++)
            for (int k = 0; k < K; k++)
                pC[i*N + j] += pA[i*K + k] * pB[k*N + j];

    return (PyObject*)C;
}

static PyObject* py_matmul_cpp_ikj(PyObject* self, PyObject* args) {
    PyArrayObject *A, *B;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &A, &PyArray_Type, &B))
        return NULL;

    int M = (int)PyArray_DIM(A, 0);
    int K = (int)PyArray_DIM(A, 1);
    int N = (int)PyArray_DIM(B, 1);

    npy_intp dims[2] = {M, N};
    PyArrayObject* C = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_FLOAT, 0);
    if (!C) return NULL;

    float* __restrict__ pA = (float*)PyArray_DATA(A);
    float* __restrict__ pB = (float*)PyArray_DATA(B);
    float* __restrict__ pC = (float*)PyArray_DATA(C);

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++)
            for (int j = 0; j < N; j++)
                pC[i*N + j] += pA[i*K + k] * pB[k*N + j];

    return (PyObject*)C;
}

static PyObject* py_matmul_cpp_ikj_vec(PyObject* self, PyObject* args) {
    PyArrayObject *A, *B;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &A, &PyArray_Type, &B))
        return NULL;

    int M = (int)PyArray_DIM(A, 0);
    int K = (int)PyArray_DIM(A, 1);
    int N = (int)PyArray_DIM(B, 1);

    npy_intp dims[2] = {M, N};
    PyArrayObject* C = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_FLOAT, 0);
    if (!C) return NULL;

    float* __restrict__ pA = (float*)PyArray_DATA(A);
    float* __restrict__ pB = (float*)PyArray_DATA(B);
    float* __restrict__ pC = (float*)PyArray_DATA(C);

    for (int i = 0; i < M; i++)
        for (int k = 0; k < K; k++) {
            float a_ik = pA[i*K + k];  // hoist scalar: inner j loop becomes pure SAXPY
            for (int j = 0; j < N; j++)
                pC[i*N + j] += a_ik * pB[k*N + j];
        }

    return (PyObject*)C;
}

static PyObject* py_matmul_cpp_ikj_unroll(PyObject* self, PyObject* args) {
    PyArrayObject *A, *B;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &A, &PyArray_Type, &B))
        return NULL;

    int M = (int)PyArray_DIM(A, 0);
    int K = (int)PyArray_DIM(A, 1);
    int N = (int)PyArray_DIM(B, 1);

    npy_intp dims[2] = {M, N};
    PyArrayObject* C = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_FLOAT, 0);
    if (!C) return NULL;

    float* __restrict__ pA = (float*)PyArray_DATA(A);
    float* __restrict__ pB = (float*)PyArray_DATA(B);
    float* __restrict__ pC = (float*)PyArray_DATA(C);

    // Unroll inner j loop by 4 vectors (8 floats each = 32 floats per iteration).
    // 4 independent FMA accumulator chains to hide the 4-cycle FMA latency.
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = pA[i*K + k];
            float* __restrict__ c = pC + i*N;
            const float* __restrict__ b = pB + k*N;
            int j = 0;
            for (; j + 31 < N; j += 32) {
                for (int jj = 0; jj < 32; jj++)
                    c[j + jj] += a_ik * b[j + jj];
            }
            for (; j < N; j++)
                c[j] += a_ik * b[j];
        }
    }

    return (PyObject*)C;
}

static PyObject* py_matmul_cpp_ikj_omp(PyObject* self, PyObject* args) {
    PyArrayObject *A, *B;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &A, &PyArray_Type, &B))
        return NULL;

    int M = (int)PyArray_DIM(A, 0);
    int K = (int)PyArray_DIM(A, 1);
    int N = (int)PyArray_DIM(B, 1);

    npy_intp dims[2] = {M, N};
    PyArrayObject* C = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_FLOAT, 0);
    if (!C) return NULL;

    float* __restrict__ pA = (float*)PyArray_DATA(A);
    float* __restrict__ pB = (float*)PyArray_DATA(B);
    float* __restrict__ pC = (float*)PyArray_DATA(C);

    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            for (int j = 0; j < N; j++)
                pC[i*N + j] += pA[i*K + k] * pB[k*N + j];
        }
    }

    return (PyObject*)C;
}

static PyObject* py_matmul_cpp_ikj_unroll_omp(PyObject* self, PyObject* args) {
    PyArrayObject *A, *B;
    if (!PyArg_ParseTuple(args, "O!O!", &PyArray_Type, &A, &PyArray_Type, &B))
        return NULL;

    int M = (int)PyArray_DIM(A, 0);
    int K = (int)PyArray_DIM(A, 1);
    int N = (int)PyArray_DIM(B, 1);

    npy_intp dims[2] = {M, N};
    PyArrayObject* C = (PyArrayObject*)PyArray_ZEROS(2, dims, NPY_FLOAT, 0);
    if (!C) return NULL;

    float* __restrict__ pA = (float*)PyArray_DATA(A);
    float* __restrict__ pB = (float*)PyArray_DATA(B);
    float* __restrict__ pC = (float*)PyArray_DATA(C);

    // Parallelize the i loop across cores; each thread owns distinct rows of C.
    #pragma omp parallel for schedule(static)
    for (int i = 0; i < M; i++) {
        for (int k = 0; k < K; k++) {
            float a_ik = pA[i*K + k];
            float* __restrict__ c = pC + i*N;
            const float* __restrict__ b = pB + k*N;
            int j = 0;
            for (; j + 31 < N; j += 32) {
                for (int jj = 0; jj < 32; jj++)
                    c[j + jj] += a_ik * b[j + jj];
            }
            for (; j < N; j++)
                c[j] += a_ik * b[j];
        }
    }

    return (PyObject*)C;
}

static PyMethodDef methods[] = {
    {"matmul_cpp_ijk",        py_matmul_cpp_ijk,        METH_VARARGS, "Naive C++ matmul, loop order i-j-k"},
    {"matmul_cpp_ikj",        py_matmul_cpp_ikj,        METH_VARARGS, "Naive C++ matmul, loop order i-k-j"},
    {"matmul_cpp_ikj_vec",    py_matmul_cpp_ikj_vec,    METH_VARARGS, "C++ matmul i-k-j with hoisted scalar, SAXPY inner loop"},
    {"matmul_cpp_ikj_unroll",     py_matmul_cpp_ikj_unroll,     METH_VARARGS, "C++ matmul i-k-j, SAXPY unrolled 4x to fill both FMA pipes"},
    {"matmul_cpp_ikj_omp",        py_matmul_cpp_ikj_omp,        METH_VARARGS, "C++ matmul i-k-j + OpenMP over i"},
    {"matmul_cpp_ikj_unroll_omp", py_matmul_cpp_ikj_unroll_omp, METH_VARARGS, "C++ matmul i-k-j, SAXPY unrolled 4x + OpenMP over i"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "_matmul_cpp_ext", NULL, -1, methods
};

PyMODINIT_FUNC PyInit__matmul_cpp_ext(void) {
    import_array();
    return PyModule_Create(&moduledef);
}
