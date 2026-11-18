/* Greedy FPS: legacy FP64 plus explicit fresh-training FP32. No NumPy C ABI. */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#ifndef FPS_SOURCE_SHA256
#define FPS_SOURCE_SHA256 "unknown"
#endif
#ifndef FPS_COMPILE_FLAGS
#define FPS_COMPILE_FLAGS "unknown"
#endif

/* The floating dtype changes only at this loop's explicit entry point. */
#define FPS_LOOP(Name, Scalar) \
static void Name(const Scalar *points, Py_ssize_t size, Py_ssize_t count, \
                 Py_ssize_t first, Scalar *distance, Py_ssize_t *output) { \
    for (Py_ssize_t j = 0; j < size; ++j) distance[j] = INFINITY; \
    Py_ssize_t current = first; \
    for (Py_ssize_t i = 0; i < count; ++i) { \
        output[i] = current; \
        distance[current] = -1.; \
        Scalar best = -2.; \
        Py_ssize_t next = 0; \
        for (Py_ssize_t j = 0; j < size; ++j) { \
            if (distance[j] >= 0.) { \
                const Scalar dx = points[3*j] - points[3*current]; \
                const Scalar dy = points[3*j+1] - points[3*current+1]; \
                const Scalar dz = points[3*j+2] - points[3*current+2]; \
                const Scalar candidate = (dx*dx + dy*dy) + dz*dz; \
                if (candidate < distance[j]) distance[j] = candidate; \
            } \
            /* Strict > keeps the first sorted raw index on equal distances. */ \
            if (distance[j] > best) { \
                best = distance[j]; \
                next = j; \
            } \
        } \
        current = next; \
    } \
}

FPS_LOOP(fps_loop_f64, double)
FPS_LOOP(fps_loop_f32, float)
#undef FPS_LOOP

static PyObject *fps_positions_typed(PyObject *args, int float32) {
    PyObject *object, *result = NULL;
    Py_buffer view = {0};
    Py_ssize_t count, first;
    void *distance = NULL;
    const Py_ssize_t itemsize = float32 ? sizeof(float) : sizeof(double);
    Py_ssize_t *output = NULL;
    if (!PyArg_ParseTuple(args, "Onn:fps_positions", &object, &count, &first))
        return NULL;
    if (PyObject_GetBuffer(object, &view, PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0)
        return NULL;
    const char *format = view.format ? view.format : "";
    if (*format == '@' || *format == '=') ++format;
    if (view.ndim != 2 || view.shape[1] != 3 || view.itemsize != itemsize ||
        strcmp(format, float32 ? "f" : "d") || (uintptr_t)view.buf % itemsize != 0) {
        PyErr_SetString(PyExc_ValueError, float32 ?
            "points must be aligned C-contiguous native float32 [M,3]" :
            "points must be aligned C-contiguous native float64 [M,3]");
        goto done;
    }
    const Py_ssize_t size = view.shape[0];
    if (count < 0 || count > size || first < 0 || (size ? first >= size : first != 0)) {
        PyErr_SetString(PyExc_ValueError, "FPS count or first position outside points");
        goto done;
    }
    if (!count) {
        result = PyList_New(0);
        goto done;
    }
    if (size > PY_SSIZE_T_MAX / itemsize ||
        count > PY_SSIZE_T_MAX / (Py_ssize_t)sizeof(Py_ssize_t)) {
        PyErr_NoMemory();
        goto done;
    }
    distance = PyMem_Malloc((size_t)size * itemsize);
    output = PyMem_Malloc((size_t)count * sizeof(Py_ssize_t));
    if (distance == NULL || output == NULL) {
        PyErr_NoMemory();
        goto done;
    }
    Py_BEGIN_ALLOW_THREADS
    if (float32) fps_loop_f32((const float *)view.buf, size, count, first,
                              (float *)distance, output);
    else fps_loop_f64((const double *)view.buf, size, count, first,
                      (double *)distance, output);
    Py_END_ALLOW_THREADS
    result = PyList_New(count);
    if (result == NULL) goto done;
    for (Py_ssize_t i = 0; i < count; ++i) {
        PyObject *position = PyLong_FromSsize_t(output[i]);
        if (position == NULL) {
            Py_CLEAR(result);
            goto done;
        }
        PyList_SET_ITEM(result, i, position);
    }
done:
    PyMem_Free(distance);
    PyMem_Free(output);
    PyBuffer_Release(&view);
    return result;
}

static PyObject *fps_positions(PyObject *self, PyObject *args) {
    (void)self;
    return fps_positions_typed(args, 0);
}
static PyObject *fps_positions_float32(PyObject *self, PyObject *args) {
    (void)self;
    return fps_positions_typed(args, 1);
}

static PyMethodDef methods[] = {
    {"fps_positions", fps_positions, METH_VARARGS, "Exact greedy FPS positions(points_f64, count, first_position)."},
    {"fps_positions_float32", fps_positions_float32, METH_VARARGS, "FP32 greedy FPS positions(points_f32, count, first_position)."},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_fps_cpu", NULL, -1, methods};
PyMODINIT_FUNC PyInit__fps_cpu(void) {
    PyObject *m = PyModule_Create(&module);
    if (m == NULL) return NULL;
    if (PyModule_AddIntConstant(m, "FPS_ABI_VERSION", 1) < 0 ||
        PyModule_AddIntConstant(m, "FPS_FLOAT32_VERSION", 1) < 0 ||
        PyModule_AddStringConstant(m, "SOURCE_SHA256", FPS_SOURCE_SHA256) < 0 ||
        PyModule_AddStringConstant(m, "COMPILE_FLAGS", FPS_COMPILE_FLAGS) < 0 ||
        PyModule_AddStringConstant(m, "COMPILER", __VERSION__) < 0) {
        Py_DECREF(m);
        return NULL;
    }
    return m;
}
