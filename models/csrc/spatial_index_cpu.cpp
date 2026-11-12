// Exact FP64 all-point CSR grid. CPython buffer protocol; no NumPy C ABI.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

#ifndef GRID_SOURCE_SHA256
#define GRID_SOURCE_SHA256 "unknown"
#endif
#ifndef GRID_COMPILE_FLAGS
#define GRID_COMPILE_FLAGS "unknown"
#endif

using Row = std::int64_t;
static const char *CAPSULE_NAME = "suction20.cpu_grid.v1";

struct Grid {
    Py_buffer xyz_view{};  // Pins the exporter, even after Python index deletion.
    const double *xyz = nullptr;
    Row start = 0, stop = 0, low[3]{}, dims[3]{};
    double cell = .02;
    std::vector<Row> offsets, rows;
    ~Grid() { if (xyz_view.obj) PyBuffer_Release(&xyz_view); }
    Row cell_id(Row x, Row y, Row z) const {
        return ((x-low[0])*dims[1]+y-low[1])*dims[2]+z-low[2];
    }
};

static Row coordinate(double value, double cell) {
    const double k = std::floor(value/cell);
    // Keep conversions and subsequent subtraction in signed int64 range.
    const double limit = static_cast<double>(std::numeric_limits<Row>::max()/4);
    if (!std::isfinite(k) || k <= -limit || k >= limit)
        throw std::invalid_argument("grid coordinates must be finite and within int64 range");
    return static_cast<Row>(k);
}

static void build(Grid &g) {
    if (g.start == g.stop) return;
    Row high[3];
    for (int d=0; d<3; ++d)
        g.low[d] = high[d] = coordinate(g.xyz[3*g.start+d], g.cell);
    for (Row i=g.start+1; i<g.stop; ++i) for (int d=0; d<3; ++d) {
        const Row k = coordinate(g.xyz[3*i+d], g.cell);
        g.low[d] = std::min(g.low[d], k);
        high[d] = std::max(high[d], k);
    }
    Row cells = 1;
    const Row max_cells = PY_SSIZE_T_MAX/static_cast<Row>(sizeof(Row))-1;
    for (int d=0; d<3; ++d) {
        g.dims[d] = high[d]-g.low[d]+1;
        if (g.dims[d] > max_cells/cells)
            throw std::invalid_argument("grid cell allocation exceeds addressable range");
        cells *= g.dims[d];
    }
    g.offsets.assign(static_cast<std::size_t>(cells)+1, 0);
    std::vector<Row> cell_of(static_cast<std::size_t>(g.stop-g.start));
    for (Row i=g.start; i<g.stop; ++i) {
        const Row k = g.cell_id(coordinate(g.xyz[3*i], g.cell),
                               coordinate(g.xyz[3*i+1], g.cell),
                               coordinate(g.xyz[3*i+2], g.cell));
        cell_of[i-g.start] = k;
        ++g.offsets[k+1];
    }
    for (Row k=0; k<cells; ++k) g.offsets[k+1] += g.offsets[k];
    auto cursor = g.offsets;
    g.rows.resize(static_cast<std::size_t>(g.stop-g.start));
    for (Row i=g.start; i<g.stop; ++i) g.rows[cursor[cell_of[i-g.start]]++] = i;
}

static void translate_exception(const std::exception_ptr &error) {
    try { std::rethrow_exception(error); }
    catch (const std::bad_alloc &) { PyErr_NoMemory(); }
    catch (const std::invalid_argument &e) { PyErr_SetString(PyExc_ValueError, e.what()); }
    catch (const std::exception &e) { PyErr_SetString(PyExc_RuntimeError, e.what()); }
    catch (...) { PyErr_SetString(PyExc_RuntimeError, "native grid failed"); }
}

static void destroy_grid(PyObject *capsule) {
    delete static_cast<Grid *>(PyCapsule_GetPointer(capsule, CAPSULE_NAME));
}

static PyObject *create(PyObject *, PyObject *args) {
    PyObject *object;
    Py_ssize_t start, stop;
    double cell;
    if (!PyArg_ParseTuple(args, "Onnd:create", &object, &start, &stop, &cell)) return nullptr;
    std::unique_ptr<Grid> g;
    try { g.reset(new Grid); }
    catch (...) { translate_exception(std::current_exception()); return nullptr; }
    if (PyObject_GetBuffer(object, &g->xyz_view, PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS) < 0)
        return nullptr;
    const auto &v = g->xyz_view;
    if (v.ndim != 2 || v.shape[1] != 3 || v.itemsize != sizeof(double) || !v.format ||
        (std::strcmp(v.format, "d") && std::strcmp(v.format, "@d") && std::strcmp(v.format, "=d")) ||
        reinterpret_cast<std::uintptr_t>(v.buf) % alignof(double)) {
        PyErr_SetString(PyExc_ValueError, "xyz must be aligned C-contiguous native float64 [P,3]");
        return nullptr;
    }
    if (start < 0 || stop < start || stop > v.shape[0] || !std::isfinite(cell) || cell <= 0) {
        PyErr_SetString(PyExc_ValueError, "invalid grid frame range or cell size");
        return nullptr;
    }
    g->xyz = static_cast<const double *>(v.buf);
    g->start = start; g->stop = stop; g->cell = cell;
    std::exception_ptr error;
    Py_BEGIN_ALLOW_THREADS
    try { build(*g); } catch (...) { error = std::current_exception(); }
    Py_END_ALLOW_THREADS
    if (error) { translate_exception(error); return nullptr; }
    PyObject *result = PyCapsule_New(g.get(), CAPSULE_NAME, destroy_grid);
    if (result) g.release();
    return result;
}

static std::vector<Row> query(const Grid &g, const double *center, double radius) {
    std::vector<Row> output;  // Each call owns its output; CSR/XYZ are read-only.
    if (g.start == g.stop) return output;
    Row lo[3], hi[3];
    for (int d=0; d<3; ++d) {
        // Clip in FP64 before conversion, including balls outside the frame.
        const double lower = std::floor((center[d]-radius)/g.cell);
        const double upper = std::floor((center[d]+radius)/g.cell);
        const Row top = g.low[d]+g.dims[d]-1;
        if (lower > static_cast<double>(top) || upper < static_cast<double>(g.low[d])) return output;
        lo[d] = lower <= static_cast<double>(g.low[d]) ? g.low[d] : static_cast<Row>(lower);
        hi[d] = upper >= static_cast<double>(top) ? top : static_cast<Row>(upper);
    }
    const double radius2 = radius*radius;
    for (Row x=lo[0]; x<=hi[0]; ++x) for (Row y=lo[1]; y<=hi[1]; ++y)
        for (Row z=lo[2]; z<=hi[2]; ++z) {
            const Row k = g.cell_id(x, y, z);
            for (Row j=g.offsets[k]; j<g.offsets[k+1]; ++j) {
                const Row row = g.rows[j];
                const double dx = g.xyz[3*row]-center[0];
                const double dy = g.xyz[3*row+1]-center[1];
                const double dz = g.xyz[3*row+2]-center[2];
                if ((dx*dx+dy*dy)+dz*dz <= radius2) output.push_back(row);
            }
        }
    std::sort(output.begin(), output.end());
    return output;
}

static PyObject *ball(PyObject *, PyObject *args) {
    PyObject *capsule;
    double center[3], radius;
    if (!PyArg_ParseTuple(args, "Odddd:ball", &capsule, &center[0], &center[1], &center[2], &radius))
        return nullptr;
    auto *g = static_cast<Grid *>(PyCapsule_GetPointer(capsule, CAPSULE_NAME));
    if (!g) return nullptr;
    if (!std::isfinite(radius) || radius < 0 || !std::isfinite(center[0]) ||
        !std::isfinite(center[1]) || !std::isfinite(center[2])) {
        PyErr_SetString(PyExc_ValueError, "ball requires finite center and nonnegative finite radius");
        return nullptr;
    }
    std::vector<Row> output;
    std::exception_ptr error;
    Py_BEGIN_ALLOW_THREADS
    try { output = query(*g, center, radius+1e-8); }
    catch (...) { error = std::current_exception(); }
    Py_END_ALLOW_THREADS
    if (error) { translate_exception(error); return nullptr; }
    // One immutable owned buffer, no Python integer per neighbor and no scratch alias.
    return PyBytes_FromStringAndSize(reinterpret_cast<const char *>(output.data()),
                                    static_cast<Py_ssize_t>(output.size()*sizeof(Row)));
}

static PyMethodDef methods[] = {
    {"create", create, METH_VARARGS, "Build grid(xyz64, start, stop, cell_size_m)."},
    {"ball", ball, METH_VARARGS, "Owned native int64 bytes for ball(grid, x, y, z, radius)."},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_spatial_index_cpu", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__spatial_index_cpu() {
    PyObject *m = PyModule_Create(&module);
    if (!m) return nullptr;
    if (PyModule_AddIntConstant(m, "GRID_ABI_VERSION", 1) < 0 ||
        PyModule_AddStringConstant(m, "SOURCE_SHA256", GRID_SOURCE_SHA256) < 0 ||
        PyModule_AddStringConstant(m, "COMPILE_FLAGS", GRID_COMPILE_FLAGS) < 0 ||
        PyModule_AddStringConstant(m, "COMPILER", __VERSION__) < 0) {
        Py_DECREF(m); return nullptr;
    }
    return m;
}
