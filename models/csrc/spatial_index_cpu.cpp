// All-point CSR grid with legacy FP64 and explicit fresh-training FP32 paths.
// CPython buffer protocol; no NumPy C ABI or fixed-size neighborhood truncation.
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <exception>
#include <initializer_list>
#include <limits>
#include <memory>
#include <stdexcept>
#include <thread>
#include <type_traits>
#include <vector>

#ifndef GRID_SOURCE_SHA256
#define GRID_SOURCE_SHA256 "unknown"
#endif
#ifndef GRID_COMPILE_FLAGS
#define GRID_COMPILE_FLAGS "unknown"
#endif
#ifndef GRID_LINK_FLAGS
#define GRID_LINK_FLAGS "unknown"
#endif

using Row = std::int64_t;
static const char *CAPSULE_NAME = "suction20.cpu_grid.v1";

struct Grid {
    Py_buffer xyz_view{};  // Pins the exporter, even after Python index deletion.
    bool float32 = false;
    template <typename Scalar> const Scalar *coordinates() const {
        return static_cast<const Scalar *>(xyz_view.buf);
    }
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

template <typename Scalar>
static void build(Grid &g) {
    const auto *xyz = g.coordinates<Scalar>();
    if (g.start == g.stop) return;
    Row high[3];
    for (int d=0; d<3; ++d)
        g.low[d] = high[d] = coordinate(xyz[3*g.start+d], g.cell);
    for (Row i=g.start+1; i<g.stop; ++i) for (int d=0; d<3; ++d) {
        const Row k = coordinate(xyz[3*i+d], g.cell);
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
        const Row k = g.cell_id(coordinate(xyz[3*i], g.cell),
                               coordinate(xyz[3*i+1], g.cell),
                               coordinate(xyz[3*i+2], g.cell));
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

template <typename Scalar>
static PyObject *create_typed(PyObject *args) {
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
    const bool is_float32 = std::is_same<Scalar, float>::value;
    const char *format = v.format ? v.format : "";
    if (*format == '@' || *format == '=') ++format;
    if (v.ndim != 2 || v.shape[1] != 3 || v.itemsize != sizeof(Scalar) ||
        std::strcmp(format, is_float32 ? "f" : "d") ||
        reinterpret_cast<std::uintptr_t>(v.buf) % alignof(Scalar)) {
        PyErr_SetString(PyExc_ValueError, is_float32 ?
            "xyz must be aligned C-contiguous native float32 [P,3]" :
            "xyz must be aligned C-contiguous native float64 [P,3]");
        return nullptr;
    }
    if (start < 0 || stop < start || stop > v.shape[0] || !std::isfinite(cell) || cell <= 0) {
        PyErr_SetString(PyExc_ValueError, "invalid grid frame range or cell size");
        return nullptr;
    }
    g->float32 = is_float32;
    g->start = start; g->stop = stop; g->cell = cell;
    std::exception_ptr error;
    Py_BEGIN_ALLOW_THREADS
    try { build<Scalar>(*g); } catch (...) { error = std::current_exception(); }
    Py_END_ALLOW_THREADS
    if (error) { translate_exception(error); return nullptr; }
    PyObject *result = PyCapsule_New(g.get(), CAPSULE_NAME, destroy_grid);
    if (result) g.release();
    return result;
}

static PyObject *create(PyObject *, PyObject *args) { return create_typed<double>(args); }
static PyObject *create_float32(PyObject *, PyObject *args) { return create_typed<float>(args); }

template <typename Scalar>
static void query_into(const Grid &g, const Scalar *center, Scalar radius,
                       std::vector<Row> &output) {
    const auto *xyz = g.coordinates<Scalar>();
    output.clear();  // A contact worker reuses its own capacity, never another worker's.
    if (g.start == g.stop) return;
    Row lo[3], hi[3];
    for (int d=0; d<3; ++d) {
        // Keep the cell range conservative when FP32 addition rounds at a
        // cell wall. This only selects cells; the radius predicate below is
        // unchanged. Legacy FP64 uses its original bounds.
        double lower_coord = center[d]-radius, upper_coord = center[d]+radius;
        if (std::is_same<Scalar, float>::value) {
            lower_coord = std::nextafter(center[d]-radius, -std::numeric_limits<Scalar>::infinity());
            upper_coord = std::nextafter(center[d]+radius, std::numeric_limits<Scalar>::infinity());
        }
        const double lower = std::floor(lower_coord/g.cell);
        const double upper = std::floor(upper_coord/g.cell);
        const Row top = g.low[d]+g.dims[d]-1;
        if (lower > static_cast<double>(top) || upper < static_cast<double>(g.low[d])) return;
        lo[d] = lower <= static_cast<double>(g.low[d]) ? g.low[d] : static_cast<Row>(lower);
        hi[d] = upper >= static_cast<double>(top) ? top : static_cast<Row>(upper);
    }
    const Scalar radius2 = radius*radius;
    for (Row x=lo[0]; x<=hi[0]; ++x) for (Row y=lo[1]; y<=hi[1]; ++y)
        for (Row z=lo[2]; z<=hi[2]; ++z) {
            const Row k = g.cell_id(x, y, z);
            for (Row j=g.offsets[k]; j<g.offsets[k+1]; ++j) {
                const Row row = g.rows[j];
                const Scalar dx = xyz[3*row]-center[0];
                const Scalar dy = xyz[3*row+1]-center[1];
                const Scalar dz = xyz[3*row+2]-center[2];
                if ((dx*dx+dy*dy)+dz*dz <= radius2) output.push_back(row);
            }
        }
    std::sort(output.begin(), output.end());
}

template <typename Scalar>
static std::vector<Row> query(const Grid &g, const Scalar *center, Scalar radius) {
    std::vector<Row> output;
    query_into(g, center, radius, output);
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
    try {
        if (g->float32) {
            const float c[] = {static_cast<float>(center[0]), static_cast<float>(center[1]),
                               static_cast<float>(center[2])};
            output = query(*g, c, static_cast<float>(radius+1e-8));
        } else output = query(*g, center, radius+1e-8);
    }
    catch (...) { error = std::current_exception(); }
    Py_END_ALLOW_THREADS
    if (error) { translate_exception(error); return nullptr; }
    // One immutable owned buffer, no Python integer per neighbor and no scratch alias.
    return PyBytes_FromStringAndSize(reinterpret_cast<const char *>(output.data()),
                                    static_cast<Py_ssize_t>(output.size()*sizeof(Row)));
}


// Contact query uses only native pointers while the GIL is released. Buffer
// exports and individual capsule references pin all input/output allocations.
enum class BufferType { F64, F32, I64, Bool };

struct BufferPins {
    std::vector<Py_buffer> views;
    std::vector<PyObject *> capsules;
    BufferPins() { views.reserve(40); }
    ~BufferPins() {
        for (auto &view : views) PyBuffer_Release(&view);
        for (auto *capsule : capsules) Py_DECREF(capsule);
    }
    BufferPins(const BufferPins &) = delete;
    BufferPins &operator=(const BufferPins &) = delete;

    const Py_buffer *get(PyObject *dict, const char *key, BufferType type,
                         std::initializer_list<Py_ssize_t> shape, bool writable=false) {
        PyObject *object = PyDict_GetItemString(dict, key);
        if (!object) {
            PyErr_Format(PyExc_KeyError, "contact query is missing %s", key);
            return nullptr;
        }
        Py_buffer view{};
        const int flags = PyBUF_FORMAT | PyBUF_ND | PyBUF_C_CONTIGUOUS |
                          (writable ? PyBUF_WRITABLE : 0);
        if (PyObject_GetBuffer(object, &view, flags) < 0) return nullptr;
        const char *format = view.format ? view.format : "";
        if (*format == '@' || *format == '=') ++format;
        const char *expected = "";
        Py_ssize_t itemsize = 0;
        std::size_t alignment = 1;
        bool format_ok = false;
        switch (type) {
            case BufferType::F64:
                expected = "float64"; itemsize = sizeof(double); alignment = alignof(double);
                format_ok = std::strcmp(format, "d") == 0; break;
            case BufferType::F32:
                expected = "float32"; itemsize = sizeof(float); alignment = alignof(float);
                format_ok = std::strcmp(format, "f") == 0; break;
            case BufferType::I64:
                expected = "int64"; itemsize = sizeof(Row); alignment = alignof(Row);
                format_ok = std::strcmp(format, "q") == 0 || std::strcmp(format, "l") == 0;
                break;
            case BufferType::Bool:
                expected = "bool"; itemsize = 1;
                format_ok = std::strcmp(format, "?") == 0; break;
        }
        bool shape_ok = view.ndim == static_cast<int>(shape.size());
        if (shape_ok) {
            int axis = 0;
            for (const auto size : shape) {
                if (size >= 0 && view.shape[axis] != size) shape_ok = false;
                ++axis;
            }
        }
        if (!format_ok || view.itemsize != itemsize || !shape_ok ||
            reinterpret_cast<std::uintptr_t>(view.buf) % alignment) {
            PyErr_Format(PyExc_ValueError,
                         "%s must be aligned C-contiguous native %s with the contact-query shape",
                         key, expected);
            PyBuffer_Release(&view);
            return nullptr;
        }
        try { views.push_back(view); }
        catch (...) { PyBuffer_Release(&view); throw; }
        return &views.back();
    }
};

template <typename Scalar>
struct ContactInputs {
    const Scalar *points = nullptr, *axes = nullptr, *u = nullptr, *v = nullptr;
    const Scalar *centers = nullptr, *radii = nullptr, *segments = nullptr;
    const Row *instances = nullptr, *batch = nullptr, *selected = nullptr;
    const Row *scene_instance = nullptr;
    const float *scene_normals = nullptr;
    const unsigned char *scene_normal_valid = nullptr;
    Py_ssize_t count = 0, selected_count = 0, scene_count = 0;
    Scalar self_ignore = 0.;
};

struct ZoneOutput {
    Row *indices = nullptr, *counts = nullptr;
    unsigned char *mask = nullptr;
    float *geometry = nullptr;
    Py_ssize_t limit = 0;
};

struct ContactOutputs {
    std::array<ZoneOutput, 5> zones;
    Row *sectors = nullptr;
};

struct ContactStats {
    Row ball_returned_points = 0;
    Row fps_calls = 0, fps_input_points = 0;
    double ball_seconds = 0., region_seconds = 0., fps_seconds = 0.;
    double pack_seconds = 0., total_seconds = 0.;
};

using Clock = std::chrono::steady_clock;
static double elapsed(const Clock::time_point &start) {
    return std::chrono::duration<double>(Clock::now() - start).count();
}

template <typename Scalar>
struct ContactPoint {
    Row row;
    Scalar x, y, z, rho;
    bool same;
};

template <typename Scalar>
struct Region {
    std::vector<ContactPoint<Scalar>> points;
    std::vector<std::size_t> selected;
    std::size_t first = 0;
    Scalar nearest = std::numeric_limits<Scalar>::infinity();
    void clear() {
        points.clear(); selected.clear(); first = 0;
        nearest = std::numeric_limits<Scalar>::infinity();
    }
    void append(const ContactPoint<Scalar> &point, Scalar z_mid) {
        const Scalar dz = point.z - z_mid;
        const Scalar distance = (point.x*point.x + point.y*point.y) + dz*dz;
        // Sorted raw rows and strict < reproduce np.argmin's first tie.
        if (distance < nearest) { nearest = distance; first = points.size(); }
        points.push_back(point);
    }
};

template <typename Scalar>
struct ContactScratch {
    std::array<std::vector<Row>, 4> balls;
    std::array<Region<Scalar>, 5> regions;
    std::vector<Scalar> fps_xyz, fps_distance;
};

// Arithmetic order matches the existing compiled FPS. Geometry uses the
// original camera XYZ, not the locally projected/normalized descriptor.
template <typename Scalar>
static void select_region(Region<Scalar> &region, Py_ssize_t limit, const Scalar *xyz,
                          ContactScratch<Scalar> &scratch, ContactStats &stats) {
    const auto size = region.points.size();
    const auto count = std::min(size, static_cast<std::size_t>(limit));
    region.selected.resize(count);
    if (size <= static_cast<std::size_t>(limit)) {
        for (std::size_t j=0; j<count; ++j) region.selected[j] = j;
        return;  // Contact query bypasses FPS when M <= K.
    }
    ++stats.fps_calls;
    stats.fps_input_points += static_cast<Row>(size);
    scratch.fps_xyz.resize(3*size);
    scratch.fps_distance.assign(size, std::numeric_limits<Scalar>::infinity());
    for (std::size_t j=0; j<size; ++j) {
        const auto *point = xyz + 3*region.points[j].row;
        std::copy_n(point, 3, scratch.fps_xyz.data()+3*j);
    }
    const auto *points = scratch.fps_xyz.data();
    auto *distance = scratch.fps_distance.data();
    auto current = region.first;
    for (std::size_t i=0; i<count; ++i) {
        region.selected[i] = current;
        distance[current] = -1.;
        Scalar best = -2.;
        std::size_t next = 0;
        for (std::size_t j=0; j<size; ++j) {
            if (distance[j] >= 0.) {
                const Scalar dx = points[3*j] - points[3*current];
                const Scalar dy = points[3*j+1] - points[3*current+1];
                const Scalar dz = points[3*j+2] - points[3*current+2];
                const Scalar candidate = (dx*dx + dy*dy) + dz*dz;
                if (candidate < distance[j]) distance[j] = candidate;
            }
            if (distance[j] > best) { best = distance[j]; next = j; }
        }
        current = next;
    }
}

template <typename Scalar>
static Scalar reference_dot3(Scalar ax, Scalar ay, Scalar az,
                             Scalar bx, Scalar by, Scalar bz) {
    // NumPy's contraction starts with positive zero. Starting at the first
    // product instead preserves negative zero when all products are -0 (for
    // example an exact anchor projected onto an all-negative axis). Preserve
    // the existing nonzero operation order and disabled FMA contraction.
    Scalar result = 0.;
    result += ax*bx;
    result += ay*by;
    result += az*bz;
    return result;
}

template <typename Scalar>
static ContactPoint<Scalar> project_point(const Grid &grid, Row row, Row q,
                                  const ContactInputs<Scalar> &in) {
    const auto *center = in.points + 3*q;
    const auto *u = in.u + 3*q, *v = in.v + 3*q, *n = in.axes + 3*q;
    const auto *point = grid.coordinates<Scalar>() + 3*row;
    const Scalar dx = point[0]-center[0], dy = point[1]-center[1], dz = point[2]-center[2];
    ContactPoint<Scalar> out{row, reference_dot3<Scalar>(dx, dy, dz, u[0], u[1], u[2]),
                         reference_dot3<Scalar>(dx, dy, dz, v[0], v[1], v[2]),
                         reference_dot3<Scalar>(dx, dy, dz, n[0], n[1], n[2]), 0.,
                         in.scene_instance[row] == in.instances[q]};
    out.rho = std::hypot(out.x, out.y);
    return out;
}

template <typename Scalar>
static void process_contact(Row q, const Grid &grid, const ContactInputs<Scalar> &in,
                            ContactOutputs &out, ContactScratch<Scalar> &scratch,
                            ContactStats &stats, bool detailed) {
    auto timer = detailed ? Clock::now() : Clock::time_point{};
    for (int ball_index=0; ball_index<4; ++ball_index) {
        query_into(grid, in.centers + 12*q + 3*ball_index,
                   static_cast<Scalar>(in.radii[ball_index]+1e-8), scratch.balls[ball_index]);
        stats.ball_returned_points += static_cast<Row>(scratch.balls[ball_index].size());
    }
    if (detailed) { stats.ball_seconds += elapsed(timer); timer = Clock::now(); }
    for (auto &region : scratch.regions) region.clear();
    Row sector_counts[8]{};
    constexpr double pi = 3.141592653589793238462643383279502884;
    const double sector_scale = 8. / (2.*pi);
    // core and lip share the covering ball and each point's local coordinates.
    for (const auto row : scratch.balls[0]) {
        const auto point = project_point(grid, row, q, in);
        if (point.same && std::abs(point.z) <= .008) {
            if (point.rho <= .018) {
                scratch.regions[0].append(point, Scalar(0));
            } else if (point.rho <= .022) {
                scratch.regions[1].append(point, Scalar(0));
                const auto sector = static_cast<Row>(
                    std::floor((std::atan2(point.y, point.x)+pi)*sector_scale)) % 8;
                ++sector_counts[sector];
            }
        }
    }
    for (int zone=2; zone<5; ++zone) {
        const auto *segment = in.segments + 3*(zone-2);
        const Scalar lower = segment[0], upper = segment[1], radius = segment[2];
        const Scalar middle = (lower+upper)/2.;
        for (const auto row : scratch.balls[zone-1]) {
            const auto point = project_point(grid, row, q, in);
            const bool above_lower = zone == 2 ? point.z >= lower : point.z > lower;
            if (point.rho <= radius && above_lower && point.z <= upper &&
                (!point.same || point.z > in.self_ignore))
                scratch.regions[zone].append(point, middle);
        }
    }
    for (int zone=0; zone<5; ++zone)
        out.zones[zone].counts[q] = static_cast<Row>(scratch.regions[zone].points.size());
    for (int sector=0; sector<8; ++sector) out.sectors[8*q+sector] = sector_counts[sector];
    if (detailed) { stats.region_seconds += elapsed(timer); timer = Clock::now(); }
    for (int zone=0; zone<5; ++zone)
        select_region(scratch.regions[zone], out.zones[zone].limit, grid.coordinates<Scalar>(), scratch, stats);
    if (detailed) { stats.fps_seconds += elapsed(timer); timer = Clock::now(); }
    const auto *normal = in.axes + 3*q;
    for (int zone=0; zone<5; ++zone) {
        const Scalar radius = zone < 2 ? .022 : in.segments[3*(zone-2)+2];
        const Scalar height = zone < 2 ? .008 : std::max(
            std::abs(in.segments[3*(zone-2)]), std::abs(in.segments[3*(zone-2)+1]));
        const auto &region = scratch.regions[zone];
        auto &output = out.zones[zone];
        for (std::size_t i=0; i<region.selected.size(); ++i) {
            const auto &point = region.points[region.selected[i]];
            const Row position = q*output.limit+static_cast<Row>(i);
            output.indices[position] = point.row;
            output.mask[position] = 1;
            const bool normal_valid = in.scene_normal_valid[point.row] != 0;
            Scalar normal_dot = 0.;
            if (normal_valid) {
                const auto *observed = in.scene_normals + 3*point.row;
                normal_dot = reference_dot3<Scalar>(observed[0], observed[1], observed[2],
                                           normal[0], normal[1], normal[2]);
            }
            auto *geometry = output.geometry + 7*position;
            geometry[0] = static_cast<float>(point.x/radius);
            geometry[1] = static_cast<float>(point.y/radius);
            geometry[2] = static_cast<float>(point.z/height);
            geometry[3] = static_cast<float>(point.rho/radius);
            geometry[4] = static_cast<float>(normal_dot);
            geometry[5] = point.same ? 1.f : 0.f;
            geometry[6] = normal_valid ? 1.f : 0.f;
        }
    }
    if (detailed) stats.pack_seconds += elapsed(timer);
}

static bool put_row(PyObject *dict, const char *key, Row value) {
    PyObject *item = PyLong_FromLongLong(static_cast<long long>(value));
    if (!item) return false;
    const int status = PyDict_SetItemString(dict, key, item);
    Py_DECREF(item);
    return status == 0;
}

static bool put_seconds(PyObject *dict, const char *key, double value) {
    PyObject *item = PyFloat_FromDouble(value);
    if (!item) return false;
    const int status = PyDict_SetItemString(dict, key, item);
    Py_DECREF(item);
    return status == 0;
}

template <typename Scalar>
static PyObject *contact_query_typed(PyObject *grid_objects, PyObject *input_dict,
                                     PyObject *output_dict, int workers, int detailed) {
    if (!PyDict_Check(input_dict) || !PyDict_Check(output_dict) ||
        !PyList_Check(grid_objects) || workers < 1 || workers > 256) {
        PyErr_SetString(PyExc_ValueError,
                        "contact_query requires grid list, input/output dicts, and 1..256 workers");
        return nullptr;
    }
    try {
        BufferPins pins;
        ContactInputs<Scalar> in;
        constexpr auto scalar_type = std::is_same<Scalar, float>::value ? BufferType::F32 : BufferType::F64;
        ContactOutputs out;
        const Py_buffer *view = pins.get(input_dict, "points", scalar_type, {-1, 3});
        if (!view) return nullptr;
        in.count = view->shape[0]; in.points = static_cast<const Scalar *>(view->buf);
#define INPUT_BUFFER(member, key, type, cast_type, ...) \
        view = pins.get(input_dict, key, type, {__VA_ARGS__}); \
        if (!view) return nullptr; \
        in.member = static_cast<const cast_type *>(view->buf)
        INPUT_BUFFER(axes, "axes", scalar_type, Scalar, in.count, 3);
        INPUT_BUFFER(u, "tangent_u", scalar_type, Scalar, in.count, 3);
        INPUT_BUFFER(v, "tangent_v", scalar_type, Scalar, in.count, 3);
        INPUT_BUFFER(centers, "centers", scalar_type, Scalar, in.count, 4, 3);
        INPUT_BUFFER(radii, "radii", scalar_type, Scalar, 4);
        INPUT_BUFFER(segments, "segments", scalar_type, Scalar, 3, 3);
        INPUT_BUFFER(instances, "instances", BufferType::I64, Row, in.count);
        INPUT_BUFFER(batch, "batch", BufferType::I64, Row, in.count);
        INPUT_BUFFER(selected, "selected", BufferType::I64, Row, -1);
        in.selected_count = view->shape[0];
        INPUT_BUFFER(scene_instance, "scene_instance", BufferType::I64, Row, -1);
        in.scene_count = view->shape[0];
        INPUT_BUFFER(scene_normals, "scene_normals", BufferType::F32, float, in.scene_count, 3);
        INPUT_BUFFER(scene_normal_valid, "scene_normal_valid", BufferType::Bool,
                     unsigned char, in.scene_count);
#undef INPUT_BUFFER
        PyObject *ignore = PyDict_GetItemString(input_dict, "self_ignore");
        if (!ignore) {
            PyErr_SetString(PyExc_KeyError, "contact query is missing self_ignore");
            return nullptr;
        }
        in.self_ignore = PyFloat_AsDouble(ignore);
        if (PyErr_Occurred()) return nullptr;
        if (!std::isfinite(in.self_ignore))
            throw std::invalid_argument("self_ignore must be finite");
        for (int k=0; k<4; ++k)
            if (!std::isfinite(in.radii[k]) || in.radii[k] < 0.)
                throw std::invalid_argument("contact radii must be finite and nonnegative");
        for (int k=0; k<3; ++k) {
            const auto *segment = in.segments+3*k;
            if (!std::isfinite(segment[0]) || !std::isfinite(segment[1]) ||
                !std::isfinite(segment[2]) || segment[1] < segment[0] ||
                segment[2] <= 0. || std::max(std::abs(segment[0]), std::abs(segment[1])) <= 0.)
                throw std::invalid_argument("invalid contact segment bounds");
        }
        static const char *names[] = {"core", "lip", "cup", "mount", "approach"};
        static const Py_ssize_t limits[] = {32, 64, 32, 32, 32};
        for (int zone=0; zone<5; ++zone) {
            PyObject *dict = PyDict_GetItemString(output_dict, names[zone]);
            if (!dict || !PyDict_Check(dict)) {
                PyErr_Format(PyExc_ValueError, "contact output %s must be a dict", names[zone]);
                return nullptr;
            }
            auto &output = out.zones[zone];
            output.limit = limits[zone];
            view = pins.get(dict, "indices", BufferType::I64, {in.count, output.limit}, true);
            if (!view) return nullptr;
            output.indices = static_cast<Row *>(view->buf);
            view = pins.get(dict, "mask", BufferType::Bool, {in.count, output.limit}, true);
            if (!view) return nullptr;
            output.mask = static_cast<unsigned char *>(view->buf);
            view = pins.get(dict, "geometry", BufferType::F32, {in.count, output.limit, 7}, true);
            if (!view) return nullptr;
            output.geometry = static_cast<float *>(view->buf);
            view = pins.get(dict, "counts", BufferType::I64, {in.count}, true);
            if (!view) return nullptr;
            output.counts = static_cast<Row *>(view->buf);
        }
        view = pins.get(output_dict, "lip_sector_counts", BufferType::I64, {in.count, 8}, true);
        if (!view) return nullptr;
        out.sectors = static_cast<Row *>(view->buf);
        const auto frames = PyList_GET_SIZE(grid_objects);
        std::vector<Grid *> grids;
        grids.reserve(static_cast<std::size_t>(frames));
        pins.capsules.reserve(static_cast<std::size_t>(frames));
        for (Py_ssize_t frame=0; frame<frames; ++frame) {
            PyObject *capsule = PyList_GET_ITEM(grid_objects, frame);
            auto *grid = static_cast<Grid *>(PyCapsule_GetPointer(capsule, CAPSULE_NAME));
            if (!grid) return nullptr;
            if (grid->float32 != std::is_same<Scalar, float>::value)
                throw std::invalid_argument("contact-query precision must match the grid");
            if (grid->xyz_view.shape[0] != in.scene_count)
                throw std::invalid_argument("grid XYZ and scene point fields have different lengths");
            grids.push_back(grid);
            Py_INCREF(capsule);
            pins.capsules.push_back(capsule);
        }
        Row previous = -1;
        for (Py_ssize_t i=0; i<in.selected_count; ++i) {
            const auto q = in.selected[i];
            if (q <= previous || q >= in.count || in.batch[q] < 0 || in.batch[q] >= frames)
                throw std::invalid_argument("selected must contain ascending unique valid query rows and frames");
            previous = q;
            for (int axis=0; axis<3; ++axis) {
                if (!std::isfinite(in.points[3*q+axis]) || !std::isfinite(in.axes[3*q+axis]) ||
                    !std::isfinite(in.u[3*q+axis]) || !std::isfinite(in.v[3*q+axis]))
                    throw std::invalid_argument("selected contact geometry must be finite");
            }
            for (int k=0; k<12; ++k)
                if (!std::isfinite(in.centers[12*q+k]))
                    throw std::invalid_argument("selected contact centers must be finite");
        }
        const int active_workers = static_cast<int>(
            std::min<Py_ssize_t>(workers, in.selected_count));
        std::vector<ContactStats> stats(static_cast<std::size_t>(active_workers));
        std::vector<std::exception_ptr> errors(static_cast<std::size_t>(active_workers));
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(active_workers));
        std::exception_ptr launch_error;
        const auto started = Clock::now();
        Py_BEGIN_ALLOW_THREADS
        auto run_worker = [&](int worker) {
            try {
                ContactScratch<Scalar> scratch;
                const auto worker_started = detailed ? Clock::now() : Clock::time_point{};
                for (Py_ssize_t i=worker; i<in.selected_count; i+=active_workers) {
                    const auto q = in.selected[i];
                    process_contact(q, *grids[in.batch[q]], in, out, scratch, stats[worker], detailed);
                }
                if (detailed) stats[worker].total_seconds = elapsed(worker_started);
            } catch (...) { errors[worker] = std::current_exception(); }
        };
        try {
            if (active_workers == 1) run_worker(0);
            else for (int worker=0; worker<active_workers; ++worker)
                threads.emplace_back(run_worker, worker);
        } catch (...) { launch_error = std::current_exception(); }
        for (auto &thread : threads) thread.join();
        Py_END_ALLOW_THREADS
        const double wall = elapsed(started);
        if (launch_error) { translate_exception(launch_error); return nullptr; }
        for (const auto &error : errors)
            if (error) { translate_exception(error); return nullptr; }
        ContactStats total;
        for (const auto &value : stats) {
            total.ball_returned_points += value.ball_returned_points;
            total.fps_calls += value.fps_calls;
            total.fps_input_points += value.fps_input_points;
            total.ball_seconds += value.ball_seconds;
            total.region_seconds += value.region_seconds;
            total.fps_seconds += value.fps_seconds;
            total.pack_seconds += value.pack_seconds;
            total.total_seconds += value.total_seconds;
        }
        PyObject *result = PyDict_New();
        if (!result) return nullptr;
        bool okay = put_seconds(result, "native_query_wall_s", wall) &&
            put_row(result, "native_workers", active_workers) &&
            put_row(result, "valid_candidates", in.selected_count) &&
            put_row(result, "ball_calls", 4*in.selected_count) &&
            put_row(result, "ball_api_calls", 0) &&
            put_row(result, "native_contact_calls", 1) &&
            put_row(result, "ball_returned_points", total.ball_returned_points) &&
            put_row(result, "fps_calls", total.fps_calls) &&
            put_row(result, "fps_input_points", total.fps_input_points);
        if (okay && detailed)
            okay = put_seconds(result, "native_ball_query_worker_s", total.ball_seconds) &&
                put_seconds(result, "native_region_filter_worker_s", total.region_seconds) &&
                put_seconds(result, "native_fps_worker_s", total.fps_seconds) &&
                put_seconds(result, "native_descriptor_pack_worker_s", total.pack_seconds) &&
                put_seconds(result, "native_total_worker_s", total.total_seconds);
        if (!okay) { Py_DECREF(result); return nullptr; }
        return result;
    } catch (...) {
        translate_exception(std::current_exception());
        return nullptr;
    }
}

// Legacy five-argument call stays FP64. FP32 is explicitly requested by its
// input descriptor and must use a grid constructed from FP32 coordinates.
static PyObject *contact_query(PyObject *, PyObject *args, PyObject *kwargs) {
    PyObject *grids, *inputs, *outputs;
    int workers, detailed;
    static const char *keywords[] = {"grids", "inputs_dict", "outputs_dict",
                                     "workers", "detailed", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOOip:contact_query",
                                    const_cast<char **>(keywords), &grids, &inputs,
                                    &outputs, &workers, &detailed)) return nullptr;
    const char *precision = "float64";
    if (PyDict_Check(inputs)) {
        PyObject *value = PyDict_GetItemString(inputs, "geometry_precision");
        if (value) {
            precision = PyUnicode_AsUTF8(value);
            if (!precision) return nullptr;
        }
    }
    if (std::strcmp(precision, "float32") == 0)
        return contact_query_typed<float>(grids, inputs, outputs, workers, detailed);
    if (std::strcmp(precision, "float64") == 0)
        return contact_query_typed<double>(grids, inputs, outputs, workers, detailed);
    PyErr_SetString(PyExc_ValueError, "geometry_precision must be float32 or float64");
    return nullptr;
}

static PyMethodDef methods[] = {
    {"create", create, METH_VARARGS, "Build legacy FP64 grid(xyz64, start, stop, cell_size_m)."},
    {"create_float32", create_float32, METH_VARARGS, "Build fresh-training FP32 all-point grid."},
    {"ball", ball, METH_VARARGS, "Owned native int64 bytes for ball(grid, x, y, z, radius)."},
    {"contact_query", reinterpret_cast<PyCFunction>(contact_query), METH_VARARGS | METH_KEYWORDS,
     "Fill full-point contact outputs using configurable native candidate workers."},
    {nullptr, nullptr, 0, nullptr}
};
static PyModuleDef module = {PyModuleDef_HEAD_INIT, "_spatial_index_cpu", nullptr, -1, methods};
PyMODINIT_FUNC PyInit__spatial_index_cpu() {
    PyObject *m = PyModule_Create(&module);
    if (!m) return nullptr;
    if (PyModule_AddIntConstant(m, "GRID_ABI_VERSION", 1) < 0 ||
        PyModule_AddIntConstant(m, "CONTACT_ABI_VERSION", 1) < 0 ||
        PyModule_AddIntConstant(m, "GEOMETRY_FLOAT32_VERSION", 1) < 0 ||
        PyModule_AddStringConstant(m, "SOURCE_SHA256", GRID_SOURCE_SHA256) < 0 ||
        PyModule_AddStringConstant(m, "COMPILE_FLAGS", GRID_COMPILE_FLAGS) < 0 ||
        PyModule_AddStringConstant(m, "LINK_FLAGS", GRID_LINK_FLAGS) < 0 ||
        PyModule_AddStringConstant(m, "COMPILER", __VERSION__) < 0) {
        Py_DECREF(m); return nullptr;
    }
    return m;
}
