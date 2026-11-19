// Route-A contact kernel. Included by spatial_index_cpu.cpp so the existing
// all-point CSR grid, pinned buffer ownership and native worker framework are
// shared. No five-zone E5 feature or sampling rule enters this implementation.

struct RouteAInputs {
    const double *points = nullptr, *normals = nullptr, *observed_normals = nullptr;
    const double *gravity = nullptr;
    const Row *components = nullptr, *source_components = nullptr, *source_ids = nullptr;
    const Row *selected = nullptr, *fps_axes = nullptr;
    const unsigned char *feature_valid = nullptr, *normal_valid = nullptr, *rgb_valid = nullptr;
    Py_ssize_t count = 0, source_count = 0, selected_count = 0;
    double boundary = 1e-12;
    bool source_ids_ordered = true;
};

struct RouteAOutputs {
    Row *indices = nullptr, *counts = nullptr, *angular_support = nullptr;
    unsigned char *mask = nullptr, *uncertain = nullptr;
    double *descriptor = nullptr, *summary = nullptr;
};

struct RouteAPoint {
    Row row;
    double rho, height, tangent[3];
};

struct RouteAScratch {
    std::vector<Row> ball;
    std::array<std::vector<RouteAPoint>, 3> regions;
    std::vector<std::size_t> selected;
    std::vector<double> distance, angles, heights, work;
};

// NumPy's contiguous 1D reductions use this eight-lane pairwise shape. Keeping
// it here also bounds roundoff for large raw lip rings; native trig functions
// can still differ from NumPy's vector math by a few FP64 ULPs.
static double route_a_pairwise_sum(const double *values, std::size_t size) {
    if (size < 8) {
        double result = -0.;
        for (std::size_t i=0; i<size; ++i) result += values[i];
        return result;
    }
    if (size <= 128) {
        double acc[8];
        std::copy_n(values, 8, acc);
        const auto stop = size - size % 8;
        for (std::size_t i=8; i<stop; i+=8)
            for (int k=0; k<8; ++k) acc[k] += values[i+k];
        double result = ((acc[0]+acc[1])+(acc[2]+acc[3]))+
                        ((acc[4]+acc[5])+(acc[6]+acc[7]));
        for (std::size_t i=stop; i<size; ++i) result += values[i];
        return result;
    }
    auto half = size / 2;
    half -= half % 8;
    return route_a_pairwise_sum(values, half)+route_a_pairwise_sum(values+half, size-half);
}

static double route_a_dot(const double *left, const double *right) {
    return reference_dot3(left[0], left[1], left[2], right[0], right[1], right[2]);
}

static void route_a_cross(const double *left, const double *right, double *output) {
    output[0] = left[1]*right[2]-left[2]*right[1];
    output[1] = left[2]*right[0]-left[0]*right[2];
    output[2] = left[0]*right[1]-left[1]*right[0];
}

static void route_a_fps(const std::vector<RouteAPoint> &region, std::size_t budget,
                        const double *xyz, const Row *axes, RouteAScratch &scratch) {
    const auto size = region.size(), count = std::min(size, budget);
    scratch.selected.resize(count);
    scratch.distance.assign(size, std::numeric_limits<double>::infinity());
    std::size_t current = 0;
    for (std::size_t step=0; step<count; ++step) {
        scratch.selected[step] = current;
        scratch.distance[current] = -1.;
        const auto *center = xyz+3*region[current].row;
        double best = -2.;
        std::size_t next = 0;
        for (std::size_t j=0; j<size; ++j) {
            if (scratch.distance[j] >= 0.) {
                const auto *point = xyz+3*region[j].row;
                const double dx = point[axes[0]]-center[axes[0]];
                const double dy = point[axes[1]]-center[axes[1]];
                const double dz = point[axes[2]]-center[axes[2]];
                const double candidate = (dx*dx+dy*dy)+dz*dz;
                if (candidate < scratch.distance[j]) scratch.distance[j] = candidate;
            }
            // Regions are in source-ID order. Strict > retains the first ID
            // on exact ties, including distinct source pixels with equal XYZ.
            if (scratch.distance[j] > best) { best = scratch.distance[j]; next = j; }
        }
        current = next;
    }
}

static void route_a_lip(Row q, const double *normal, RouteAScratch &scratch,
                        RouteAOutputs &out) {
    const auto &lip = scratch.regions[2];
    const auto count = lip.size();
    if (!count) return;
    int axis = 0;
    for (int k=1; k<3; ++k)
        if (std::abs(normal[k]) < std::abs(normal[axis])) axis = k;
    double reference[3]{}, first[3], second[3];
    reference[axis] = 1.;
    route_a_cross(reference, normal, first);
    const double length = std::sqrt(route_a_dot(first, first));
    for (double &value : first) value /= length;
    route_a_cross(normal, first, second);
    const double tau = 2.*std::acos(-1.);
    scratch.angles.resize(count);
    scratch.heights.resize(count);
    scratch.work.resize(count);
    for (std::size_t i=0; i<count; ++i) {
        double angle = std::atan2(route_a_dot(lip[i].tangent, second),
                                  route_a_dot(lip[i].tangent, first));
        if (angle < 0.) angle += tau;
        // np.mod returns +0 for both signs of an exact zero.
        if (angle == 0.) angle = 0.;
        scratch.angles[i] = angle;
        scratch.heights[i] = lip[i].height;
        scratch.work[i] = angle;
    }
    std::sort(scratch.work.begin(), scratch.work.end());
    Row angular = 0;
    double maximum_gap = 0.;
    const double angular_guard = 256.*std::numeric_limits<double>::epsilon()*tau;
    for (std::size_t i=0; i<count; ++i) {
        const double gap = (i+1<count ? scratch.work[i+1] : scratch.work[0]+tau)-scratch.work[i];
        angular += gap > 1e-6;
        maximum_gap = std::max(maximum_gap, gap);
        if (std::abs(gap-1e-6) <= angular_guard) out.uncertain[q] = 1;
    }
    angular = std::max<Row>(1, angular);
    out.angular_support[q] = angular;
    const double mean = route_a_pairwise_sum(scratch.heights.data(), count)/count;
    for (std::size_t i=0; i<count; ++i) {
        scratch.heights[i] -= mean;
        scratch.work[i] = scratch.heights[i]*scratch.heights[i];
    }
    const double rms = std::sqrt(route_a_pairwise_sum(scratch.work.data(), count)/count);
    auto *summary = out.summary+8*q;
    summary[0] = std::log1p(static_cast<double>(count));
    summary[1] = maximum_gap/tau;
    summary[2] = 1.;
    summary[3] = rms/.040;
    summary[7] = angular >= 7;
    if (angular >= 7) for (int harmonic=1; harmonic<=3; ++harmonic) {
        double cosine = 0., sine = 0.;
        // The reference has shape [M,3] and reduces the noncontiguous axis 0.
        for (std::size_t i=0; i<count; ++i) {
            const double argument = scratch.angles[i]*harmonic;
            cosine += scratch.heights[i]*std::cos(argument);
            sine += scratch.heights[i]*std::sin(argument);
        }
        summary[3+harmonic] = std::hypot(cosine/count, sine/count)/.040;
    }
}

static void route_a_process(Row q, const Grid &grid, const RouteAInputs &in,
                            RouteAOutputs &out, RouteAScratch &scratch) {
    const auto *xyz = grid.coordinates<double>();
    const auto *center = in.points+3*q, *normal = in.normals+3*q;
    const double bound = std::hypot(.022, .040)+2.*in.boundary;
    // A slightly conservative ball cannot add a contact outside the explicit
    // cylinder below, and avoids grid-cell/radius roundoff dropping its corners.
    query_into(grid, center, bound+16.*std::numeric_limits<double>::epsilon()*bound, scratch.ball);
    // Normal A support is already in source-ID order; query_into sorted raw
    // rows, so avoid repeating the same sort for every candidate.
    if (!in.source_ids_ordered)
        std::stable_sort(scratch.ball.begin(), scratch.ball.end(), [&](Row left, Row right) {
            return in.source_ids[left] < in.source_ids[right];
        });
    for (auto &region : scratch.regions) region.clear();
    static const double boundaries[] = {0., .010, .018, .022};
    for (const auto row : scratch.ball) {
        if (in.source_components[row] != in.components[q]) continue;
        const auto *point = xyz+3*row;
        const double delta[] = {point[0]-center[0], point[1]-center[1], point[2]-center[2]};
        RouteAPoint projected;
        projected.row = row;
        projected.height = route_a_dot(delta, normal);
        double size = 0.;
        for (int k=0; k<3; ++k) {
            projected.tangent[k] = delta[k]-projected.height*normal[k];
            size += std::abs(delta[k]);
        }
        projected.rho = std::sqrt((projected.tangent[0]*projected.tangent[0]+
                                  projected.tangent[1]*projected.tangent[1])+
                                  projected.tangent[2]*projected.tangent[2]);
        const double guard = 128.*std::numeric_limits<double>::epsilon()*(size+.040);
        if (std::abs(std::abs(projected.height)-(.040+in.boundary)) <= guard)
            out.uncertain[q] = 1;
        double membership = projected.rho;
        for (const auto boundary : boundaries) {
            const double difference = std::abs(projected.rho-boundary);
            if (std::abs(difference-in.boundary) <= guard) out.uncertain[q] = 1;
            if (difference <= in.boundary) membership = boundary;
        }
        if (std::abs(projected.height) > .040+in.boundary) continue;
        int zone = membership < .010 ? 0 : membership < .018 ? 1 : membership <= .022 ? 2 : -1;
        if (zone >= 0) scratch.regions[zone].push_back(projected);
    }
    route_a_lip(q, normal, scratch, out);
    static const std::size_t budgets[] = {32, 32, 64};
    static const double outer[] = {.010, .018, .022};
    std::size_t offset = 0;
    for (int zone=0; zone<3; ++zone) {
        const auto &region = scratch.regions[zone];
        out.counts[3*q+zone] = static_cast<Row>(region.size());
        route_a_fps(region, budgets[zone], xyz, in.fps_axes, scratch);
        for (std::size_t j=0; j<scratch.selected.size(); ++j) {
            const auto &point = region[scratch.selected[j]];
            const auto slot = 128*q+offset+j;
            out.indices[slot] = point.row;
            out.mask[slot] = 1;
            auto *descriptor = out.descriptor+8*slot;
            const auto *observed = in.observed_normals+3*point.row;
            descriptor[0] = point.rho/outer[zone];
            descriptor[1] = point.height/.040;
            descriptor[2] = route_a_dot(observed, normal);
            descriptor[3] = route_a_dot(point.tangent, in.gravity)/outer[zone];
            descriptor[4] = route_a_dot(observed, in.gravity);
            descriptor[5] = in.feature_valid[point.row];
            descriptor[6] = in.normal_valid[point.row];
            descriptor[7] = in.rgb_valid[point.row];
        }
        offset += budgets[zone];
    }
}

static PyObject *route_a_contact_query(PyObject *, PyObject *args, PyObject *kwargs) {
    PyObject *capsule, *input_dict, *output_dict;
    int workers = 1;
    static const char *keywords[] = {"grid", "inputs_dict", "outputs_dict", "workers", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OOO|i:route_a_contact_query",
                                    const_cast<char **>(keywords), &capsule, &input_dict,
                                    &output_dict, &workers)) return nullptr;
    if (!PyDict_Check(input_dict) || !PyDict_Check(output_dict) || workers < 1 || workers > 256) {
        PyErr_SetString(PyExc_ValueError, "route-A contact query requires input/output dicts and 1..256 workers");
        return nullptr;
    }
    auto *grid = static_cast<Grid *>(PyCapsule_GetPointer(capsule, CAPSULE_NAME));
    if (!grid) return nullptr;
    try {
        BufferPins pins;
        pins.capsules.reserve(1);
        Py_INCREF(capsule);
        pins.capsules.push_back(capsule);
        if (grid->float32 || grid->start != 0 || grid->stop != grid->xyz_view.shape[0])
            throw std::invalid_argument("route-A requires one full FP64 support grid");
        RouteAInputs in;
        RouteAOutputs out;
        in.source_count = grid->stop;
        const Py_buffer *view = pins.get(input_dict, "points", BufferType::F64, {-1, 3});
        if (!view) return nullptr;
        in.count = view->shape[0]; in.points = static_cast<const double *>(view->buf);
#define A_INPUT(member, key, type, cast_type, ...) \
        view = pins.get(input_dict, key, type, {__VA_ARGS__}); \
        if (!view) return nullptr; \
        in.member = static_cast<const cast_type *>(view->buf)
        A_INPUT(normals, "normals", BufferType::F64, double, in.count, 3);
        A_INPUT(components, "components", BufferType::I64, Row, in.count);
        A_INPUT(selected, "selected", BufferType::I64, Row, -1);
        in.selected_count = view->shape[0];
        A_INPUT(observed_normals, "observed_normals", BufferType::F64, double, in.source_count, 3);
        A_INPUT(source_components, "source_components", BufferType::I64, Row, in.source_count);
        A_INPUT(source_ids, "source_ids", BufferType::I64, Row, in.source_count);
        A_INPUT(feature_valid, "feature_valid", BufferType::Bool, unsigned char, in.source_count);
        A_INPUT(normal_valid, "normal_valid", BufferType::Bool, unsigned char, in.source_count);
        A_INPUT(rgb_valid, "rgb_valid", BufferType::Bool, unsigned char, in.source_count);
        A_INPUT(gravity, "gravity", BufferType::F64, double, 3);
        A_INPUT(fps_axes, "fps_axes", BufferType::I64, Row, 3);
#undef A_INPUT
        PyObject *boundary = PyDict_GetItemString(input_dict, "boundary");
        if (!boundary) { PyErr_SetString(PyExc_KeyError, "route-A query needs boundary"); return nullptr; }
        in.boundary = PyFloat_AsDouble(boundary);
        if (PyErr_Occurred()) return nullptr;
        if (!std::isfinite(in.boundary) || in.boundary < 0.)
            throw std::invalid_argument("route-A boundary tolerance must be finite and nonnegative");
        if (in.fps_axes[0]<0 || in.fps_axes[0]>2 || in.fps_axes[1]<0 || in.fps_axes[1]>2 ||
            in.fps_axes[2]<0 || in.fps_axes[2]>2 || in.fps_axes[0]==in.fps_axes[1] ||
            in.fps_axes[0]==in.fps_axes[2] || in.fps_axes[1]==in.fps_axes[2])
            throw std::invalid_argument("route-A FPS axes must be a permutation of 0,1,2");
#define A_OUTPUT(member, key, type, cast_type, ...) \
        view = pins.get(output_dict, key, type, {__VA_ARGS__}, true); \
        if (!view) return nullptr; \
        out.member = static_cast<cast_type *>(view->buf)
        A_OUTPUT(indices, "indices", BufferType::I64, Row, in.count, 128);
        A_OUTPUT(mask, "mask", BufferType::Bool, unsigned char, in.count, 128);
        A_OUTPUT(descriptor, "descriptor", BufferType::F64, double, in.count, 128, 8);
        A_OUTPUT(counts, "counts", BufferType::I64, Row, in.count, 3);
        A_OUTPUT(summary, "summary", BufferType::F64, double, in.count, 8);
        A_OUTPUT(angular_support, "angular_support", BufferType::I64, Row, in.count);
        A_OUTPUT(uncertain, "uncertain", BufferType::Bool, unsigned char, in.count);
#undef A_OUTPUT
        Row previous = -1;
        for (Py_ssize_t i=0; i<in.selected_count; ++i) {
            const auto q = in.selected[i];
            if (q <= previous || q >= in.count)
                throw std::invalid_argument("route-A selected rows must be ascending, unique and in range");
            previous = q;
            for (int k=0; k<3; ++k)
                if (!std::isfinite(in.points[3*q+k]) || !std::isfinite(in.normals[3*q+k]))
                    throw std::invalid_argument("route-A selected poses must be finite");
            const double norm2 = route_a_dot(in.normals+3*q, in.normals+3*q);
            if (std::abs(norm2-1.) > 1e-10)
                throw std::invalid_argument("route-A selected normals must be normalized");
        }
        for (int k=0; k<3; ++k)
            if (!std::isfinite(in.gravity[k]))
                throw std::invalid_argument("route-A gravity must be finite");
        for (Py_ssize_t row=1; row<in.source_count; ++row)
            if (in.source_ids[row] < in.source_ids[row-1]) { in.source_ids_ordered = false; break; }
        const int active = static_cast<int>(std::min<Py_ssize_t>(workers, in.selected_count));
        std::vector<std::exception_ptr> errors(static_cast<std::size_t>(active));
        std::vector<std::thread> threads;
        threads.reserve(static_cast<std::size_t>(active));
        std::exception_ptr launch_error;
        Py_BEGIN_ALLOW_THREADS
        auto run_worker = [&](int worker) {
            try {
                RouteAScratch scratch;
                for (Py_ssize_t i=worker; i<in.selected_count; i+=active)
                    route_a_process(in.selected[i], *grid, in, out, scratch);
            } catch (...) { errors[worker] = std::current_exception(); }
        };
        try {
            if (active == 1) run_worker(0);
            else for (int worker=0; worker<active; ++worker) threads.emplace_back(run_worker, worker);
        } catch (...) { launch_error = std::current_exception(); }
        for (auto &thread : threads) thread.join();
        Py_END_ALLOW_THREADS
        if (launch_error) { translate_exception(launch_error); return nullptr; }
        for (const auto &error : errors)
            if (error) { translate_exception(error); return nullptr; }
        Py_RETURN_NONE;
    } catch (...) { translate_exception(std::current_exception()); return nullptr; }
}
