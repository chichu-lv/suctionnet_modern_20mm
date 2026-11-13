// Temporary exact CPU spatial-index prototype. Not repository product code.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <vector>

struct Grid {
    const double *xyz;
    std::int64_t n, low[3], dims[3];
    double cell;
    std::vector<std::int64_t> offsets, rows, scratch;
    std::int64_t cell_id(std::int64_t x, std::int64_t y, std::int64_t z) const {
        return ((x-low[0])*dims[1]+y-low[1])*dims[2]+z-low[2];
    }
};
static char last_error[512] = {0};

extern "C" const char *grid_last_error() { return last_error; }

extern "C" void *grid_create(const double *xyz, std::int64_t n, double cell) {
    Grid *g = nullptr;
    try {
        g = new Grid;
        g->xyz=xyz; g->n=n; g->cell=cell;
        std::int64_t high[3];
        for (int d=0;d<3;++d) {
            g->low[d]=high[d]=(std::int64_t)std::floor(xyz[d]/cell);
        }
        for (std::int64_t i=1;i<n;++i) for (int d=0;d<3;++d) {
            const auto k=(std::int64_t)std::floor(xyz[3*i+d]/cell);
            g->low[d]=std::min(g->low[d],k); high[d]=std::max(high[d],k);
        }
        std::int64_t nc=1;
        for (int d=0;d<3;++d) {g->dims[d]=high[d]-g->low[d]+1; nc*=g->dims[d];}
        g->offsets.assign((std::size_t)nc+1,0);
        std::vector<std::int64_t> cell_of((std::size_t)n);
        for (std::int64_t i=0;i<n;++i) {
            const auto k=g->cell_id((std::int64_t)std::floor(xyz[3*i]/cell),
                (std::int64_t)std::floor(xyz[3*i+1]/cell),
                (std::int64_t)std::floor(xyz[3*i+2]/cell));
            cell_of[(std::size_t)i]=k; ++g->offsets[(std::size_t)k+1];
        }
        for (std::int64_t k=0;k<nc;++k) g->offsets[(std::size_t)k+1]+=g->offsets[(std::size_t)k];
        auto cursor=g->offsets;
        g->rows.resize((std::size_t)n);
        for (std::int64_t i=0;i<n;++i) g->rows[(std::size_t)cursor[(std::size_t)cell_of[(std::size_t)i]]++]=i;
        return g;
    } catch (const std::exception &e) {
        std::snprintf(last_error,sizeof(last_error),"%s",e.what()); delete g; return nullptr;
    }
}

extern "C" void grid_destroy(void *handle) { delete static_cast<Grid*>(handle); }

// Result points into this handle's scratch; caller copies before its next call.
// Radius is already expanded by caller using the current production +1e-8.
extern "C" std::int64_t grid_ball(void *handle, const double *center, double radius,
                                   const std::int64_t **output) {
    auto *g=static_cast<Grid*>(handle);
    try {
        g->scratch.clear();
        std::int64_t lo[3],hi[3];
        for (int d=0;d<3;++d) {
            lo[d]=std::max(g->low[d],(std::int64_t)std::floor((center[d]-radius)/g->cell));
            hi[d]=std::min(g->low[d]+g->dims[d]-1,(std::int64_t)std::floor((center[d]+radius)/g->cell));
        }
        const double radius2=radius*radius;
        for (auto x=lo[0];x<=hi[0];++x) for (auto y=lo[1];y<=hi[1];++y) for (auto z=lo[2];z<=hi[2];++z) {
            const auto k=g->cell_id(x,y,z);
            for (auto j=g->offsets[(std::size_t)k];j<g->offsets[(std::size_t)k+1];++j) {
                const auto row=g->rows[(std::size_t)j];
                const double dx=g->xyz[3*row]-center[0];
                const double dy=g->xyz[3*row+1]-center[1];
                const double dz=g->xyz[3*row+2]-center[2];
                if ((dx*dx+dy*dy)+dz*dz <= radius2) g->scratch.push_back(row);
            }
        }
        std::sort(g->scratch.begin(),g->scratch.end());
        *output=g->scratch.data();
        return (std::int64_t)g->scratch.size();
    } catch (const std::exception &e) {
        std::snprintf(last_error,sizeof(last_error),"%s",e.what()); return -1;
    }
}
