/******************************************************************************
 * vdb_in.cpp  -  Phase 5b: Houdini -> MSBG VDB input (collider SDF)
 ******************************************************************************/
#include "vdb_in.h"

#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>

namespace VdbIn {

namespace {
bool g_vdbInitialized = false;
}

bool readColliderSDF(const std::string& path,
                     int sx, int sy, int sz,
                     std::vector<float>& sdfOut)
{
    if(path.empty() || sx<=0 || sy<=0 || sz<=0) return false;

    if(!g_vdbInitialized) { openvdb::initialize(); g_vdbInitialized = true; }

    openvdb::GridPtrVecPtr grids;
    try {
        openvdb::io::File file(path);
        file.open();
        grids = file.getGrids();
        file.close();
    } catch(const std::exception&) {
        return false;
    }
    if(!grids || grids->empty()) return false;

    // 1. "solid_sdf" 名の FloatGrid を優先
    openvdb::FloatGrid::Ptr src;
    for(const auto& g : *grids) {
        if(g && g->getName() == "solid_sdf" && g->isType<openvdb::FloatGrid>()) {
            src = openvdb::gridPtrCast<openvdb::FloatGrid>(g);
            break;
        }
    }
    // 2. fallback: 最初の FloatGrid
    if(!src) {
        for(const auto& g : *grids) {
            if(g && g->isType<openvdb::FloatGrid>()) {
                src = openvdb::gridPtrCast<openvdb::FloatGrid>(g);
                break;
            }
        }
    }
    if(!src) return false;

    // background (= 外側) で初期化
    const float bg = src->background();
    sdfOut.assign(size_t(sx)*size_t(sy)*size_t(sz), bg);

    // 値を flat 配列に転送
    for(auto it = src->cbeginValueOn(); it; ++it) {
        const openvdb::Coord c = it.getCoord();
        if(c.x() < 0 || c.x() >= sx) continue;
        if(c.y() < 0 || c.y() >= sy) continue;
        if(c.z() < 0 || c.z() >= sz) continue;
        const size_t idx = size_t(c.x())
                         + size_t(c.y()) * size_t(sx)
                         + size_t(c.z()) * size_t(sx) * size_t(sy);
        sdfOut[idx] = it.getValue();
    }
    return true;
}

} // namespace VdbIn
