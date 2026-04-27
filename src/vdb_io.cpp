/******************************************************************************
 * vdb_io.cpp  -  Phase 5a-0: OpenVDB output scaffolding
 ******************************************************************************/
#include "vdb_io.h"

#include <filesystem>
#include <iomanip>
#include <sstream>
#include <system_error>

#include <openvdb/io/File.h>
#include <openvdb/openvdb.h>
#include <openvdb/tools/Prune.h>

namespace {

std::string makeLevelFilePath(const std::string& basePath, int level)
{
    std::ostringstream oss;
    oss << basePath << "_L" << level << ".vdb";
    return oss.str();
}

void writeGridFile(const openvdb::GridBase::Ptr& grid, const std::string& filePath)
{
    openvdb::GridPtrVec grids;
    grids.push_back(grid);

    openvdb::io::File file(filePath);
    file.write(grids);
    file.close();
}

} // namespace

namespace VdbIO {

void writeFrame(const FlipState&,
                const FlipGridBundle& grid,
                const FlipConfig& cfg,
                int frame)
{
    static bool initialized = false;
    if(!initialized)
    {
        openvdb::initialize();
        initialized = true;
    }

    if(cfg.outputDir.empty()) return;

    const int outputStride = (cfg.outputStride > 0) ? cfg.outputStride : 1;
    if((frame % outputStride) != 0) return;

    std::ostringstream frameName;
    frameName << "frame_" << std::setw(4) << std::setfill('0') << frame;

    const std::filesystem::path frameDir =
        std::filesystem::path(cfg.outputDir) / frameName.str();

    std::error_code ec;
    std::filesystem::create_directories(frameDir, ec);
    if(ec) return;

    if(cfg.outputDensity)
        writeDensityVDB(grid, cfg, (frameDir / "density").string());

    // velocity: explicit flag, OR force-emit when outputVelocityForBlend
    // is set together with any other channel (so Houdini can interpolate
    // between keyframes by advecting through this velocity field).
    const bool anyOther = cfg.outputDensity || cfg.outputSurface || cfg.outputPressure;
    if(cfg.outputVelocity || (cfg.outputVelocityForBlend && anyOther))
        writeVelocityVDB(grid, cfg, (frameDir / "velocity").string());

    if(cfg.outputSurface)
        writeSurfaceVDB(grid, cfg, (frameDir / "surface").string());
    if(cfg.outputPressure)
        writePressureVDB(grid, cfg, (frameDir / "pressure").string());
}

void writeDensityVDB(const FlipGridBundle& grid,
                     const FlipConfig& cfg,
                     const std::string& basePath)
{
    if(!grid.msbg || !grid.mass) return;

    MSBG::MultiresSparseGrid* msbg = grid.msbg;
    const int nLevels = msbg->getNumLevels();

    for(int level=0; level<nLevels; ++level)
    {
        SBG::SparseGrid<float>* sgMass =
            (level == 0) ? grid.mass : msbg->getFloatChannel(MSBG::CH_FLOAT_1, level);
        if(!sgMass) continue;

        openvdb::FloatGrid::Ptr vdbGrid = openvdb::FloatGrid::create(0.0f);
        vdbGrid->setName("density");
        vdbGrid->setGridClass(openvdb::GRID_FOG_VOLUME);
        vdbGrid->setTransform(
            openvdb::math::Transform::createLinearTransform(double(1 << level)));
        if(cfg.useFloat16) vdbGrid->setSaveFloatAsHalf(true);

        openvdb::FloatGrid::Accessor accessor = vdbGrid->getAccessor();
        const int bsx = sgMass->bsx();
        const int bsx2 = bsx * bsx;
        const int nVoxels = sgMass->nVoxelsInBlock();

        for(int bid=0; bid<sgMass->nBlocks(); ++bid)
        {
            if(msbg->getBlockLevel(bid) != level) continue;

            float* blockData = sgMass->getBlockDataPtr(bid);
            if(!blockData) continue;

            int bx, by, bz;
            sgMass->getBlockCoordsById(bid, bx, by, bz);

            const int x0 = bx * bsx;
            const int y0 = by * bsx;
            const int z0 = bz * bsx;

            for(int vid=0; vid<nVoxels; ++vid)
            {
                const float value = blockData[vid];
                if(value < cfg.massEps) continue;

                const int vx = vid % bsx;
                const int vy = (vid / bsx) % bsx;
                const int vz = vid / bsx2;

                accessor.setValue(openvdb::Coord(x0 + vx, y0 + vy, z0 + vz), value);
            }
        }

        writeGridFile(vdbGrid, makeLevelFilePath(basePath, level));
    }
}

void writeVelocityVDB(const FlipGridBundle& grid,
                      const FlipConfig& cfg,
                      const std::string& basePath)
{
    if(!grid.msbg || !grid.vel || !grid.mass) return;

    MSBG::MultiresSparseGrid* msbg = grid.msbg;
    const int nLevels = msbg->getNumLevels();

    for(int level=0; level<nLevels; ++level)
    {
        SBG::SparseGrid<Vec3Float>* sgVel =
            (level == 0) ? grid.vel : msbg->getVecChannel(MSBG::CH_VEC3_4, level);
        SBG::SparseGrid<float>* sgMass =
            (level == 0) ? grid.mass : msbg->getFloatChannel(MSBG::CH_FLOAT_1, level);
        if(!sgVel || !sgMass) continue;

        openvdb::Vec3SGrid::Ptr vdbGrid =
            openvdb::Vec3SGrid::create(openvdb::Vec3s(0.0f, 0.0f, 0.0f));
        vdbGrid->setName("velocity");
        vdbGrid->setGridClass(openvdb::GRID_STAGGERED);
        vdbGrid->setTransform(
            openvdb::math::Transform::createLinearTransform(double(1 << level)));
        if(cfg.useFloat16) vdbGrid->setSaveFloatAsHalf(true);

        openvdb::Vec3SGrid::Accessor accessor = vdbGrid->getAccessor();
        const int bsx = sgVel->bsx();
        const int bsx2 = bsx * bsx;
        const int nVoxels = sgVel->nVoxelsInBlock();

        for(int bid=0; bid<sgVel->nBlocks(); ++bid)
        {
            if(msbg->getBlockLevel(bid) != level) continue;

            Vec3Float* velData = sgVel->getBlockDataPtr(bid);
            float* massData = sgMass->getBlockDataPtr(bid);
            if(!velData || !massData) continue;

            int bx, by, bz;
            sgVel->getBlockCoordsById(bid, bx, by, bz);

            const int x0 = bx * bsx;
            const int y0 = by * bsx;
            const int z0 = bz * bsx;

            for(int vid=0; vid<nVoxels; ++vid)
            {
                if(massData[vid] < cfg.massEps) continue;

                const Vec3Float& value = velData[vid];
                const int vx = vid % bsx;
                const int vy = (vid / bsx) % bsx;
                const int vz = vid / bsx2;

                accessor.setValue(
                    openvdb::Coord(x0 + vx, y0 + vy, z0 + vz),
                    openvdb::Vec3s(value.x, value.y, value.z));
            }
        }

        writeGridFile(vdbGrid, makeLevelFilePath(basePath, level));
    }
}

void writeSurfaceVDB(const FlipGridBundle& grid,
                     const FlipConfig& cfg,
                     const std::string& basePath)
{
    if(!grid.msbg || !grid.beta || !grid.mass) return;

    MSBG::MultiresSparseGrid* msbg = grid.msbg;
    SBG::SparseGrid<Vec3Float>* sgBeta = grid.beta;
    SBG::SparseGrid<float>*     sgMass = grid.mass;

    const float RHO_L    = cfg.rhoL;
    const float RHO_G    = cfg.rhoG;
    const float MASS_EPS = cfg.massEps;
    const float PHI0     = cfg.narrowBandPhi;
    const float HALF_BAND = cfg.narrowBandWidth;
    const float DENOM    = std::max(1e-6f, RHO_L - RHO_G);

    openvdb::FloatGrid::Ptr sdf = openvdb::FloatGrid::create(HALF_BAND);
    sdf->setName("surface");
    sdf->setGridClass(openvdb::GRID_LEVEL_SET);
    sdf->setTransform(openvdb::math::Transform::createLinearTransform(1.0));
    if(cfg.useFloat16) sdf->setSaveFloatAsHalf(true);

    openvdb::FloatGrid::Accessor acc = sdf->getAccessor();
    const int bsx = sgBeta->bsx();
    const int bsx2 = bsx * bsx;
    const int nVoxels = sgBeta->nVoxelsInBlock();

    for(int bid=0; bid<sgBeta->nBlocks(); ++bid)
    {
        if(msbg->getBlockLevel(bid) != 0) continue;

        Vec3Float* betaData = sgBeta->getBlockDataPtr(bid);
        float*     massData = sgMass->getBlockDataPtr(bid);
        if(!betaData || !massData) continue;

        int bx, by, bz;
        sgBeta->getBlockCoordsById(bid, bx, by, bz);
        const int x0 = bx * bsx;
        const int y0 = by * bsx;
        const int z0 = bz * bsx;

        for(int vid=0; vid<nVoxels; ++vid)
        {
            if(massData[vid] < MASS_EPS) continue;
            const float beta = betaData[vid].x;
            if(!(beta > 1e-9f)) continue;
            const float rho = 1.0f / beta;
            const float phi = (rho - RHO_G) / DENOM;
            const float sd  = (PHI0 - phi);
            if(!(std::fabs(sd) < HALF_BAND)) continue;

            const int vx = vid % bsx;
            const int vy = (vid / bsx) % bsx;
            const int vz = vid / bsx2;
            acc.setValue(openvdb::Coord(x0 + vx, y0 + vy, z0 + vz), sd);
        }
    }

    openvdb::tools::pruneInactive(sdf->tree());
    writeGridFile(sdf, makeLevelFilePath(basePath, 0));
}

void writePressureVDB(const FlipGridBundle& grid,
                      const FlipConfig& cfg,
                      const std::string& basePath)
{
    if(!grid.msbg || !grid.pressure || !grid.mass) return;

    MSBG::MultiresSparseGrid* msbg = grid.msbg;
    const int nLevels = msbg->getNumLevels();

    for(int level=0; level<nLevels; ++level)
    {
        SBG::SparseGrid<float>* sgP =
            (level == 0) ? grid.pressure : msbg->getFloatChannel(MSBG::CH_PRESSURE, level);
        SBG::SparseGrid<float>* sgMass =
            (level == 0) ? grid.mass : msbg->getFloatChannel(MSBG::CH_FLOAT_1, level);
        if(!sgP || !sgMass) continue;

        openvdb::FloatGrid::Ptr vdbGrid = openvdb::FloatGrid::create(0.0f);
        vdbGrid->setName("pressure");
        vdbGrid->setGridClass(openvdb::GRID_UNKNOWN);
        vdbGrid->setTransform(
            openvdb::math::Transform::createLinearTransform(double(1 << level)));

        openvdb::FloatGrid::Accessor accessor = vdbGrid->getAccessor();
        const int bsx = sgP->bsx();
        const int bsx2 = bsx * bsx;
        const int nVoxels = sgP->nVoxelsInBlock();

        for(int bid=0; bid<sgP->nBlocks(); ++bid)
        {
            if(msbg->getBlockLevel(bid) != level) continue;

            float* pData = sgP->getBlockDataPtr(bid);
            float* massData = sgMass->getBlockDataPtr(bid);
            if(!pData || !massData) continue;

            int bx, by, bz;
            sgP->getBlockCoordsById(bid, bx, by, bz);
            const int x0 = bx * bsx;
            const int y0 = by * bsx;
            const int z0 = bz * bsx;

            for(int vid=0; vid<nVoxels; ++vid)
            {
                if(massData[vid] < cfg.massEps) continue;
                const float p = pData[vid];
                if(p == 0.0f) continue;

                const int vx = vid % bsx;
                const int vy = (vid / bsx) % bsx;
                const int vz = vid / bsx2;
                accessor.setValue(openvdb::Coord(x0 + vx, y0 + vy, z0 + vz), p);
            }
        }

        writeGridFile(vdbGrid, makeLevelFilePath(basePath, level));
    }
}

} // namespace VdbIO
