/******************************************************************************
 * vdb_io.h  -  Phase 5a-0: OpenVDB output scaffolding
 ******************************************************************************/
#ifndef VDB_IO_H
#define VDB_IO_H

#include <string>

#include "flip_sim.h"

namespace VdbIO {

void writeFrame(const FlipState& state,
                const FlipGridBundle& grid,
                const FlipConfig& cfg,
                int frame);

void writeDensityVDB(const FlipGridBundle& grid,
                     const FlipConfig& cfg,
                     const std::string& basePath);

void writeVelocityVDB(const FlipGridBundle& grid,
                      const FlipConfig& cfg,
                      const std::string& basePath);

void writeSurfaceVDB(const FlipGridBundle& grid,
                     const FlipConfig& cfg,
                     const std::string& basePath);

void writePressureVDB(const FlipGridBundle& grid,
                      const FlipConfig& cfg,
                      const std::string& basePath);

void writeParticlesBGEO(const FlipState& state,
                        const FlipConfig& cfg,
                        const std::string& basePath);

} // namespace VdbIO

#endif // VDB_IO_H
