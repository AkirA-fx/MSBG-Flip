/******************************************************************************
 * gen_sphere_sdf.cpp
 *
 * Phase 5b テスト用: 球の signed distance field を VDB ファイルに書き出す。
 * usage: gen_sphere_sdf <radius> <cx> <cy> <cz> <out.vdb>
 ******************************************************************************/
#include <cstdio>
#include <cstdlib>
#include <openvdb/openvdb.h>
#include <openvdb/tools/LevelSetSphere.h>

int main(int argc, char** argv)
{
    if(argc != 6) {
        std::fprintf(stderr,
            "usage: %s <radius> <cx> <cy> <cz> <out.vdb>\n", argv[0]);
        return 1;
    }
    const float radius = (float)std::atof(argv[1]);
    const float cx     = (float)std::atof(argv[2]);
    const float cy     = (float)std::atof(argv[3]);
    const float cz     = (float)std::atof(argv[4]);
    const char* path   = argv[5];

    openvdb::initialize();

    // narrow-band level set sphere、voxelSize=1.0、halfWidth=3 voxel
    auto sdf = openvdb::tools::createLevelSetSphere<openvdb::FloatGrid>(
        radius, openvdb::Vec3f(cx, cy, cz), /*voxelSize=*/1.0f, /*halfWidth=*/3.0f);
    sdf->setName("solid_sdf");

    openvdb::io::File file(path);
    openvdb::GridPtrVec grids;
    grids.push_back(sdf);
    file.write(grids);
    file.close();

    std::printf("Wrote sphere SDF: r=%.1f center=(%.1f,%.1f,%.1f) -> %s\n",
                radius, cx, cy, cz, path);
    return 0;
}
