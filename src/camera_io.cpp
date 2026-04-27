/******************************************************************************
 * camera_io.cpp  -  Phase 5a Step C: camera path I/O + frustum / LOD helpers
 ******************************************************************************/
#include "camera_io.h"

#include <cmath>
#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

namespace CameraIO {

namespace {

inline void vsub(const float a[3], const float b[3], float out[3])
{ out[0]=a[0]-b[0]; out[1]=a[1]-b[1]; out[2]=a[2]-b[2]; }

inline float vdot(const float a[3], const float b[3])
{ return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]; }

inline float vlen(const float a[3])
{ return std::sqrt(std::max(0.f, vdot(a,a))); }

inline void vnormalize(float a[3])
{
    float L = vlen(a);
    if(L > 1e-12f) { a[0]/=L; a[1]/=L; a[2]/=L; }
}

inline void vcross(const float a[3], const float b[3], float out[3])
{
    out[0] = a[1]*b[2] - a[2]*b[1];
    out[1] = a[2]*b[0] - a[0]*b[2];
    out[2] = a[0]*b[1] - a[1]*b[0];
}

} // anonymous namespace


size_t readCameraPath(const std::string& path,
                      std::vector<CameraFrame>& frames)
{
    frames.clear();
    if(path.empty()) return 0;

    std::ifstream is(path);
    if(!is) return 0;

    std::string line;
    while(std::getline(is, line))
    {
        // strip leading whitespace
        size_t p = line.find_first_not_of(" \t\r\n");
        if(p == std::string::npos) continue;
        if(line[p] == '#') continue;  // comment

        std::istringstream iss(line);
        CameraFrame cf;
        if(!(iss >> cf.frame
                 >> cf.eye[0]    >> cf.eye[1]    >> cf.eye[2]
                 >> cf.lookAt[0] >> cf.lookAt[1] >> cf.lookAt[2]
                 >> cf.up[0]     >> cf.up[1]     >> cf.up[2]
                 >> cf.fovY      >> cf.aspect
                 >> cf.nearZ     >> cf.farZ))
        {
            // 不正行はスキップ
            continue;
        }
        frames.push_back(cf);
    }
    return frames.size();
}


const CameraFrame* findCameraForFrame(
    const std::vector<CameraFrame>& frames, int frame)
{
    if(frames.empty()) return nullptr;
    const CameraFrame* best = &frames.front();
    for(const auto& cf : frames) {
        if(cf.frame <= frame) best = &cf;
        else break;
    }
    return best;
}


// 6 面平面テスト。視錐台は near/far + 4 サイド。
// near/far 平面はカメラ前方 (forward = lookAt - eye) を基準に判定。
// 4 サイド平面は forward と up から right を組み立て、フェース法線を作る。
bool inFrustum(const CameraFrame& cam, const float p[3])
{
    float fwd[3];   vsub(cam.lookAt, cam.eye, fwd);  vnormalize(fwd);
    float up[3]   = { cam.up[0], cam.up[1], cam.up[2] };
    float right[3]; vcross(fwd, up, right);          vnormalize(right);
    // up を right と fwd に直交化
    vcross(right, fwd, up);                          vnormalize(up);

    float v[3]; vsub(p, cam.eye, v);
    float zf = vdot(v, fwd);
    if(zf < cam.nearZ || zf > cam.farZ) return false;

    const float halfV = std::tan(0.5f * cam.fovY) * zf;
    const float halfH = halfV * cam.aspect;
    float xr = vdot(v, right);
    float yu = vdot(v, up);
    if(xr < -halfH || xr > halfH) return false;
    if(yu < -halfV || yu > halfV) return false;
    return true;
}


int levelFromPixelSize(const CameraFrame& cam,
                       const float p[3],
                       float blockSizeWorld,
                       int   imageHeight,
                       float lodScale,
                       int   maxLevel)
{
    if(maxLevel <= 0) return 0;
    if(imageHeight <= 0) return 0;
    if(blockSizeWorld <= 0.f) return 0;

    float v[3]; vsub(p, cam.eye, v);
    const float dist = vlen(v);
    if(!(dist > 0.f)) return 0;

    // 1 px が world 上で表す長さ
    const float pixelSize =
        2.0f * dist * std::tan(0.5f * cam.fovY) / float(imageHeight);

    // blockSizeWorld * 2^L <= pixelSize * lodScale を満たす最大の L を求める
    int L = 0;
    float bs = blockSizeWorld;
    while(L < maxLevel && bs * 2.0f <= pixelSize * lodScale) {
        bs *= 2.0f;
        L++;
    }
    return L;
}

} // namespace CameraIO
