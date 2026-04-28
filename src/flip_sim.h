/******************************************************************************
 * flip_sim.h  -  Phase-Field FLIP simulation on MSBG
 *
 * Phase 0: FlipSimulation class (HDK-ready core library)
 *
 * チャンネル:
 *   CH_VEC3_4       現在速度 (x=u面, y=v面, z=w面)
 *   CH_VEC3_2       投影前速度
 *   CH_VEC3_3       フェース β = 1/ρ (x=u面, y=v面, z=w面)
 *   CH_FLOAT_1      cell mass (全相合算)
 *   CH_FLOAT_2      PCG残差 r
 *   CH_FLOAT_3      前処理後 z = M^{-1} r
 *   CH_FLOAT_4      探索方向 d
 *   CH_FLOAT_6      q = A*d
 *   CH_DIVERGENCE
 *   CH_PRESSURE
 ******************************************************************************/
#ifndef FLIP_SIM_H
#define FLIP_SIM_H

#include <vector>
#include <memory>
#include <string>
#include "msbg.h"
#include "camera_io.h"

//=== Solver selection ========================================================
enum class PressureSolverKind { MIC0_PCG, HYPRE_AMG_PCG, MSBG_VCYCLE_PCG };

//=== FlipConfig ==============================================================
struct FlipConfig {
    float gravity           = -9.8f;
    float dtMax             = 0.05f;
    float dtMin             = 1e-3f;
    float cflNumber         = 3.0f;
    int   particlesPerVoxel = 8;
    float flipAlpha         = 0.95f;
    float massEps           = 1e-5f;
    float pcgTol            = 5e-4f;
    int   pcgMaxIter        = 500;
    float rhoL              = 1000.0f;
    float rhoG              = 1.0f;
    float alphaPhi          = 1.0f;
    float betaMin           = 1e-6f;
    PressureSolverKind solverKind = PressureSolverKind::MIC0_PCG;  // paper baseline
    bool  enableDebugSlice  = true;

    // Phase 5a: VDB / BGEO output
    std::string outputDir = "";
    bool  outputDensity     = true;
    bool  outputVelocity    = true;
    bool  outputPressure    = false;
    bool  outputSurface     = true;
    bool  outputParticles   = false;
    bool  useFloat16        = true;
    float narrowBandPhi     = 0.5f;
    float narrowBandWidth   = 3.0f;
    int   outputStride      = 1;
    bool  outputVelocityForBlend = true;
    bool  useFrustumRefinement = false;
    std::string cameraPath  = "";
    int   imageHeight       = 1080;
    float frustumLodScale   = 1.0f;

    // Phase 5b: Houdini -> MSBG collider input
    std::string colliderPath    = "";
    float       colliderEps     = 0.5f;   // SDF この値以下で solid 判定
    float       colliderPushOut = 1.0f;   // 粒子を表面+N voxel 外側に押し出す
};

//=== FlipParticle ============================================================
struct FlipParticle { Vec3Float pos, vel; int phase; /* 0=liquid 1=air */ };

//=== FlipState ===============================================================
struct FlipState {
    std::vector<FlipParticle> particles;
    std::vector<int> activeBlocks;
    std::vector<int> refinementMap;
    std::vector<CameraIO::CameraFrame> cameraPath;
    int   step = 0;
    float time = 0.0f;
};

//=== FlipGridBundle ==========================================================
struct FlipGridBundle {
    MSBG::MultiresSparseGrid* msbg = nullptr;
    int sx = 0, sy = 0, sz = 0;

    SBG::SparseGrid<Vec3Float>* vel      = nullptr;
    SBG::SparseGrid<Vec3Float>* velOld   = nullptr;
    SBG::SparseGrid<Vec3Float>* beta     = nullptr;
    SBG::SparseGrid<float>*     mass     = nullptr;
    SBG::SparseGrid<float>*     r        = nullptr;
    SBG::SparseGrid<float>*     z        = nullptr;
    SBG::SparseGrid<float>*     d        = nullptr;
    SBG::SparseGrid<float>*     q        = nullptr;
    SBG::SparseGrid<float>*     div      = nullptr;
    SBG::SparseGrid<float>*     pressure = nullptr;
    SBG::SparseGrid<float>*     tmpRelax = nullptr;

    // Phase 5b: collider SDF (dense, sx*sy*sz floats; <0 = inside solid)
    std::vector<float> colliderSdf;
    bool hasCollider = false;

    // Phase 5c-ext: external force field
    //   Layout: (ix + iy*sx + iz*sx*sy)*3 + {0,1,2} for {fx, fy, fz}
    //   Value: per-cell acceleration in voxel/sec^2
    //   Persists until cleared or replaced.
    std::vector<float> externalForce;
    bool hasExternalForce = false;

    void prepareChannels();
    bool rebind();
    void touchAllBlocks();
};

//=== PressureSystem ==========================================================
struct PressureSystem {
    int n  = 0;
    int nx = 0, ny = 0, nz = 0;
    std::vector<int>   rowPtr;
    std::vector<int>   colInd;
    std::vector<float> val;
    std::vector<int>   cellToRow;
    std::vector<int>   rowToCell;
    std::vector<float> rhs;
    std::vector<float> sol;
    std::vector<short> rowIx, rowIy, rowIz;
    std::vector<float> betaXm, betaYm, betaZm;
    std::vector<int>   diagPos, xmPos, xpPos, ymPos, ypPos, zmPos, zpPos;
    int  patternRevision = 0;
    bool topologyChanged = true;
    bool patternValid    = false;
};

//=== HypreState (forward declaration, defined in flip_sim.cpp) ===============
struct HypreState;

//=== FlipSimulation ==========================================================
class FlipSimulation {
public:
    FlipSimulation(MSBG::MultiresSparseGrid& msbg, FlipConfig cfg = {});
    ~FlipSimulation();

    int runDamBreak(int nSteps);
    static int runStandaloneDamBreak(int resolution, int blockSize, int nSteps);
    static int runStandaloneDamBreak(int resolution, int blockSize, int nSteps, FlipConfig cfg);

    // Phase 5c-ext: step-by-step driving (Python / interactive use)
    bool initialize();
    bool stepOnce();
    void addParticles(const std::vector<Vec3Float>& pos,
                      const std::vector<Vec3Float>& vel,
                      const std::vector<int>& phase);

    const FlipConfig&    config() const { return cfg_; }
    FlipConfig&          config()       { return cfg_; }
    const FlipState&     state()  const { return state_; }
    FlipState&           state()        { return state_; }
    const FlipGridBundle& grid()  const { return grid_; }
    FlipGridBundle&      grid()         { return grid_; }

private:
    FlipConfig                  cfg_;
    FlipState                   state_;
    FlipGridBundle              grid_;
    PressureSystem              pressure_;
    std::unique_ptr<HypreState> hypre_;

    void  initializeParticles();
    bool  updateRefinementMap();
    float computeDt() const;

    void particleToGrid();
    void applyGravity(float dt);
    void pressureProjection(float dt);
    void gridToParticle();
    void advectParticles(float dt);

    // Pressure solver internals
    void rebuildPressurePattern();
    void updatePressureNumerics();
    void buildPressureSystem(float dt);
    void scatterPressureToGrid();
    void buildMIC0Compact(std::vector<float>& precon);
    void applyMIC0Compact(const std::vector<float>& precon,
                          const float* r, float* z);
    void solvePressureCompactPCG();
    void solvePressureHypreAMG();

    // MSBG level-aware MAC interpolation
    float interpCompMSBG(SBG::SparseGrid<Vec3Float> *sg,
                         float px, float py, float pz,
                         float ox, float oy, float oz, int comp,
                         int chanId) const;
    // 3-component simultaneous interpolation (reduces block lookups)
    Vec3Float interpVec3MSBG(SBG::SparseGrid<Vec3Float> *sg,
                             float px, float py, float pz,
                             int chanId) const;
};

//=== FlipDebugOutput =========================================================
namespace FlipDebugOutput {
    void saveParticleSlice(const FlipState& state, int step,
                           int sx, int sy, int sz);
}

//=== Backward-compatible entry point =========================================
int flip_dam_break(int resolution, int blockSize, int nSteps);

#endif // FLIP_SIM_H
