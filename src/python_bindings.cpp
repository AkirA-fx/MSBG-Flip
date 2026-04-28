/******************************************************************************
 * python_bindings.cpp  -  Phase 5c: pybind11 bindings for MSBG-Flip
 *
 * Python から MSBG ベースの 2-phase FLIP シミュレーターを駆動できるように
 * する薄いラッパー。
 *
 *   import msbg_flip
 *   sim = msbg_flip.FlipSimulation(64, 16)
 *   sim.set_collider(np.zeros((64,64,64), dtype=np.float32))
 *   sim.run(20)
 *   density = sim.get_density()  # ndarray (sx, sy, sz) float32
 ******************************************************************************/
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <string>

#include "msbg.h"
#include "flip_sim.h"

namespace py = pybind11;

// MSBG library expects this global (defined in main.cpp for the demo binary).
// We replicate the definition here so the .pyd can link standalone.
int nMaxThreads = -1;

namespace {

// MSBG MultiresSparseGrid + FlipSimulation の lifetime を一括管理する
// Python 側用のラッパー。
class PyFlipSim {
public:
    PyFlipSim(int resolution, int blockSize)
        : resolution_(resolution), blockSize_(blockSize)
    {
        if(resolution <= 0 || blockSize <= 0)
            throw std::invalid_argument("resolution / blockSize must be positive");
        if(blockSize != 16 && blockSize != 32)
            throw std::invalid_argument("blockSize must be 16 or 32");

        const int sx = ALIGN(resolution, blockSize);
        sx_ = sy_ = sz_ = sx;

        msbg_ = MSBG::MultiresSparseGrid::create(
            "FLIP_MAC", sx, sx, sx, blockSize, -1, 0, -1, 0, NULL, NULL, 3);
        if(!msbg_)
            throw std::runtime_error("MultiresSparseGrid::create failed");

        sim_ = std::make_unique<FlipSimulation>(*msbg_, FlipConfig{});
    }

    ~PyFlipSim()
    {
        sim_.reset();
        if(msbg_) {
            MSBG::MultiresSparseGrid::destroy(msbg_);
            msbg_ = nullptr;
        }
    }

    PyFlipSim(const PyFlipSim&) = delete;
    PyFlipSim& operator=(const PyFlipSim&) = delete;

    int sx() const { return sx_; }
    int sy() const { return sy_; }
    int sz() const { return sz_; }
    int resolution() const { return resolution_; }
    int blockSize()  const { return blockSize_; }

    // FlipConfig を kwargs 経由で更新するエンドポイント。
    // pybind11 では Python 側で sim.configure(rho_l=..., ...) と呼ぶ。
    void configure(py::kwargs kwargs)
    {
        FlipConfig& cfg = sim_->config();
        for(auto item : kwargs) {
            std::string key = py::str(item.first);
            const py::handle& val = item.second;

            if      (key == "rho_l")            cfg.rhoL = val.cast<float>();
            else if (key == "rho_g")            cfg.rhoG = val.cast<float>();
            else if (key == "alpha_phi")        cfg.alphaPhi = val.cast<float>();
            else if (key == "beta_min")         cfg.betaMin = val.cast<float>();
            else if (key == "gravity")          cfg.gravity = val.cast<float>();
            else if (key == "dt_max")           cfg.dtMax = val.cast<float>();
            else if (key == "dt_min")           cfg.dtMin = val.cast<float>();
            else if (key == "cfl")              cfg.cflNumber = val.cast<float>();
            else if (key == "particles_per_voxel") cfg.particlesPerVoxel = val.cast<int>();
            else if (key == "flip_alpha")       cfg.flipAlpha = val.cast<float>();
            else if (key == "mass_eps")         cfg.massEps = val.cast<float>();
            else if (key == "pcg_tol")          cfg.pcgTol = val.cast<float>();
            else if (key == "pcg_max_iter")     cfg.pcgMaxIter = val.cast<int>();
            else if (key == "use_float16")      cfg.useFloat16 = val.cast<bool>();
            else if (key == "narrow_band_phi")  cfg.narrowBandPhi = val.cast<float>();
            else if (key == "narrow_band_width") cfg.narrowBandWidth = val.cast<float>();
            else if (key == "output_stride")    cfg.outputStride = val.cast<int>();
            else if (key == "output_dir")       cfg.outputDir = val.cast<std::string>();
            else if (key == "enable_debug_slice") cfg.enableDebugSlice = val.cast<bool>();
            else if (key == "solver") {
                std::string s = val.cast<std::string>();
                if      (s == "MIC0_PCG")        cfg.solverKind = PressureSolverKind::MIC0_PCG;
                else if (s == "HYPRE_AMG_PCG")   cfg.solverKind = PressureSolverKind::HYPRE_AMG_PCG;
                else if (s == "MSBG_VCYCLE_PCG") cfg.solverKind = PressureSolverKind::MSBG_VCYCLE_PCG;
                else throw py::value_error("unknown solver: " + s);
            }
            else throw py::value_error("unknown configure key: " + key);
        }
    }

    // numpy ndarray (sx, sy, sz) float32 を collider SDF として登録
    void set_collider(py::array_t<float, py::array::c_style | py::array::forcecast> sdf,
                      float eps = 0.5f, float push_out = 1.0f)
    {
        if(sdf.ndim() != 3)
            throw py::value_error("collider sdf must be 3-dimensional");
        if(sdf.shape(0) != sx_ || sdf.shape(1) != sy_ || sdf.shape(2) != sz_)
            throw py::value_error("collider sdf shape must match (sx, sy, sz)");

        // FlipGridBundle.colliderSdf に flat 転送 (layout: ix + iy*sx + iz*sx*sy)
        FlipGridBundle& gb = sim_->grid();
        gb.colliderSdf.assign(size_t(sx_) * sy_ * sz_, 0.f);
        auto buf = sdf.unchecked<3>();
        for(int iz=0; iz<sz_; iz++)
            for(int iy=0; iy<sy_; iy++)
                for(int ix=0; ix<sx_; ix++) {
                    gb.colliderSdf[size_t(ix)
                                  + size_t(iy)*size_t(sx_)
                                  + size_t(iz)*size_t(sx_)*size_t(sy_)]
                        = buf(ix, iy, iz);
                }
        gb.hasCollider = true;

        FlipConfig& cfg = sim_->config();
        cfg.colliderEps = eps;
        cfg.colliderPushOut = push_out;
    }

    // numpy ndarray (sx, sy, sz, 3) float32 を外力 field として登録
    void set_external_force(
        py::array_t<float, py::array::c_style | py::array::forcecast> field)
    {
        if(field.ndim() != 4)
            throw py::value_error("external force must be 4-dimensional");
        if(field.shape(0) != sx_ || field.shape(1) != sy_ ||
           field.shape(2) != sz_ || field.shape(3) != 3)
            throw py::value_error("external force shape must be (sx, sy, sz, 3)");

        FlipGridBundle& gb = sim_->grid();
        gb.externalForce.assign(size_t(sx_) * sy_ * sz_ * 3, 0.f);
        auto buf = field.unchecked<4>();
        for(int iz=0; iz<sz_; iz++)
            for(int iy=0; iy<sy_; iy++)
                for(int ix=0; ix<sx_; ix++) {
                    const size_t base = (size_t(ix)
                                       + size_t(iy)*size_t(sx_)
                                       + size_t(iz)*size_t(sx_)*size_t(sy_)) * 3;
                    gb.externalForce[base + 0] = buf(ix, iy, iz, 0);
                    gb.externalForce[base + 1] = buf(ix, iy, iz, 1);
                    gb.externalForce[base + 2] = buf(ix, iy, iz, 2);
                }
        gb.hasExternalForce = true;
    }

    void clear_external_force()
    {
        FlipGridBundle& gb = sim_->grid();
        gb.externalForce.clear();
        gb.hasExternalForce = false;
    }

    // numpy (N,3) pos, (N,3) vel, (N,) phase を粒子配列に追加
    void add_particles(
        py::array_t<float, py::array::c_style | py::array::forcecast> pos,
        py::array_t<float, py::array::c_style | py::array::forcecast> vel,
        py::array_t<int,   py::array::c_style | py::array::forcecast> phase)
    {
        if(pos.ndim() != 2 || pos.shape(1) != 3)
            throw py::value_error("pos must have shape (N, 3)");
        if(vel.ndim() != 2 || vel.shape(1) != 3)
            throw py::value_error("vel must have shape (N, 3)");
        if(phase.ndim() != 1)
            throw py::value_error("phase must have shape (N,)");
        const py::ssize_t N = pos.shape(0);
        if(vel.shape(0) != N || phase.shape(0) != N)
            throw py::value_error("pos / vel / phase must agree on N");

        std::vector<Vec3Float> pv(N), vv(N);
        std::vector<int> ph(N);
        auto pbuf = pos.unchecked<2>();
        auto vbuf = vel.unchecked<2>();
        auto phbuf = phase.unchecked<1>();
        for(py::ssize_t i=0; i<N; i++) {
            pv[i] = Vec3Float(pbuf(i,0), pbuf(i,1), pbuf(i,2));
            vv[i] = Vec3Float(vbuf(i,0), vbuf(i,1), vbuf(i,2));
            ph[i] = phbuf(i);
        }
        sim_->addParticles(pv, vv, ph);
    }

    // step-by-step driving
    bool initialize() { return sim_->initialize(); }
    bool step()       { return sim_->stepOnce(); }
    int  step_number() const { return sim_->state().step; }

    // dam-break シーンを n ステップ走らせる。内部的に runDamBreak を呼ぶ。
    int run(int nSteps)
    {
        if(nSteps < 0) throw std::invalid_argument("nSteps must be >= 0");
        return sim_->runDamBreak(nSteps);
    }

    // --- result accessors ---------------------------------------------------

    // density (mass) grid を numpy(sx, sy, sz) float32 で返す
    py::array_t<float> get_density() const
    {
        py::array_t<float> arr({sx_, sy_, sz_});
        auto buf = arr.mutable_unchecked<3>();
        SBG::SparseGrid<float>* sg = sim_->grid().mass;
        if(!sg) {
            std::memset(arr.mutable_data(), 0, arr.nbytes());
            return arr;
        }
        for(int iz=0; iz<sz_; iz++)
            for(int iy=0; iy<sy_; iy++)
                for(int ix=0; ix<sx_; ix++)
                    buf(ix, iy, iz) = readF(sg, ix, iy, iz);
        return arr;
    }

    // velocity grid を numpy(sx, sy, sz, 3) float32 で返す。
    // MAC face-staggered の値をセル中央配置として「そのまま」返す
    // (Houdini Volume VOP に渡すなら staggered 解釈で OK)。
    py::array_t<float> get_velocity() const
    {
        py::array_t<float> arr({sx_, sy_, sz_, 3});
        auto buf = arr.mutable_unchecked<4>();
        SBG::SparseGrid<Vec3Float>* sg = sim_->grid().vel;
        if(!sg) {
            std::memset(arr.mutable_data(), 0, arr.nbytes());
            return arr;
        }
        for(int iz=0; iz<sz_; iz++)
            for(int iy=0; iy<sy_; iy++)
                for(int ix=0; ix<sx_; ix++) {
                    Vec3Float v = readVec(sg, ix, iy, iz);
                    buf(ix, iy, iz, 0) = v.x;
                    buf(ix, iy, iz, 1) = v.y;
                    buf(ix, iy, iz, 2) = v.z;
                }
        return arr;
    }

    // particles を {pos: (N, 3), vel: (N, 3), phase: (N,)} の dict で返す
    py::dict get_particles() const
    {
        const auto& parts = sim_->state().particles;
        const size_t N = parts.size();
        py::array_t<float> pos({N, size_t(3)});
        py::array_t<float> vel({N, size_t(3)});
        py::array_t<int>   phase({N});
        auto pbuf = pos.mutable_unchecked<2>();
        auto vbuf = vel.mutable_unchecked<2>();
        auto phbuf = phase.mutable_unchecked<1>();
        for(size_t i=0; i<N; i++) {
            pbuf(i,0) = parts[i].pos.x;
            pbuf(i,1) = parts[i].pos.y;
            pbuf(i,2) = parts[i].pos.z;
            vbuf(i,0) = parts[i].vel.x;
            vbuf(i,1) = parts[i].vel.y;
            vbuf(i,2) = parts[i].vel.z;
            phbuf(i)  = parts[i].phase;
        }
        py::dict d;
        d["pos"]   = pos;
        d["vel"]   = vel;
        d["phase"] = phase;
        return d;
    }

private:
    // sparse grid からの値取得ヘルパー。
    // bid/vid 計算は sgMass の bsx/nbx/nby から再現。
    static float readF(SBG::SparseGrid<float>* sg, int ix, int iy, int iz)
    {
        if(ix<0||iy<0||iz<0||ix>=sg->sx()||iy>=sg->sy()||iz>=sg->sz()) return 0.f;
        const int bl=sg->bsxLog2(),bm=sg->bsx()-1,nbx=sg->nbx(),nby=sg->nby();
        const int bsx=sg->bsx();
        int bid=(ix>>bl)+(iy>>bl)*nbx+(iz>>bl)*nbx*nby;
        int vid=(ix&bm)+(iy&bm)*bsx+(iz&bm)*bsx*bsx;
        float* d = sg->getBlockDataPtr(bid);
        return d ? d[vid] : 0.f;
    }
    static Vec3Float readVec(SBG::SparseGrid<Vec3Float>* sg, int ix, int iy, int iz)
    {
        if(ix<0||iy<0||iz<0||ix>=sg->sx()||iy>=sg->sy()||iz>=sg->sz())
            return Vec3Float(0,0,0);
        const int bl=sg->bsxLog2(),bm=sg->bsx()-1,nbx=sg->nbx(),nby=sg->nby();
        const int bsx=sg->bsx();
        int bid=(ix>>bl)+(iy>>bl)*nbx+(iz>>bl)*nbx*nby;
        int vid=(ix&bm)+(iy&bm)*bsx+(iz&bm)*bsx*bsx;
        Vec3Float* d = sg->getBlockDataPtr(bid);
        return d ? d[vid] : Vec3Float(0,0,0);
    }

    int resolution_;
    int blockSize_;
    int sx_, sy_, sz_;
    MSBG::MultiresSparseGrid* msbg_ = nullptr;
    std::unique_ptr<FlipSimulation> sim_;
};

} // anonymous namespace


PYBIND11_MODULE(msbg_flip, m) {
    m.doc() = "MSBG-Flip Python bindings (Phase 5c)";

    py::class_<PyFlipSim>(m, "FlipSimulation")
        .def(py::init<int, int>(),
             py::arg("resolution"), py::arg("block_size") = 16,
             "Create a 2-phase FLIP simulator backed by an MSBG grid.")
        .def("configure", &PyFlipSim::configure,
             "Update FlipConfig fields (rho_l, rho_g, cfl, solver, ...).")
        .def("set_collider", &PyFlipSim::set_collider,
             py::arg("sdf"), py::arg("eps") = 0.5f, py::arg("push_out") = 1.0f,
             "Register a (sx,sy,sz) float32 SDF as a collider. May be called "
             "every step to drive an animated collider; <0 = inside.")
        .def("set_external_force", &PyFlipSim::set_external_force,
             py::arg("field"),
             "Register a (sx,sy,sz,3) float32 acceleration field. Persists "
             "until cleared or replaced.")
        .def("clear_external_force", &PyFlipSim::clear_external_force,
             "Disable the external force field.")
        .def("add_particles", &PyFlipSim::add_particles,
             py::arg("pos"), py::arg("vel"), py::arg("phase"),
             "Append particles. pos/vel: (N,3) float32; phase: (N,) int32.")
        .def("initialize", &PyFlipSim::initialize,
             "Run scene setup (particle init, collider, refinement, channel "
             "binding). Required before step().")
        .def("step", &PyFlipSim::step,
             "Advance one timestep. Call after initialize().")
        .def_property_readonly("step_number", &PyFlipSim::step_number,
             "Current step counter (incremented by step()/run()).")
        .def("run", &PyFlipSim::run,
             py::arg("n_steps"),
             "Run the dam-break simulation for n_steps timesteps "
             "(initialize + n*step internally).")
        .def("get_density", &PyFlipSim::get_density,
             "Return cell mass grid as ndarray (sx, sy, sz) float32.")
        .def("get_velocity", &PyFlipSim::get_velocity,
             "Return MAC face velocity as ndarray (sx, sy, sz, 3) float32.")
        .def("get_particles", &PyFlipSim::get_particles,
             "Return {pos:(N,3), vel:(N,3), phase:(N,)} as numpy arrays.")
        .def_property_readonly("sx", &PyFlipSim::sx)
        .def_property_readonly("sy", &PyFlipSim::sy)
        .def_property_readonly("sz", &PyFlipSim::sz)
        .def_property_readonly("resolution", &PyFlipSim::resolution)
        .def_property_readonly("block_size", &PyFlipSim::blockSize);
}
