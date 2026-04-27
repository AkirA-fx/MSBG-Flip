# Phase 5c: pybind11 化 — 実装仕様書

## スコープ

| 機能 | Phase 5c |
|---|---|
| C++ `FlipSimulation` を Python から起動 | ✅ |
| シミュ結果を **numpy array で取得** (density, velocity, particles) | ✅ |
| 入力 (collider SDF) を **numpy 経由で in-memory 設定** | ✅ |
| Houdini Python SOP と直接連携 | ❌ Phase 5d (HDK + ABI 揃え) |
| Source emitter / 外力 field の Python API | ❌ Phase 5d 以降 |

**Python ターゲット**: MSYS2 mingw64 Python 3.14 (`/c/msys64/mingw64/bin/python3.exe`)。
Houdini 同梱 Python (3.11) との ABI 揃えは Phase 5d で扱う。

---

## 対象ファイル

- `src/python_bindings.cpp` (新規)
- `makefile` (Python module ビルド target 追加)
- `examples/test_flip.py` (新規、動作確認用)
- `pyproject.toml` (オプション、後で考える)

`flip_sim.cpp/h`, `vdb_io.cpp/h`, `vdb_in.cpp/h`, `camera_io.cpp/h` は **修正なし**。pybind11 は既存 C++ API をそのまま薄くラップする方針。

---

## 1. Python から見た API（決め）

```python
import msbg_flip
import numpy as np

# シミュレーター生成
sim = msbg_flip.FlipSimulation(resolution=64, block_size=16)

# 設定 (省略時はデフォルト)
sim.configure(
    rho_l=1000.0,
    rho_g=1.0,
    cfl=3.0,
    flip_alpha=0.95,
    solver="MIC0_PCG",
    use_float16=True,
)

# Collider を numpy ndarray で渡す (sx*sy*sz の float SDF)
sdf = np.full((64, 64, 64), 100.0, dtype=np.float32)  # 全部外側
# ... sdf に球を書き込む ...
sim.set_collider(sdf, eps=0.5, push_out=1.0)

# シミュ初期化 (ダムブレイク粒子配置)
sim.initialize()

# ステップ進行
for step in range(20):
    sim.step()
    if step % 4 == 0:
        density = sim.get_density()    # ndarray (sx, sy, sz) float32
        velocity = sim.get_velocity()  # ndarray (sx, sy, sz, 3) float32
        particles = sim.get_particles()  # dict {pos: (N,3), vel: (N,3), phase: (N,)}
        # 解析・可視化等
```

**ポイント**:
- 1 シミュレーター = 1 Python オブジェクト
- numpy 配列はゼロコピー or 1 回コピーで返す（np.copy=True で安全側）
- 既存の `FlipConfig` はそのまま、Python 側は `configure(**kwargs)` でフィールドを設定
- `step()` は 1 タイムステップ進行、`run(n)` でまとめて n ステップ

---

## 2. C++ クラス露出方針

### 2.1 既存 `FlipSimulation` クラスを直接 wrap
```cpp
PYBIND11_MODULE(msbg_flip, m) {
    py::class_<FlipSimulation>(m, "FlipSimulation")
        .def(py::init([](int resolution, int blockSize) {
            // MSBG::MultiresSparseGrid::create を内部で呼ぶ
            // FlipSimulation オブジェクトを返す
        }))
        .def("configure", ...)
        .def("set_collider", ...)
        .def("initialize", ...)
        .def("step", ...)
        .def("run", ...)
        .def("get_density", ...)
        .def("get_velocity", ...)
        .def("get_particles", ...);
}
```

### 2.2 ライフタイム管理
`FlipSimulation` は内部で `MSBG::MultiresSparseGrid*` を保持するが、現状の `FlipSimulation` ctor は外部から msbg を渡される設計。Python ラッパーでは **ヘルパークラス** を 1 段挟んで `MultiresSparseGrid` の lifetime を Python オブジェクトに紐付ける:

```cpp
class PyFlipSim {
    MSBG::MultiresSparseGrid* msbg_;
    std::unique_ptr<FlipSimulation> sim_;
    bool initialized_ = false;
public:
    PyFlipSim(int resolution, int blockSize);
    ~PyFlipSim();  // MultiresSparseGrid::destroy(msbg_) ここで呼ぶ
    // ... API ...
};
```

これで Python の GC で `__del__` 時にちゃんと msbg が解放される。

### 2.3 step / run の分割
既存の `runDamBreak(nSteps)` は「初期化 + n ステップ + ログ出力」をまとめて実行。Python から段階的に使うため:

- `initialize()` — 既存の `runDamBreak` 冒頭部分（粒子配置、refinement map、touchAllBlocks 等）を切り出し
- `step()` — メインループの 1 イテレーション分
- `run(n)` — `step()` を n 回回すだけのヘルパー

`runDamBreak` 自体はリファクタしないで、ヘルパー側で同等の処理を組み立てる方針が安全（リグレッション避け）。

---

## 3. numpy 連携

### 3.1 grid → numpy
**dense 配列で返す方式**（最初の実装としてはこれが一番楽）:

```cpp
py::array_t<float> get_density(PyFlipSim& self) {
    const int sx = self.sx(), sy = self.sy(), sz = self.sz();
    py::array_t<float> arr({sx, sy, sz});
    auto buf = arr.mutable_unchecked<3>();
    for(int iz=0; iz<sz; iz++)
    for(int iy=0; iy<sy; iy++)
    for(int ix=0; ix<sx; ix++) {
        buf(ix, iy, iz) = getF(self.grid().mass, ix, iy, iz);
    }
    return arr;
}
```

**注意点**:
- メモリレイアウト: numpy のデフォルトは C order (slow axis = 0)。MSBG の (ix, iy, iz) を (i, j, k) として 3次元 ndarray にする時、軸順を Houdini/Numpy 慣習に合わせる必要あり。**当面 `(sx, sy, sz)` シェイプで `[ix, iy, iz]` にアクセス**する素直な並びを採用
- velocity は (sx, sy, sz, 3) shape

### 3.2 numpy → grid (collider 入力)
```cpp
void set_collider(PyFlipSim& self, py::array_t<float> sdf, float eps, float push_out) {
    auto buf = sdf.unchecked<3>();
    if(buf.shape(0) != self.sx() || buf.shape(1) != self.sy() || buf.shape(2) != self.sz())
        throw py::value_error("collider shape mismatch");
    // FlipGridBundle::colliderSdf に転送
    auto& dst = self.grid().colliderSdf;
    dst.resize(self.sx() * self.sy() * self.sz());
    for(...) dst[...] = buf(ix, iy, iz);
    self.grid().hasCollider = true;
    self.config().colliderEps = eps;
    self.config().colliderPushOut = push_out;
}
```

### 3.3 particles
粒子は `state.particles` (vector<FlipParticle>) として既にメモリ上にある。numpy 経由で `(N, 3)` ndarray にコピー:

```cpp
py::dict get_particles(PyFlipSim& self) {
    const auto& parts = self.state().particles;
    const size_t N = parts.size();
    py::array_t<float> pos({N, size_t(3)});
    py::array_t<float> vel({N, size_t(3)});
    py::array_t<int>   phase({N});
    auto pbuf = pos.mutable_unchecked<2>();
    auto vbuf = vel.mutable_unchecked<2>();
    auto phbuf = phase.mutable_unchecked<1>();
    for(size_t i=0; i<N; i++) {
        pbuf(i,0) = parts[i].pos.x; pbuf(i,1) = parts[i].pos.y; pbuf(i,2) = parts[i].pos.z;
        vbuf(i,0) = parts[i].vel.x; vbuf(i,1) = parts[i].vel.y; vbuf(i,2) = parts[i].vel.z;
        phbuf(i)  = parts[i].phase;
    }
    py::dict d;
    d["pos"]   = pos;
    d["vel"]   = vel;
    d["phase"] = phase;
    return d;
}
```

---

## 4. ビルド統合

### 4.1 makefile target
```makefile
PYTHON_MODULE = msbg_flip$(PYTHON_EXT_SUFFIX)

OBJS_PYTHON = $(OBJS_MSBG_LIB) flip_sim.$(OBJE) vdb_io.$(OBJE) \
              camera_io.$(OBJE) vdb_in.$(OBJE) python_bindings.$(OBJE)

PYBIND_INC = $(shell python3 -m pybind11 --includes)
PYTHON_EXT_SUFFIX = $(shell python3-config --extension-suffix)

$(PYTHON_MODULE): $(OBJS_PYTHON)
    $(LD) -shared $(LDFLAGS) -o $@ $^ \
      $(LD_LIBS_FOR_MSBG_DEMO) \
      $(shell python3-config --ldflags --embed)

python_bindings.$(OBJE): python_bindings.cpp
    $(CC) -c $(CPP_FLAGS_ALL) $(PYBIND_INC) -fPIC -o $@ $<
```

ただし MSYS2 mingw64 では `python3-config` がない場合がある。代替で:
- `pybind11-config --includes` で include path
- `-lpython3.14` で link

実装時に試行錯誤して確定。

### 4.2 Python module ファイル名
MSYS2 Python 3.14 + mingw64 ABI なら拡張子は:
```
msbg_flip.cp314-mingw_x86_64_msvcrt_gnu.pyd
```
これは Python 自身が `import msbg_flip` で勝手に見つけるので、出力名は気にしなくて良い (Python が自動で `.pyd` 拡張子を扱う)。

---

## 5. 段階的実装順序

```
Step 1: pybind11 / python / numpy パッケージ確認 (ユーザー側でインストール)

Step 2: PyFlipSim ヘルパークラス + コンストラクタ + initialize() 実装
        Python から sim = msbg_flip.FlipSimulation(64, 16) を作成、
        sim.initialize() で粒子初期化までできる

Step 3: configure(), step(), run() 実装
        sim.run(20) で 20 ステップ走る

Step 4: get_density(), get_velocity() 実装
        numpy で grid を取得、shape/値域確認

Step 5: get_particles() 実装

Step 6: set_collider() 実装

Step 7: examples/test_flip.py で end-to-end 動作確認
        Phase 5b と同じ「ダムブレイク + 球 collider」を Python で再現
```

---

## 6. テスト計画

### 6.1 基本起動 (Step 2-3)
```python
sim = msbg_flip.FlipSimulation(64, 16)
sim.initialize()
sim.run(5)
print("OK")
```

### 6.2 grid 取得検証 (Step 4)
```python
density = sim.get_density()
assert density.shape == (64, 64, 64)
assert density.dtype == np.float32
print(f"density: min={density.min()} max={density.max()}")
# 期待: max > 0 (粒子配置済みなら mass > 0 のセルがある)
```

### 6.3 collider 入力 (Step 6)
```python
sx, sy, sz = 64, 64, 64
xs, ys, zs = np.meshgrid(np.arange(sx), np.arange(sy), np.arange(sz), indexing='ij')
dist = np.sqrt((xs-32)**2 + (ys-24)**2 + (zs-32)**2) - 10.0
sim.set_collider(dist.astype(np.float32))
sim.run(20)
particles = sim.get_particles()
# 期待: 球の領域に粒子が無い (中心部 (32,24,32) ±10 以内に位置する粒子が少ない)
```

---

## 7. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| `python3-config` が MSYS2 mingw64 にない | makefile が組めない | `pybind11-config` + 手書きの `-I/-L` で代用 |
| MSVC vs mingw64 ABI 不一致 | Python module が import 失敗 | MSYS2 Python と同じ mingw64 でビルド固定 |
| numpy がない / version 不一致 | import 時 NumPy 例外 | `pybind11/numpy.h` インクルードして numpy 抽象化に依存 |
| MSBG の lifetime 管理 | 二重解放 / メモリリーク | `PyFlipSim::~` で `MultiresSparseGrid::destroy()` を 1 度だけ呼ぶ |
| 大きな grid を毎フレーム numpy にコピー | 性能低下 | `py::array_t` でバッファ参照渡し or buffer_protocol。最初はコピー、後で最適化 |
| `runDamBreak` 内部のログ出力が冗長 | Python 利用時にうるさい | `LogLevel` 調整 or 出力抑制フラグ追加 |
| Houdini Python (3.11) と非互換 | Houdini で import できない | Phase 5d で対処 (HDK + Houdini Python ABI) |

---

## 8. Phase 5d への接続

Phase 5c 完了後、ABI 揃えと Houdini SDK 連携を Phase 5d で:
- MSVC v143 ビルド環境構築
- Houdini 同梱 Python (3.11) ヘッダ参照
- HDK の SOP/DOP プラグインから Python wrapping を呼べるか検証
- 最終的には HDK SOP 内部に MsbgFlip オブジェクトを直接保持する形（Python ブリッジは中間でしかない）
