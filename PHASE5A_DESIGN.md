# Phase 5a: VDB / BGEO 出力 — 実装仕様書

## 対象ファイル
- `src/vdb_io.cpp` / `src/vdb_io.h` (新規)
- `src/flip_sim.cpp` / `src/flip_sim.h` (出力フック追加 + frustum 対応)
- `makefile` (OpenVDB 依存追加)
- `src/main.cpp` (CLI フラグ追加)

## 参照論文
- Braun, Bender, Thuerey: "Adaptive Phase-Field-FLIP" (SIGGRAPH 2025)
- Section 3.3 (Phase Field), 3.4 (Adaptive MG)
- Section 8.1 (bunny-of-bunnies — 表面再構成路線)

---

## 1. 現状の出力状況

| 出力 | 形式 | 用途 |
|---|---|---|
| `c:/tmp/flip_frame_NNNN.png` | PNG (RGB) | デバッグ可視化 (`saveParticleSlice`) |
| `msbg.log` | テキストログ | パフォーマンス計測 |

**Houdini に渡せる出力は無い。** Phase 5a でこれを構築する。

---

## 2. Phase 5a のゴール

**最終状態**:
- 各タイムステップで MSBG → VDB / BGEO ファイルを書き出し
- 軽量化のため、Step A→B→E→C の順で削減手法を実装
- Houdini 20.5 で File SOP / Volume Visualization で読める

**性能目標**:
- 128³ シーンで I/O オーバーヘッド < 全ステップ時間の 20%
- 1024³ シーンで Step A 適用後、1 フレーム VDB サイズ < 100MB

---

## 3. データ構造 / 設定の変更

### 3.1 `FlipConfig` 拡張 (flip_sim.h)

```cpp
struct FlipConfig {
    // ...既存フィールド...

    //=== Phase 5a: VDB / BGEO 出力 ============================================

    // 出力ディレクトリ。空なら出力無効
    std::string outputDir = "";

    // チャンネル選択 (Step B)
    bool outputDensity      = true;   // CH_FLOAT_1 → density VDB
    bool outputVelocity     = true;   // CH_VEC3_4 → vel VDB (face-staggered)
    bool outputPressure     = false;  // CH_PRESSURE → pressure VDB
    bool outputSurface      = true;   // narrow-band level set (Step A)
    bool outputParticles    = true;   // BGEO 粒子

    // 量子化 (Step B)
    bool useFloat16         = true;   // velocity を f16 で書く

    // Narrow-band 抽出 (Step A)
    float narrowBandPhi     = 0.5f;   // φ=0.5 を界面とみなす
    float narrowBandWidth   = 3.0f;   // ±3 voxel を出力範囲

    // キーフレーム (Step E)
    int   outputStride      = 1;      // N ステップ毎に書き出し (1=毎フレーム)
    bool  outputVelocityForBlend = true;  // 補間用に velocity を必ず併出

    // Frustum-aware refinement (Step C)
    bool  useFrustumRefinement = false;
    std::string cameraPath  = "";     // BGEO or JSON のカメラパス
    int   imageHeight       = 1080;   // pixel-size LOD の基準
    float frustumLodScale   = 1.0f;   // 1.0=画面1pxで1ボクセル
};
```

### 3.2 既存チャンネルとの対応

| FlipConfig フラグ | MSBG チャンネル | VDB grid | 用途 |
|---|---|---|---|
| `outputDensity` | `CH_FLOAT_1` (mass) | `FloatGrid` (FOG_VOLUME) | レンダリング |
| `outputVelocity` | `CH_VEC3_4` | `Vec3SGrid`/`Vec3HGrid` (STAGGERED) | モーションブラー / 補間 |
| `outputPressure` | `CH_PRESSURE` | `FloatGrid` | デバッグ |
| `outputSurface` | β/φ から生成 | `FloatGrid` (LEVEL_SET) | メッシュ化 |
| `outputParticles` | `state.particles` | BGEO | ホワイトウォーター etc. |

---

## 4. ファイル契約

### 4.1 ディレクトリ構成

```
out/
├── frame_0001/
│   ├── density_L0.vdb     # 最細解像度の密度
│   ├── density_L1.vdb     # 中解像度
│   ├── density_L2.vdb     # 粗解像度
│   ├── vel_L0.vdb         # 速度 (Vec3, staggered)
│   ├── vel_L1.vdb
│   ├── surface_L0.vdb     # narrow-band level set (L0 のみ)
│   ├── pressure_L0.vdb    # オプション
│   └── particles.bgeo.sc  # 全粒子 (単一解像度)
├── frame_0002/
└── ...
```

**根拠**: フレーム単位ディレクトリにすると、Houdini File SOP で `out/frame_$F4/density_L0.vdb` のような expression が使える。

### 4.2 命名規則

- `<channel>_L<level>.vdb`: チャンネル + 解像度レベル別
- `particles.bgeo.sc`: 圧縮 BGEO (`.sc` = lz4 圧縮)
- フレーム番号は 4 桁ゼロパディング (`$F4`)

### 4.3 グリッド属性

| Grid 名 | Grid Class | 値型 | 単位 |
|---|---|---|---|
| `density` | FOG_VOLUME | f32 (or f16) | 0–1 (φ) |
| `vel` | STAGGERED | Vec3f (or Vec3h) | voxel/sec (MAC) |
| `pressure` | UNKNOWN | f32 | Pa (相対) |
| `surface` | LEVEL_SET | f32 | voxel |

**注意**: VDB transform は `voxelSize = 1.0 * (1 << level)` でレベル毎に異なる。Houdini 側で Volume Merge する際はこの違いを吸収する必要あり。

---

## 5. モジュール構成

### 5.1 新規ファイル `src/vdb_io.h`

```cpp
#ifndef VDB_IO_H
#define VDB_IO_H

#include <string>
#include "msbg.h"
#include "flip_sim.h"

namespace VdbIO {

// メインエントリポイント。frame 番号と出力ディレクトリを受け取る。
// cfg.output* フラグで実際に書き出す内容を制御。
void writeFrame(const FlipState& state,
                const FlipGridBundle& grid,
                const FlipConfig& cfg,
                int frame);

// 個別書き出し関数
void writeDensityVDB(const FlipGridBundle& grid, const FlipConfig& cfg,
                     const std::string& path);
void writeVelocityVDB(const FlipGridBundle& grid, const FlipConfig& cfg,
                      const std::string& path);
void writeSurfaceVDB(const FlipGridBundle& grid, const FlipConfig& cfg,
                     const std::string& path);
void writePressureVDB(const FlipGridBundle& grid, const FlipConfig& cfg,
                      const std::string& path);
void writeParticlesBGEO(const FlipState& state, const std::string& path);

// Houdini カメラ入力読み込み (Step C)
struct CameraFrame {
    float eye[3], lookAt[3], up[3];
    float fovY;        // 縦 FOV (rad)
    float aspect;
    float nearZ, farZ;
};

bool readCameraPath(const std::string& path,
                    std::vector<CameraFrame>& frames);

} // namespace VdbIO

#endif
```

### 5.2 OpenVDB 依存 (makefile)

```makefile
LD_LIBS_FOR_MSBG_DEMO = \
    -lHYPRE -lmsmpi \
    -lopenvdb -lboost_iostreams-mt -lblosc -lzstd \  # ← 追加
    -lpng -ljpeg \
    -ltbbmalloc -ltbb12 -ltbbmalloc_proxy \
    -lz -lm
```

OpenVDB の依存:
- `boost_iostreams` (BGEO のために boost iostreams)
- `blosc`, `zstd` (VDB 圧縮)
- TBB は既存

MSYS2 で `pacman -S mingw-w64-x86_64-openvdb` で導入。

---

## 6. 関数ごとの詳細仕様

### 6.1 `writeFrame()` — メインエントリ

```cpp
void VdbIO::writeFrame(state, grid, cfg, frame)
{
    if(cfg.outputDir.empty()) return;
    if(frame % cfg.outputStride != 0) return;  // Step E: stride

    // フレームディレクトリ作成
    char dir[512];
    snprintf(dir, sizeof(dir), "%s/frame_%04d", cfg.outputDir.c_str(), frame);
    UtMkdir(dir);

    // 各チャンネル書き出し
    if(cfg.outputDensity)   writeDensityVDB(grid, cfg, dir + "/density");
    if(cfg.outputVelocity)  writeVelocityVDB(grid, cfg, dir + "/vel");
    if(cfg.outputPressure)  writePressureVDB(grid, cfg, dir + "/pressure");
    if(cfg.outputSurface)   writeSurfaceVDB(grid, cfg, dir + "/surface");
    if(cfg.outputParticles) writeParticlesBGEO(state, dir + "/particles.bgeo.sc");
}
```

### 6.2 `writeDensityVDB()` — 多解像度 float grid

```cpp
void writeDensityVDB(grid, cfg, basePath)
{
    auto *msbg = grid.msbg;
    const int nLevels = msbg->getNumLevels();

    for(int level = 0; level < nLevels; level++)
    {
        // この level に属する block のみ抽出
        auto *sgMass = grid.mass;  // L0 のみ持つ
        if(level > 0) sgMass = msbg->getFloatChannel(CH_FLOAT_1, level);

        // OpenVDB FloatGrid 作成
        auto vdbGrid = openvdb::FloatGrid::create(0.0f);
        vdbGrid->setGridClass(openvdb::GRID_FOG_VOLUME);

        // voxelSize はレベルに応じて 2^level
        const double voxelSize = 1.0 * (1 << level);
        vdbGrid->setTransform(
            openvdb::math::Transform::createLinearTransform(voxelSize));

        // MSBG block ごとに VDB に書き込み
        auto accessor = vdbGrid->getAccessor();
        for(int bid = 0; bid < sgMass->nBlocks(); bid++)
        {
            // この block がこの level に属するか
            if(msbg->getBlockLevel(bid) != level) continue;

            float *d = sgMass->getBlockDataPtr(bid);
            if(!d) continue;

            // block の voxel を VDB に転送
            const int bsx = sgMass->bsx();
            int bx, by, bz;
            sgMass->getBlockCoordsById(bid, bx, by, bz);
            for(int kz=0; kz<bsx; kz++)
            for(int ky=0; ky<bsx; ky++)
            for(int kx=0; kx<bsx; kx++)
            {
                const int vid = kx + ky*bsx + kz*bsx*bsx;
                const float val = d[vid];
                if(val < cfg.massEps) continue;  // sparse 化

                openvdb::Coord coord(bx*bsx+kx, by*bsx+ky, bz*bsx+kz);
                accessor.setValue(coord, val);
            }
        }

        // ファイル保存
        char path[512];
        snprintf(path, sizeof(path), "%s_L%d.vdb", basePath.c_str(), level);
        openvdb::io::File file(path);
        openvdb::GridPtrVec grids;
        grids.push_back(vdbGrid);
        file.write(grids);
        file.close();
    }
}
```

**ポイント**:
- 各 level で別 vdb ファイルを作る
- `massEps` 未満の値は書かない (自然な sparse 化)
- voxelSize はレベルに応じて変わる

### 6.3 `writeVelocityVDB()` — staggered Vec3 grid

```cpp
void writeVelocityVDB(grid, cfg, basePath)
{
    // FlipConfig::useFloat16 で型分岐
    if(cfg.useFloat16)
        writeVelocityVDBImpl<openvdb::Vec3HGrid>(grid, basePath);
    else
        writeVelocityVDBImpl<openvdb::Vec3SGrid>(grid, basePath);
}
```

`Vec3HGrid` (Vec3 of half-float, f16) で量子化。**Step B の主要効果**: 速度場のサイズが半分。

`setGridClass(openvdb::GRID_STAGGERED)` で MAC スタガード明示。Houdini 側は `Volume Visualization` で正しく解釈する。

### 6.4 `writeSurfaceVDB()` — Step A: narrow-band level set

```cpp
void writeSurfaceVDB(grid, cfg, basePath)
{
    auto *sgBeta = grid.beta;  // β = 1/ρ から φ を再構成

    auto sdf = openvdb::FloatGrid::create(cfg.narrowBandWidth);
    sdf->setGridClass(openvdb::GRID_LEVEL_SET);
    sdf->setTransform(openvdb::math::Transform::createLinearTransform(1.0));

    auto acc = sdf->getAccessor();
    const float halfBand = cfg.narrowBandWidth;

    // L0 only (interface は最細レベルにしか意味がない)
    for(int bid = 0; bid < sgBeta->nBlocks(); bid++)
    {
        // ...各 voxel で φ を計算...
        // signed distance = (φ - 0.5) * voxelSize ≈ narrow-band SDF

        if(fabsf(sd) < halfBand)  // narrow band のみ
            acc.setValue(coord, sd);
    }

    // OpenVDB 自身の narrow-band 機構で再構築
    openvdb::tools::pruneInactive(sdf->tree());

    // 保存
    openvdb::io::File file(basePath + "_L0.vdb");
    file.write({sdf});
    file.close();
}
```

**Step A の効果**: 流体内部の voxel を全部捨てる ≈ 体積→面積の削減。1024³ で典型 50–100倍の削減。

### 6.5 `writeParticlesBGEO()` — 粒子書き出し

OpenVDB の Houdini 連携には粒子書き出し API がない。代わりに **OpenVDB Points Grid** を使う or 自前 BGEO writer。

**設計判断**: 自前 BGEO writer を実装。Houdini BGEO 形式は仕様公開されており、ASCII 版なら 100行で書ける。バイナリ + 圧縮 (.sc = lz4) は別ライブラリ要。

→ 当面は **OpenVDB Points Grid** で `.vdb` として粒子も出す。Houdini 20.5 は VDB Points を直接読める。

```cpp
void writeParticlesBGEO(state, path)
{
    using namespace openvdb::points;

    std::vector<openvdb::Vec3f> positions, velocities;
    std::vector<int> phases;
    for(const auto& p : state.particles) {
        positions.emplace_back(p.pos.x, p.pos.y, p.pos.z);
        velocities.emplace_back(p.vel.x, p.vel.y, p.vel.z);
        phases.push_back(p.phase);
    }

    PointAttributeVector<openvdb::Vec3f> posWrap(positions);
    auto pointIndex = createPointIndexGrid<PointDataGrid>(posWrap);
    auto pointGrid  = createPointDataGrid<NullCodec, PointDataGrid>(...);

    appendAttribute<openvdb::Vec3f>(pointGrid->tree(), "vel");
    appendAttribute<int32_t>(pointGrid->tree(), "phase");
    // ...attribute 書き込み...

    openvdb::io::File file(path);
    file.write({pointGrid});
}
```

(実装的には ASCII BGEO writer のほうが軽い。トレードオフは review で詰める)

---

## 7. Step C: Frustum-aware refinement の詳細

### 7.1 `updateRefinementMap()` の拡張

現状 (flip_sim.cpp:480-548):
```
distance_from_liquid (BFS) → level
```

拡張後:
```
level = max(
    level_from_liquid,            // 既存: 液体追従
    level_from_frustum,           // 新規: 視錐台外なら粗く
    level_from_pixel_size         // 新規: 遠景の pixel-size 補正
)
```

**ルール**:
- 視錐台外: 即 coarsest (L2)
- 視錐台内 + 液体近傍: L0
- 視錐台内 + 遠景 + 1 voxel が screen で 1 px 以下: L1 / L2

### 7.2 視錐台 / 距離テスト

```cpp
bool inFrustum(const CameraFrame& cam, Vec3 worldPos)
{
    // 6-plane test: near, far, left, right, top, bottom
    // ...
}

int levelFromPixelSize(const CameraFrame& cam, Vec3 worldPos,
                        int blockSizeWorld, int imageHeight)
{
    Vec3 toEye = worldPos - cam.eye;
    float dist = length(toEye);
    float pixelSize = 2 * dist * tanf(cam.fovY/2) / imageHeight;
    if(blockSizeWorld * 2 < pixelSize) return 2;  // L2
    if(blockSizeWorld     < pixelSize) return 1;  // L1
    return 0;                                      // L0
}
```

### 7.3 カメラパス入力

JSON が手堅い (Houdini 側でも書きやすい):
```json
[
  {"frame": 1, "eye": [0,5,10], "lookAt": [0,0,0], "up": [0,1,0],
   "fovY": 0.785, "aspect": 1.778, "near": 0.1, "far": 1000.0},
  ...
]
```

`readCameraPath()` で読み込み、`runDamBreak()` ループで現在フレームのカメラを選ぶ。

---

## 8. 段階的実装順序

```
Step 5a-0 (基盤): ~3 日
  - makefile に OpenVDB 依存追加
  - vdb_io.h / vdb_io.cpp 新規作成
  - writeDensityVDB, writeVelocityVDB の最小版 (level 0 のみ)
  - flip_sim.cpp の runDamBreak() に writeFrame() フック挿入
  - main.cpp に -o <dir> CLI フラグ
  - 64³ で動作確認 (Houdini で読めるか)

Step A (Narrow-band): ~1 日
  - writeSurfaceVDB() 追加
  - cfg.outputSurface / cfg.narrowBandWidth
  - 64³ で SDF 出力 → Houdini Convert VDB SOP でメッシュ化確認

Step B (量子化 + ch 選択): ~1 日
  - cfg.useFloat16, Vec3HGrid 切替
  - cfg.outputDensity 等のフラグでスキップ
  - 1024³ でファイルサイズ比較

Step E (キーフレーム): ~0.5 日
  - cfg.outputStride
  - velocity を必ず併出するロジック
  - Houdini Time Blend で補間動作確認

Step C (Frustum-aware refinement): ~3-5 日
  - readCameraPath()
  - updateRefinementMap() 拡張
  - cfg.useFrustumRefinement
  - カメラを動かして refinement map が追従するか確認
  - シミュ計算量も削減されることを確認 (Press 段時間が減るか)
```

---

## 9. テスト計画

### 9.1 Step 5a-0 基盤
- 64³ ダムブレイク 5 ステップ実行
- 各 frame で density / vel VDB が生成
- Houdini 20.5 で File SOP → Volume Visualize で 1ステップ目を表示
- 値が物理的に妥当 (水領域 density=1, 空気領域 density=0)

### 9.2 Step A (Narrow-band)
- 同じ 64³ シーン
- surface_L0.vdb のサイズが density_L0.vdb の 1/10 以下
- Houdini Convert VDB SOP でメッシュ化、見た目が水面っぽい

### 9.3 Step B (量子化)
- vel_L0.vdb のサイズが f32 → f16 で半減
- Houdini で読み込んで visualization に違和感ないか目視

### 9.4 Step E (キーフレーム)
- outputStride=4 で 20 ステップ走らせる
- 5 フレーム分の VDB だけ出力される
- Houdini Time Blend で補間してアニメーション確認

### 9.5 Step C (Frustum-aware)
- 静止カメラ + 動く水のシーン
- カメラが見ていない領域の refinement level が L2 になる
- シミュ全体時間が削減される (理想は 30–50% 減)
- カメラ視野内では水の精度が変わらない

---

## 10. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| OpenVDB 11 の API が Houdini 同梱と微妙に違う | ファイル読めない | `.vdb` は forward compat 高い。読み書きは ABI 影響なし |
| boost / blosc 依存が膨らむ | リンクトラブル | MSYS2 パッケージで一括解決 |
| VDB ファイルサイズが想定より大きい | ディスク逼迫 | Step B (f16) と Step A (narrow-band) で対処 |
| Houdini で VDB の voxelSize が level 別だと混乱 | 解像度跨ぎで誤差 | Volume Resample / VDB Combine で揃える docs を書く |
| Step C で refinement map が振動 | シミュ不安定 | regularizeRefinementMap() の hysteresis を追加検討 |
| Step C のカメラ入力フォーマットが Houdini で書きにくい | 連携破綻 | JSON で固定、Houdini 側に書き出し HDA を提供 |
| 粒子の BGEO 化が VDB Points 経由だと Houdini 内で扱いづらい | 後段 SOP で困る | ASCII BGEO writer に切替検討 |

---

## 11. Phase 5b 以降との関係

Phase 5a 完了後の流れ:
- **Phase 5b**: Houdini → MSBG 入力 (collider VDB 読込、source emitter BGEO)
- **Phase 5c**: pybind11 化 (ファイル経由を in-memory に)
- **Phase 5d**: HDK プラグイン (要 MSVC ビルド環境構築)

Phase 5a は **ファイル経由連携の決定版**として、後続フェーズの入出力契約のリファレンスにもなる。
