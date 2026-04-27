# Phase 5b: Houdini → MSBG 入力（Collider VDB） — 実装仕様書

## スコープ

| 入力 | 形式 | Phase 5b で実装 |
|---|---|---|
| 固体 collider | VDB float SDF (`solid_sdf` grid) | ✅ 本フェーズ |
| Source emitter | BGEO 粒子 | ❌ Phase 5c 以降 |
| 外力 field | VDB Vec3f | ❌ Phase 5c 以降 |

**Phase 5b は collider のみ**に集中する。理由:
- 双方向連携の最初の一歩として、Houdini で固体オブジェクトを作って MSBG が「壁」として認識する一連の流れを完成させる
- emitter / 外力は技術的にはより独立で、後段に回しやすい
- 既存 MSBG コードに `MSBG::CELL_SOLID` 機構があり、最小改修で接続できる

---

## 対象ファイル

- `src/vdb_in.cpp` / `src/vdb_in.h` (新規)
- `src/flip_sim.cpp` / `src/flip_sim.h` (collider 連携)
- `src/main.cpp` (CLI `-C <collider.vdb>`)
- `makefile` (変更不要、既に OpenVDB リンク済)

---

## 1. 入力契約（Houdini → MSBG）

### 1.1 ファイル形式
- 単一 OpenVDB ファイル（`.vdb`、Houdini File SOP で書き出し可能）
- 1 grid 含む: float SDF、grid 名 `solid_sdf`（fallback: 任意名の最初の FloatGrid）
- `setGridClass(GRID_LEVEL_SET)` 推奨だが必須ではない（FOG_VOLUME でも `value <= 0` を solid 内部とみなす）
- voxel size はシミュ空間と同じ（1.0 = 1 ボクセル単位）または異なる場合は読込時に変換

### 1.2 SDF 規約
- `value < 0` → solid 内部
- `value == 0` → 表面
- `value > 0` → 外側（流体可動領域）
- narrow-band で十分（OpenVDB の標準 LEVEL_SET）

### 1.3 静的（時間不変）
- Phase 5b では **動かない collider のみ**
- 動く collider (`collider_NNNN.vdb` のシーケンス) は Phase 5c 以降で対応
- 静的なら 1 フレーム目でラスタライズして以降は使い回し

---

## 2. データ構造

### 2.1 `FlipGridBundle` 拡張
新規チャンネル `colliderSdf` を追加。MSBG の既存チャンネル `CH_FLOAT_5` 等を流用するか、`std::vector<float>` を bundle 内に dense で持つかのどちらか。

**選択**: `FlipGridBundle::colliderSdf` を **dense `std::vector<float>` で持つ**（サイズ = sx*sy*sz）。
- 理由: collider は静的なので毎ステップ走査せず、起動時 1 回だけ flat にラスタライズ
- メモリ: 64³ なら 1MB、128³ で 8MB、1024³ で 4GB（このサイズになったら sparse 化検討）

```cpp
struct FlipGridBundle {
    // 既存...
    std::vector<float> colliderSdf;  // sx*sy*sz, NaN if no collider
    bool hasCollider = false;
};
```

### 2.2 `FlipConfig` 拡張
```cpp
struct FlipConfig {
    // 既存...
    std::string colliderPath = "";       // VDB ファイルパス
    float       colliderEps  = 0.5f;     // SDF この値以下で solid 判定（buffer cell）
    float       colliderPushOut = 1.0f;  // 粒子を表面+1ボクセル外側に押し出す
};
```

---

## 3. モジュール構成

### 3.1 新規ファイル `src/vdb_in.h`
```cpp
namespace VdbIn {

// VDB ファイルから signed distance field を読み、
// シミュ空間 (sx*sy*sz) に同サイズ float 配列として返す。
// 値: < 0 = solid 内部, > 0 = 外側
// 戻り値: 成功で true。失敗時は sdfOut 不変、false。
bool readColliderSDF(const std::string& path,
                     int sx, int sy, int sz,
                     std::vector<float>& sdfOut);

} // namespace VdbIn
```

### 3.2 `vdb_in.cpp` 実装ロジック
```cpp
bool readColliderSDF(path, sx, sy, sz, sdfOut)
{
    openvdb::initialize();  // safe to call multiple times

    openvdb::io::File file(path);
    file.open();
    auto grids = file.getGrids();
    file.close();

    // Pick first FloatGrid (or 'solid_sdf' by name preferentially)
    openvdb::FloatGrid::Ptr src;
    for(auto& g : *grids) {
        if(g->getName() == "solid_sdf") { src = openvdb::gridPtrCast<openvdb::FloatGrid>(g); break; }
    }
    if(!src) {
        for(auto& g : *grids) {
            if(g->isType<openvdb::FloatGrid>()) { src = openvdb::gridPtrCast<openvdb::FloatGrid>(g); break; }
        }
    }
    if(!src) return false;

    // Background (= outside) で初期化
    sdfOut.assign(size_t(sx)*sy*sz, src->background());

    // Active voxel を走査して flat 配列に転送
    auto acc = src->getConstAccessor();
    for(auto it = src->cbeginValueOn(); it; ++it) {
        const openvdb::Coord c = it.getCoord();
        if(c.x()<0 || c.x()>=sx) continue;
        if(c.y()<0 || c.y()>=sy) continue;
        if(c.z()<0 || c.z()>=sz) continue;
        sdfOut[size_t(c.x()) + size_t(c.y())*sx + size_t(c.z())*sx*sy] = it.getValue();
    }
    return true;
}
```

**ポイント**:
- VDB の voxel space ↔ シミュ空間は同じ scale を仮定（後で transform 対応可）
- background 値で「外側」初期化 → narrow-band の外は自動的に solid 判定にならない
- 1024³ では `assign` で 4GB 確保 → 後続で sparse 化検討

---

## 4. flip_sim 側の連携

### 4.1 起動時に collider をラスタライズ

`runDamBreak()` の冒頭、`initializeParticles()` の後に追加:

```cpp
if(!cfg_.colliderPath.empty()) {
    bool ok = VdbIn::readColliderSDF(
        cfg_.colliderPath, sx, sy, sz, grid_.colliderSdf);
    if(ok) {
        grid_.hasCollider = true;
        TRCP(("Loaded collider SDF: %s (%dx%dx%d)\n",
              cfg_.colliderPath.c_str(), sx, sy, sz));
    } else {
        TRCERR(("Failed to load collider: %s\n", cfg_.colliderPath.c_str()));
    }
}
```

### 4.2 `particleToGrid()` 後に collider セル mass=0 化

P2G で粒子由来の mass が collider 内のセルに散布されることがある。これを除去:

```cpp
// 既存 particleToGrid() の最後に追加
if(grid_.hasCollider) {
    auto* sgMass = grid_.mass;
    const int sx = grid_.sx, sy = grid_.sy, sz = grid_.sz;
    tbb::parallel_for(0, sz, [&](int iz) {
        for(int iy=0; iy<sy; iy++) for(int ix=0; ix<sx; ix++) {
            const float sd = grid_.colliderSdf[ix + iy*sx + iz*sx*sy];
            if(sd < cfg_.colliderEps)
                setF(sgMass, ix, iy, iz, 0.f);  // collider セルは「非流体」扱い
        }
    });
}
```

これだけで pressure ソルバ (MIC0_PCG / HYPRE_AMG_PCG) は collider セルを自動的に Neumann BC 扱いする（既存の `isFluid` 判定が使われる）。

### 4.3 `advectParticles()` で collider 内の粒子を押し出す

粒子が collider に侵入したら SDF 法線方向に押し出す:

```cpp
// advectParticles() のメインループ末尾、boundary clamp の直前
if(grid_.hasCollider) {
    int ix = clamp((int)floorf(p.pos.x), 0, sx-1);
    int iy = clamp((int)floorf(p.pos.y), 0, sy-1);
    int iz = clamp((int)floorf(p.pos.z), 0, sz-1);
    const float sd = grid_.colliderSdf[ix + iy*sx + iz*sx*sy];
    if(sd < 0.f) {
        // 法線推定: 中心差分
        const float dx_ = sampleSdf(ix+1,iy,iz) - sampleSdf(ix-1,iy,iz);
        const float dy_ = sampleSdf(ix,iy+1,iz) - sampleSdf(ix,iy-1,iz);
        const float dz_ = sampleSdf(ix,iy,iz+1) - sampleSdf(ix,iy,iz-1);
        const float L = sqrtf(dx_*dx_ + dy_*dy_ + dz_*dz_);
        if(L > 1e-6f) {
            const float n[3] = { dx_/L, dy_/L, dz_/L };
            const float push = -sd + cfg_.colliderPushOut;
            p.pos.x += push * n[0];
            p.pos.y += push * n[1];
            p.pos.z += push * n[2];
            // 法線方向の速度成分をゼロ化（slip BC）
            const float vn = p.vel.x*n[0] + p.vel.y*n[1] + p.vel.z*n[2];
            if(vn < 0.f) {
                p.vel.x -= vn * n[0];
                p.vel.y -= vn * n[1];
                p.vel.z -= vn * n[2];
            }
        }
    }
}
```

`sampleSdf()` は範囲チェック付き lookup ヘルパー。

### 4.4 (任意) MSBG V-cycle の `flipCellTypeCB` 拡張

将来 `MSBG_VCYCLE_PCG` ソルバーに切り替えた時に効くよう、cell type コールバックも対応させておく:

```cpp
MSBG::CellFlags flipCellTypeCB(void *user, int x, int y, int z) {
    // ...既存の domain 境界判定...
    if(ctx->hasCollider && /* sdf<0 at (x,y,z) */) return MSBG::CELL_SOLID;
    // ...既存の fluid/void 判定...
}
```

ただし `FlipPressureCtx` に collider 参照を渡す必要があるので拡張要。デフォルト MIC0_PCG では使わないので **後回し**。

---

## 5. CLI

### main.cpp
```
-C<path>  Collider SDF VDB file (read once at startup)
```

getopt 文字列に `C:` を追加。`flip_dam_break(testCase==3)` 経由で `cfg.colliderPath` に流す。

---

## 6. 段階的実装順序

```
Step 1: vdb_in.cpp/h 新規作成、readColliderSDF() 実装
        → 単体テストはせず、次の統合で確認

Step 2: flip_sim.h で FlipGridBundle::colliderSdf, hasCollider 追加
        FlipConfig::colliderPath, colliderEps, colliderPushOut 追加

Step 3: flip_sim.cpp で:
        - runDamBreak() 冒頭で readColliderSDF() 呼び出し
        - particleToGrid() で collider セル mass=0 化
        - advectParticles() で粒子押し出し

Step 4: main.cpp に -C <path> CLI 追加

Step 5: makefile に vdb_in.$(OBJE) 追加

Step 6: テスト用 collider VDB 作成（球 SDF を Python or C++ で生成）
        ダムブレイク + 球 collider のシミュで水が球を避けることを確認
```

---

## 7. テスト計画

### 7.1 球 collider テスト（64³）
- ドメイン中央に半径 10 の球を SDF VDB として生成
  - 自前生成スクリプト or `vdb_view` の test gen
- ダムブレイク + 球 collider で実行
- 期待: 水柱が崩れて球に当たり、球を避けて広がる
- 失敗ケース: 水が球を貫通 → push out 機構の動作確認

### 7.2 退化ケース
- collider なし (`-C` 未指定) → 従来通りの動作（リグレッション無し）
- 空 VDB ファイル → readColliderSDF が false を返し、TRCERR 出力 → 通常動作続行

---

## 8. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| collider voxel size とシミュ space の不一致 | SDF が伸縮 | 当面 1.0 固定、ドキュメント明記。将来 transform 対応 |
| narrow-band 範囲外で SDF が +inf 等 | 押し出し失敗 | sdfOut を background で初期化。background は通常 +halfBand |
| collider が refinement map に影響しない | L1/L2 ブロックで collider が解像不足 | Phase 5c で `level = max(liquid, camera, collider)` 拡張 |
| 1024³ で 4GB の dense 配列 | OOM | sparse 化（`std::vector<int8_t> mask` + narrow-band SDF）に切替 |
| 粒子の push out で過剰な変位 | 不安定 | `colliderPushOut` で調整、上限クランプ |
| static のみ | アニメ collider 不可 | Phase 5c でフレームごと collider 切替 |

---

## 9. Phase 5c 以降への接続

Phase 5b 完了後:
- **Phase 5c**: source emitter BGEO 読込、外力 field VDB 読込、動く collider
- **Phase 5d**: pybind11 化（ファイル経由を in-memory に）
- **Phase 5e**: HDK プラグイン（MSVC ビルド環境）
