# Phase 5c-ext: Step-by-step Python API + 動的 collider / 外力 / emitter

## 背景

Phase 5c で `sim.run(N)` を提供したが、N ステップを丸ごと走らせるだけで、
途中で collider を更新したり、粒子を追加したり、外力 field を変えたりする
用途には向かない。本フェーズで step-by-step 駆動の Python API を整備する。

**ファイル経由 (.vdb シーケンス) は実装しない**。Python から numpy で渡す
パターンのみ。理由:
- 毎ステップ VDB 読込は数百 ms のオーバーヘッド
- Houdini との連携も最終的に Python SOP / HDK 経由で in-memory に集約予定
- 現状でも `add_particles` を Python から呼べば emitter 動作は実現できる

---

## スコープ

| 機能 | このフェーズ |
|---|---|
| `initialize()` / `step()` 分割 | ✅ |
| `set_collider(sdf)` を毎ステップ呼び直せる | ✅ (既存 API を per-step 用に再活用) |
| `set_external_force(field)` 新規 | ✅ |
| `add_particles(pos, vel, phase)` 新規 | ✅ |
| Python API での state 取得 (`sim.step_number` など) | ✅ |
| ファイル経由の collider シーケンス | ❌ |
| ファイル経由の force/emitter シーケンス | ❌ |
| Source emitter の宣言的 API (region + rate) | ❌ (Python helper としてユーザーが書ける) |

---

## 1. 対象ファイル

- `src/flip_sim.h` (新メソッド宣言)
- `src/flip_sim.cpp` (initialize/step 分割、外力 field 加算)
- `src/python_bindings.cpp` (Python API 拡張)
- `examples/test_flip_dynamic.py` (新規動作確認スクリプト)

---

## 2. flip_sim 側の変更

### 2.1 `FlipSimulation` の API 拡張

```cpp
class FlipSimulation {
public:
    // 既存: runDamBreak(int nSteps), runStandaloneDamBreak(...)

    // 新規: step-by-step 駆動
    bool initialize();   // 既存 runDamBreak の 1〜5 (前準備) 部分
    bool stepOnce();     // メインループの 1 イテレーション

    // 新規: Python から粒子を追加
    void addParticles(const std::vector<Vec3Float>& pos,
                      const std::vector<Vec3Float>& vel,
                      const std::vector<int>& phase);

    // 既存
    int  runDamBreak(int nSteps);  // wrapper to initialize() + stepOnce() x N

    // ...
};
```

### 2.2 `runDamBreak` の分解方針

既存の処理をそのまま 2 メソッドに分けるだけで、ロジック変更なし。
リグレッションリスク最小化のため:

```cpp
bool FlipSimulation::initialize()
{
    // setDomainBoundarySpec_ + initializeParticles() + collider load
    //  + camera load + updateRefinementMap + setRefinementMap
    //  + prepareChannels + rebind + touchAllBlocks
    // (既存 runDamBreak の上部をそのままここへ移動)
    // 失敗時 false を返す
}

bool FlipSimulation::stepOnce()
{
    // 既存ループの 1 イテレーション
    // updateRefinementMap (変化したら再 prepare/rebind/touch)
    //   + computeDt + particleToGrid + applyGravity + pressureProjection
    //   + gridToParticle + advectParticles
    //   + state_.step++, state_.time += dt
    //   + log + saveParticleSlice + VdbIO::writeFrame
    // 失敗時 false (rebind null など)
}

int FlipSimulation::runDamBreak(int nSteps)
{
    // ヘッダログ
    if(!initialize()) return 1;
    for(int i=0; i<nSteps; i++) {
        if(!stepOnce()) return 1;
    }
    // フッタログ
    return 0;
}
```

`stepOnce` のフレーム表示は `state_.step` ベースに変更
(現状 `step+1/nSteps` の `nSteps` は totalsteps で文脈情報なので、
分割後は単に `frame=state_.step` に統一)。

### 2.3 外力 field 加算

`FlipGridBundle` に dense Vec3 配列を追加:

```cpp
struct FlipGridBundle {
    // ...既存...

    // Phase 5c-ext: external force field (per-cell, applied each step)
    // Layout: ix + iy*sx + iz*sx*sy, cleared after each step (one-shot
    // per-frame contribution; caller must re-set every step if needed).
    std::vector<float> externalForce;  // size = sx*sy*sz*3, or empty
    bool hasExternalForce = false;
};
```

`applyGravity()` の最後に追加:

```cpp
void FlipSimulation::applyGravity(float dt)
{
    // ...既存の重力処理...

    if(grid_.hasExternalForce) {
        const std::vector<float>& f = grid_.externalForce;
        // f を MAC 速度に dt 倍して加算 (cell-centered として近似的に)
        // u 面: 0.5*(f[ix-1] + f[ix]) を使うと厳密だが、まず単純に
        // cell-centered とみなして u/v/w 面に同じ値を加算する近似でよい
        tbb::parallel_for(0, sz, [&](int iz) {
            for(int iy=0; iy<sy; iy++) for(int ix=0; ix<sx; ix++) {
                if(!isFluid(sgMass, ix, iy, iz, MASS_EPS)) continue;
                size_t idx = (ix + iy*size_t(sx) + iz*size_t(sx)*size_t(sy)) * 3;
                Vec3Float v = readVel(sgVel, ix, iy, iz);
                v.x += dt * f[idx + 0];
                v.y += dt * f[idx + 1];
                v.z += dt * f[idx + 2];
                writeVel(sgVel, ix, iy, iz, v);
            }
        });
    }
}
```

### 2.4 `addParticles()` 実装

```cpp
void FlipSimulation::addParticles(
    const std::vector<Vec3Float>& pos,
    const std::vector<Vec3Float>& vel,
    const std::vector<int>& phase)
{
    if(pos.size() != vel.size() || pos.size() != phase.size())
        throw std::invalid_argument("addParticles: array sizes mismatch");
    state_.particles.reserve(state_.particles.size() + pos.size());
    for(size_t i=0; i<pos.size(); i++) {
        FlipParticle p;
        p.pos = pos[i];
        p.vel = vel[i];
        p.phase = phase[i];
        state_.particles.push_back(p);
    }
}
```

ドメイン外の粒子は弾く方が安全だが、`advectParticles()` の境界クランプで
最終的に押し戻されるので最初は許容する。

---

## 3. Python API

### 3.1 新規メソッド

```python
sim.initialize()                 # 初期化、必須 (sim.run() の代わりにこれ + step ループ)
sim.step()                       # 1 ステップ進行
sim.add_particles(pos, vel, phase)  # ndarray 受け取り
sim.set_external_force(field)    # ndarray (sx, sy, sz, 3) 受け取り
sim.clear_external_force()       # 外力フィールドを無効化
sim.set_collider(sdf, eps=0.5, push_out=1.0)  # 既存 (毎フレーム呼び直し可)
```

### 3.2 既存メソッドとの関係

| 呼び出し方 | 用途 |
|---|---|
| `sim.run(N)` | 全部おまかせ。ベンチ・テスト用 |
| `sim.initialize(); for..: sim.step()` | step ごとに何かする (collider 更新等) |
| 両者を混ぜる | 不可 — `run()` は内部で initialize する |

実装ガード: `run()` は「未初期化なら initialize して N ステップ」、
「既に初期化済みなら そのまま N ステップ」とする。

### 3.3 add_particles の API

```python
import numpy as np

# 1万粒子を追加
n = 10000
pos = np.random.rand(n, 3).astype(np.float32) * 64.0
vel = np.zeros((n, 3), dtype=np.float32)
phase = np.zeros((n,), dtype=np.int32)  # 0=liquid

sim.add_particles(pos, vel, phase)
```

shape チェック:
- `pos.shape == (N, 3)`, dtype float32
- `vel.shape == (N, 3)`, dtype float32
- `phase.shape == (N,)`, dtype int32
- `N >= 0`

### 3.4 set_external_force の API

```python
# 風 (一定方向に流す外力)
field = np.zeros((sim.sx, sim.sy, sim.sz, 3), dtype=np.float32)
field[:, :, :, 0] = 5.0  # +x 方向に 5 m/s² 加速
sim.set_external_force(field)
```

shape: `(sx, sy, sz, 3)` float32。

**重要**: 外力は「毎ステップ加算される」が、`set_external_force` を呼んだ
時点の field がメモリにコピーされ、`clear_external_force()` を呼ぶか別の
field で上書きするまで効き続ける。

---

## 4. 段階的実装順序

```
Step 1: flip_sim.h/cpp に initialize() / stepOnce() / addParticles() を追加。
        runDamBreak はそれらを使う wrapper に変更。
        既存 msbg_demo -c3 で動作確認 (リグレッション無し)。

Step 2: FlipGridBundle::externalForce / hasExternalForce 追加。
        applyGravity 末尾で加算ロジック実装。
        msbg_demo は外力をデフォルト無効にするので影響なし。

Step 3: python_bindings.cpp に initialize / step / add_particles /
        set_external_force / clear_external_force を追加。
        既存 .pyd を再ビルド。

Step 4: examples/test_flip_dynamic.py で:
        - 動く球 collider (毎ステップ中心位置を変える)
        - +x 方向の風 field
        - 100 ステップごとに粒子追加 (シーン上部から落ちる粒子)
        を組み合わせて 20 ステップ走らせ、結果を get_particles で確認。
```

---

## 5. テスト計画

### 5.1 リグレッション (Step 1 後)
```
./msbg_demo.exe -c3 -r64 -b16 -o out -C sphere_collider.vdb
```
が Phase 5b と同じ結果（粒子数、ステップ時間、collider 効果）。

### 5.2 step-by-step (Step 3 後)
```python
sim.initialize()
for i in range(5):
    sim.step()
density = sim.get_density()  # Phase 5c の sim.run(5) と同じ結果
```

### 5.3 dynamic collider (Step 4)
```python
sim.initialize()
for i in range(20):
    cx = 32 + 0.5 * i  # 球の中心が +x に動く
    sdf = make_sphere_sdf(64, 64, 64, cx, 24, 32, 10.0)
    sim.set_collider(sdf)
    sim.step()
particles = sim.get_particles()
# 球が通過した跡に水が押し流されているか目視確認
```

### 5.4 外力 (Step 4)
```python
field = np.zeros((sim.sx, sim.sy, sim.sz, 3), dtype=np.float32)
field[:, :, :, 0] = 5.0
sim.set_external_force(field)
sim.run(20)
# get_velocity の x 成分平均が +方向に偏るか確認
```

### 5.5 add_particles (Step 4)
```python
sim.initialize()
for i in range(20):
    if i % 5 == 0:
        # シーン上部に新粒子を追加
        pos = np.array([[16, 60, 32], [17, 60, 32], ...], dtype=np.float32)
        vel = np.zeros_like(pos)
        phase = np.zeros((len(pos),), dtype=np.int32)
        sim.add_particles(pos, vel, phase)
    sim.step()
# total particle 数が増えているか
```

---

## 6. リスクと対策

| リスク | 影響 | 対策 |
|---|---|---|
| `initialize()` 分割でリグレッション | msbg_demo がコケる | 既存 runDamBreak を thin wrapper に変えるだけ、ロジック変更なし |
| `addParticles` で粒子配列が再 alloc されて P2G に影響 | NaN | `reserve()` で事前確保、最初の `step()` で次の P2G に反映 |
| 外力 field の MAC face vs cell-centered | やや誤差 | 最初は cell-centered 扱いで近似。精度要なら face-averaging に拡張 |
| step-by-step だと debug slice / VDB output のフレーム番号が `state_.step` ベースに統一される | 既存挙動と微妙に違う？ | runDamBreak ベースは前から `step+1` で出していたので影響なし |
| Python から `set_external_force(None)` で消したい | API がない | `clear_external_force()` を提供 |

---

## 7. Phase 5d への接続

このフェーズ完了後、Phase 5d では:
- HDK プラグインから同じ initialize/step/set_collider API を呼ぶ
- numpy ↔ Houdini geometry の変換層を追加
- step-by-step 駆動が前提なので、本フェーズの成果がそのまま土台になる
