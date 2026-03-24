# Phase 3: Phase-Field 導入 — 実装仕様書

## 対象ファイル
- `src/flip_sim.cpp` （主要変更対象、現在 ~700行）

## 参照論文
- Braun, Bender, Thuerey: "Adaptive Phase-Field-FLIP" (SIGGRAPH 2025)
- Section 3.2 (P2G), 3.3 (Phase Field), 3.5 (Two-Phase Pressure)

---

## 1. 現在の実装状態

### 構造
```
FlipParticle { Vec3Float pos, vel; }       ← 単相（水のみ）
```

### チャンネル使用状況（OPT_SINGLE_LEVEL）
| チャンネル       | 用途              | 型         |
|-----------------|-------------------|-----------|
| CH_VEC3_4       | 現在速度 (u/v/w)   | Vec3Float |
| CH_VEC3_2       | 投影前速度         | Vec3Float |
| CH_FLOAT_1      | cell mass          | float     |
| CH_FLOAT_2      | PCG残差 r          | float     |
| CH_FLOAT_3      | PCG z = M^{-1}r    | float     |
| CH_FLOAT_4      | PCG探索方向 d       | float     |
| CH_FLOAT_6      | PCG q = A*d        | float     |
| CH_DIVERGENCE   | 発散               | float     |
| CH_PRESSURE     | 圧力               | float     |

### 利用可能な未使用チャンネル
| チャンネル          | 型         | 備考                                   |
|--------------------|-----------|---------------------------------------|
| CH_VEC3_3          | Vec3Float | 常に割り当て済み                         |
| CH_VEC3_3    | Vec3Float | 常に割り当て済み（値=25、論文の意図通り！） |
| CH_VELOCITY_AVG    | Vec3Float | !OPT_SINGLE_CHANNEL なら割り当て済み     |

### 主要関数
```
particleToGrid()      — P2G: 三線形補間、wU/wV/wW重みベクトル
applyGravity()        — v面速度に重力加算 + 固体BC
pressureProjection()  — MAC発散 → solvePressurePCG → 圧力勾配補正
  solvePressurePCG()  — MIC(0)前処理付きPCG
  buildMIC0Precon()   — precon[i] = 1/sqrt(A_diag - Σprecon²)
  applyMIC0Precon()   — 前進代入(L*q=r) → 後退代入(L^T*z=q)
  applyPressureMatrix() — q = A*x（均一密度ラプラシアン）
  pressureDiag()       — 対角成分 = in-domain隣接数
gridToParticle()      — MAC補間でPIC/FLIPブレンド
advectParticles()     — 位置更新 + 境界クランプ
```

---

## 2. Phase 3 のゴール

**最終状態**: 水＋空気の2相粒子を持ち、P2Gの蓄積質量からphase-fieldを計算し、
可変密度ポアソン方程式を解いて非圧縮性を維持する。

---

## 3. データ構造の変更

### FlipParticle
```cpp
// 変更前
struct FlipParticle { Vec3Float pos, vel; };

// 変更後
struct FlipParticle { Vec3Float pos, vel; int phase; };
// phase: 0 = liquid (水),  1 = air (空気)
// 粒子質量は phase から決定: m_p = (phase==0) ? RHO_L : RHO_G
```
論文では `m_p = rho_phase * V_p` で `V_p = 1`（均一解像度時）。

### 新規定数
```cpp
static const float RHO_L       = 1000.0f;   // 水の密度
static const float RHO_G       = 1.0f;      // 空気の密度
static const float ALPHA_PHI   = 1.0f;      // φ の鋭さパラメータ
```

---

## 4. チャンネル割り当て（変更後）

```
CH_VEC3_4        現在速度 (u/v/w)                     ← 変更なし
CH_VEC3_2        投影前速度                            ← 変更なし
CH_VEC3_3  フェース密度 β = 1/ρ (u面/v面/w面)    ← 新規利用
CH_FLOAT_1       cell mass（流体判定用、全相合算）       ← 変更なし
CH_FLOAT_2       PCG残差 r                             ← 変更なし
CH_FLOAT_3       PCG z = M^{-1}r                       ← 変更なし
CH_FLOAT_4       PCG探索方向 d                          ← 変更なし
CH_FLOAT_6       PCG q = A*d                           ← 変更なし
CH_DIVERGENCE    発散                                   ← 変更なし
CH_PRESSURE      圧力                                   ← 変更なし
```

**設計判断**: φ自体はチャンネルに保持しない。P2Gの蓄積質量からβ = 1/ρを計算し、
CH_VEC3_3 に直接格納する。φは中間値として一時的にのみ使用。

---

## 5. 関数ごとの変更仕様

### 5.1 initParticles() — 2相パーティクル初期化

```
変更点:
- 水ブロック（現在の dam 領域）: phase = 0, 密度 RHO_L
- 残り（dam 外）の領域を空気粒子で埋める: phase = 1, 密度 RHO_G
- 空気粒子の PPV は水と同じ（8 particles/voxel）でよい
- ドメイン全体を埋める（水 + 空気）

注意:
- 粒子数が大幅に増加する（64³の場合: 水 52万 + 空気 ~157万 = ~209万）
- 128³では 419万 + ~1258万 = ~1677万（メモリ注意）
- → 最初は 64³ でテストすること
```

**擬似コード**:
```cpp
for iz in [0,sz): for iy in [0,sy): for ix in [0,sx):
    int phase = (ix < damX && iy < damY) ? 0 : 1;  // 0=liquid, 1=air
    // nSub=2 のサブグリッドサンプリング（現行通り）
    for kz,ky,kx in [0,nSub):
        p.pos = (ix+(kx+0.5)/nSub, iy+(ky+0.5)/nSub, iz+(kz+0.5)/nSub)
        p.vel = (0, 0, 0)
        p.phase = phase
        gParticles.push_back(p)
```

### 5.2 particleToGrid() — P2G + φ計算 + β格納

**変更の概要**:
1. P2G の質量蓄積を**相ごとに分けて**行う（液体質量と全質量を別々にカウント）
2. 蓄積結果からフェースφを計算
3. φからβ = 1/ρを計算し CH_VEC3_3 に格納

**詳細設計**:

```
(a) MAC速度の散布: 現行通り wU/wV/wW で重み付き蓄積
    ただし、散布する運動量は m_p * u_p ではなく u_p のまま（現行通り）
    → 正規化で wU/wV/wW で割るので結果は同じ

(b) セル質量(sgMass): 全粒子（水+空気）の個数カウント（現行通り）
    → isFluid() の判定基準は「セルに粒子がいるか」のまま

(c) フェース生密度の蓄積（新規）:
    - U面: rhoTildeU[bid*nvox+vid] += m_p * w  (m_p = RHO_L or RHO_G)
    - V面: rhoTildeV[bid*nvox+vid] += m_p * w
    - W面: rhoTildeW[bid*nvox+vid] += m_p * w
    → std::vector<float> で一時保持（wU/wV/wW と同じパターン）

(d) φ計算（フェースごと）:
    rhoTilde0 = PPV * RHO_L  // 完全充填時の参照密度
    etaPhi    = logf(RHO_L / RHO_G)
    rhoTildeMin = etaPhi * RHO_G * rhoTilde0

    phi(rhoTilde) =
        0                                                    if rhoTilde < rhoTildeMin
        min(sqrtf(max(rhoTilde - rhoTildeMin, 0) / (ALPHA_PHI * rhoTilde0 * RHO_L)), 1)

(e) β計算とセル分類:
    rho  = RHO_G + phi * (RHO_L - RHO_G)
    beta = 1.0f / rho
    → sgBeta（CH_VEC3_3）の x/y/z コンポーネントに格納

(f) 界面から離れたフェースのクランプ（論文 Section 3.3）:
    - フェースの両側セルが共に liquid → phi = 1, beta = 1/RHO_L
    - フェースの両側セルが共に air    → phi = 0, beta = 1/RHO_G
    → 界面ノイズの除去
```

**関数シグネチャ変更**:
```cpp
static void particleToGrid(
    SBG::SparseGrid<Vec3Float> *sgVel,
    SBG::SparseGrid<float>     *sgMass,
    SBG::SparseGrid<Vec3Float> *sgBeta )   // ← 追加: フェース β
```

### 5.3 applyGravity() — 変更なし（最小変更版）

重力は全流体フェースに均一に加える（密度に依存しない形式）。
論文では外力項は `f/ρ` だが、重力 `g` は質量に比例するので `ρg/ρ = g` となり
密度非依存。したがって現行コードはそのまま使える。

### 5.4 pressureProjection() — 可変密度ポアソン方程式

**最も重要な変更箇所。**

#### 5.4.1 発散計算
変更なし。`div = (u_{i+1}-u_i + v_{j+1}-v_j + w_{k+1}-w_k) / dt`

#### 5.4.2 圧力行列 A の変更

**変更前（均一密度）**:
```
A * p[i,j,k] = diag * p[i,j,k] - Σ_{adj fluid} p[adj]
diag = in-domain 隣接数
```

**変更後（可変密度）**:
```
A * p[i,j,k] = Σ_{in-domain adj} beta_face * (p[i,j,k] - p[adj])

具体的に:
  = beta_{i+1/2}*(p[i]-p[i+1]) + beta_{i-1/2}*(p[i]-p[i-1])
  + beta_{j+1/2}*(p[i]-p[j+1]) + beta_{j-1/2}*(p[i]-p[j-1])
  + beta_{k+1/2}*(p[i]-p[k+1]) + beta_{k-1/2}*(p[i]-p[k-1])

where beta_{i+1/2} = getVC(sgBeta, ix+1, iy, iz, 0)  // U面のβ
      beta_{j+1/2} = getVC(sgBeta, ix, iy+1, iz, 1)  // V面のβ
      etc.
```

**対角成分**:
```
diag[i,j,k] = Σ_{in-domain adj faces} beta_face
```

#### 5.4.3 applyPressureMatrix() の変更

```cpp
// 変更前
const int diag = pressureDiag(sgMass, ix,iy,iz, nx,ny,nz);
float sum = 0.f;
if(isFluid(sgMass,ix-1,iy,iz)) sum += getF(sgX,ix-1,iy,iz);
...
setF(sgQ, ix,iy,iz, (float)diag * getF(sgX,ix,iy,iz) - sum);

// 変更後
float diagVal = 0.f, offSum = 0.f;
// x-方向
float bxp = getBetaFace(sgBeta, ix+1,iy,iz, 0);  // β at u-face (i+1,j,k)
float bxm = getBetaFace(sgBeta, ix,  iy,iz, 0);  // β at u-face (i,j,k)
if(ix+1 < nx) { diagVal += bxp; offSum += bxp * getF(sgX, ix+1,iy,iz); }
if(ix   > 0 ) { diagVal += bxm; offSum += bxm * getF(sgX, ix-1,iy,iz); }
// y, z 同様...
setF(sgQ, ix,iy,iz, diagVal * getF(sgX, ix,iy,iz) - offSum);
```

**注意**: `isFluid()` による条件分岐は不要になる。代わりにβ = 0のフェースが
自然にNeumann BCを表現する。ただし非流体セル（mass=0）のpは引き続き0に固定。

#### 5.4.4 pressureDiag() の変更

```cpp
// 変更前: int を返す（隣接数）
static int pressureDiag(sgMass, ix,iy,iz, nx,ny,nz)

// 変更後: float を返す（β の合計）
static float pressureDiagBeta(sgBeta, ix,iy,iz, nx,ny,nz)
{
    float d = 0.f;
    if(ix+1 < nx) d += getBetaFace(sgBeta, ix+1,iy,iz, 0);
    if(ix   > 0 ) d += getBetaFace(sgBeta, ix,  iy,iz, 0);
    if(iy+1 < ny) d += getBetaFace(sgBeta, ix,iy+1,iz, 1);
    if(iy   > 0 ) d += getBetaFace(sgBeta, ix,iy,  iz, 1);
    if(iz+1 < nz) d += getBetaFace(sgBeta, ix,iy,iz+1, 2);
    if(iz   > 0 ) d += getBetaFace(sgBeta, ix,iy,iz,   2);
    return d;
}
```

#### 5.4.5 MIC(0) 前処理の変更

buildMIC0Precon の変更点:
- `(float)Adiag` → `pressureDiagBeta(sgBeta, ...)` に置換
- 下方隣接のβ値を保持して `(A_off * precon)^2` の A_off を β に変更

```
// 変更前: A_off = -1 (全隣接で一定)
e -= precon(neighbor)^2

// 変更後: A_off = -beta_face
e -= (beta_face * precon(neighbor))^2
```

applyMIC0Precon の変更点:
- 前進/後退代入で `precon(k)` → `beta_face * precon(k)` に変更

```
// 前進代入（変更前）
t += precon(ix-1,iy,iz) * q(ix-1,iy,iz)

// 前進代入（変更後）
t += beta_xm * precon(ix-1,iy,iz) * q(ix-1,iy,iz)
// where beta_xm = getBetaFace(sgBeta, ix,iy,iz, 0)  （i面のβ）
```

#### 5.4.6 圧力勾配補正の変更

```cpp
// 変更前（均一密度）
setVC(sgVel, ix,iy,iz, 0,
      getVC(sgVel,ix,iy,iz,0) - dt*(pR-pL));

// 変更後（可変密度）
float beta = getVC(sgBeta, ix,iy,iz, 0);  // この面のβ
setVC(sgVel, ix,iy,iz, 0,
      getVC(sgVel,ix,iy,iz,0) - dt * beta * (pR-pL));
```

### 5.5 gridToParticle() — 変更なし（Phase 3 では）

FLIP_ALPHA は水・空気で共通のまま。
Phase 4 以降で相ごとに α を変える余地あり。

### 5.6 advectParticles() — 変更なし

### 5.7 saveParticleSlice() — 相の可視化

```
変更: 水粒子を青、空気粒子を灰色で描画
if(p.phase == 0) col = 青系; else col = 灰系;
```

---

## 6. ヘルパー関数（新規追加）

```cpp
// β を Vec3 チャンネル（sgBeta = CH_VEC3_3）から取得
// comp: 0=U面, 1=V面, 2=W面
static float getBetaFace(SBG::SparseGrid<Vec3Float> *sgBeta,
                          int ix, int iy, int iz, int comp)
{
    return getVC(sgBeta, ix, iy, iz, comp);
    // 範囲外は getVC → 0 → solid wall BC (Neumann) と等価
}
```

---

## 7. エントリポイントの変更

```cpp
int flip_dam_break(int resolution, int blockSize, int nSteps)
{
    // ...（既存のチャンネル準備に追加）
    msbg->prepareDataAccess(CH_VEC3_3);

    SBG::SparseGrid<Vec3Float> *sgBeta =
        msbg->getVecChannel(CH_VEC3_3, 0);

    // ... null チェックに sgBeta を追加

    // ブロック確保に sgBeta を追加
    sgBeta->getBlockDataPtr(bid, 1, 1);

    // メインループ
    particleToGrid(sgVel, sgMass, sgBeta);        // ← sgBeta 追加
    applyGravity(sgVel, sgVelOld, sgMass, DT);    // 変更なし
    pressureProjection(sgVel, sgMass, sgDiv, sgP,
                       sgR, sgZ, sgD, sgQ,
                       sgBeta, DT);                // ← sgBeta 追加
}
```

---

## 8. 段階的実装順序

```
Step 1: FlipParticle に phase を追加、initParticles で2相初期化
        → ビルド確認（既存動作に影響なし）

Step 2: particleToGrid に rhoTilde 蓄積と φ→β 計算を追加
        sgBeta に書き込み
        → 64³ でβ値のデバッグ出力（水領域β≈0.001、空気領域β≈1.0 を確認）

Step 3: pressureDiagBeta() を実装、applyPressureMatrix() を可変密度版に変更
        → PCG が動作するか確認（収束はまだ遅くてもOK）

Step 4: buildMIC0Precon() と applyMIC0Precon() をβ対応に更新
        → 収束を確認

Step 5: 圧力勾配補正をβ対応に変更
        → ダムブレイクの水面挙動が物理的か目視確認

Step 6: saveParticleSlice() で2相可視化
```

---

## 9. テスト計画

### 回帰テスト（Phase 2 互換性）
- 空気粒子を入れずに(RHO_G=RHO_L=1にして)単相実行し、Phase 2の結果と一致確認

### 基本動作確認（64³）
- dam break: 水が崩れ、空気が上に行くか
- β値の妥当性: 水内部≈0.001, 空気内部≈1.0, 界面で中間値
- PCG 収束: MIC(0) で ~50-100 iter 以内（密度比1000:1は悪条件）
- 粒子総数の保存: 各ステップで不変

### 数値安定性
- 密度比 1000:1 で PCG が発散しないか
- 空気の圧力が異常な値にならないか
- → 最悪の場合 PCG_MAXITER を増やすか TOL を緩める必要あり

---

## 10. リスクと対策

| リスク | 影響 | 対策 |
|-------|------|------|
| 密度比1000:1でPCGが収束しない | 圧力が正しく解けない | PCG_MAXITERを増やす or TOLを緩める |
| メモリ不足（128³で空気粒子追加） | 実行不可 | 64³でテスト、空気PPVを減らす |
| 空気粒子が境界からリーク | 非物理的挙動 | 境界クランプ強化 |
| β=0のフェースでゼロ除算 | NaN | β下限ガードを入れる: max(β, ε) |

---

## 11. 現在のflip_sim.cppのコード行数マップ

```
L1-14     ヘッダコメント・チャンネル表
L15-37    includes, 定数, FlipParticle, globals
L39-58    zeroVecChannel, zeroFloatChannel
L60-121   index helpers (getBidVid*, getF, setF, getVC, setVC, isFluid, copyVecChannel)
L123-142  MAC interpolation (interpComp, interpU/V/W)
L144-172  saveParticleSlice
L174-257  initParticles, particleToGrid（P2G + activeBlocks）
L259-280  applyGravity
L282-412  PCG solver (pressureDiag, applyPressureMatrix, buildMIC0Precon,
          applyMIC0Precon, dotFluid)
L414-540  solvePressurePCG, pressureProjection
L542-575  gridToParticle, advectParticles
L577-end  flip_dam_break (entry point)
```
