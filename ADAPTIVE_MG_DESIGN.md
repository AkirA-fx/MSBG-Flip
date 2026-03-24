# §3.4 適応型MGソルバー設計書

## 現状
- Step 1-4 完了: 3レベルグリッド、refinement map毎step更新動作確認済み
- P2G/G2P/圧力はlevel=0前提で動作中
- 圧力ソルバーは compact CSR + hypre AMG-PCG

## 目標
MSBGカーネルを改造し、フル適応MGソルバーを自作する。

---

## Phase 1: msbg3.cppカーネルにfaceCoeff対応を追加

### 変更箇所
`processBlockLaplacian()` 内の `tmpDataW` ロード部分（L488付近）。

**現状:** `W.load_a(dataFaceAreaU + idx)` のみ
**変更後:** `W.load_a(dataFaceAreaU + idx); C.load_a(dataFaceCoeffU + idx); W *= C;`

6箇所のW1-W6ロードで同じパターンを適用。

### procNeighMixedRes2の変更
シグネチャに `float *dataFC2` を追加:
```cpp
void procNeighMixedRes2(
    const int&level, const int&level2, const double& h,
    CellFlags *dataFlg2, PSFloat *data2,
    float *dataFA2, float *dataFC2,  // ← 追加
    int vx, int vy, int vz, int vid,
    double& sumVal, double& sumCoeff0)
```

`SUB_CONTRIB` マクロ内で `if(dataFC2) a *= dataFC2[...]` を追加。

---

## Phase 2: V-cycle wrapper + PCG solver

### 新規関数
1. `prolongateCellCopyAdd()` — coarse→fine 2x2x2コピー加算
2. `msbgVCycle()` — pre-smooth→residual→restrict→coarse solve→prolongate→post-smooth
3. `solvePressureMsbgPCG()` — PCG outer solver + V-cycle preconditioner

### チャンネル使用
| チャンネル | 用途 |
|-----------|------|
| CH_PRESSURE | 解 x |
| CH_DIVERGENCE | RHS b / 初期段階で r も兼用 |
| CH_CG_P | PCG探索方向 p / coarseのRHS |
| CH_CG_Q | PCG q = A*p |
| CH_FLOAT_TMP_PS | residual / relax temp |
| CH_DIAGONAL | inverse diagonal (1/diag) |
| CH_FACE_AREA | face area (cut-cell用、今は1.0) |
| CH_FACE_COEFF | face coefficient β = 1/ρ |

### V-cycle擬似コード
```
msbgVCycle(levelMg, chRhs, chX):
  if levelMg == coarsest:
    relax(levelMg, chX, chRhs, nCoarse=24)
    return
  relax(levelMg, chX, chRhs, nPre=2)           // pre-smooth
  multiplyLaplacianMatrixOpt(CALC_RESIDUAL, ...) // r = b - Ax
  downsampleChannel(levelMg+1, r)                // restrict
  zero(chX, levelMg+1)
  msbgVCycle(levelMg+1, r_coarse, chX)          // recurse
  prolongateCellCopyAdd(levelMg, chX)             // prolongate
  relax(reverse, levelMg, chX, chRhs, nPost=2)   // post-smooth
```

---

## Phase 3: flip_sim.cpp統合

### 新規関数
1. `buildMsbgPressureLevel0()` — cell flags, divergence, faceCoeff, faceArea, diagonal設定
2. `pressureProjectionMSBG()` — 既存compact/hypre系を置換

### cell flags規則
- liquid: `flags = 0`
- air/free-surface: `flags = CELL_VOID`
- solid: `flags = CELL_SOLID`

### faceCoeff設定
```cpp
fcU[vid] = (c || xm) ? betaXm : 0.f;  // β = 1/ρ
faU[vid] = (c || xm) ? 1.f : 0.f;     // face area (cut-cellなし)
```

### inverse diagonal
```cpp
diag = sum(faceArea * faceCoeff) for all 6 faces
dinv[vid] = (diag > eps) ? 1.0/diag : 0
```

---

## 実装順序
1. Phase 2-3 を先に実装（flip_sim.cpp内で完結、msbg3.cppは触らない）
   - ただしfaceCoeff対応なしでは均一密度ラプラシアンになる
   - まず均一密度でMGソルバー動作確認
2. Phase 1 のカーネル改造（faceCoeff対応）
3. 可変密度でのテスト
4. マルチレベル対応（coarse level hierarchy構築）
