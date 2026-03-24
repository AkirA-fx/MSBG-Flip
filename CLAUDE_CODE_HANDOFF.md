# Claude Code 移行ドキュメント
## プロジェクト：大規模2相流体シミュレーション実装

---

## 背景・最終目標

波・泡・フォームなどのホワイトウォーターエフェクトを、物理ベースで大規模にシミュレートするシステムの実装。
最終的にはHoudiniへの連携も視野に入れている。

---

## 参照論文

### 論文1：Guided Bubbles and Wet Foam（SIGGRAPH 2022）
- 著者：Wretborn, Flynn, Stomakhin（Weta Digital）
- 内容：泡とフォームのホワイトウォーターシミュレーション
- URL：https://alexey.stomakhin.com/research/siggraph2022_whitewater.pdf

### 論文2：Adaptive Phase-Field-FLIP（SIGGRAPH 2025）★メイン実装対象
- 著者：Braun, Bender, Thuerey（TU Munich / RWTH Aachen）
- 内容：超大規模2相流体シミュレーション
- URL：https://ge.in.tum.de/publications/very-large-scale-two-phase-flip/

---

## 実装計画（フェーズ構成）

### ✅ Phase 0：環境構築（完了）
- MSYS2/MinGW64環境、GCC 15.2.0、TBB, libpng, libjpeg
- MSBGのビルド成功、msbg_demoの動作確認済み

### ✅ Phase 1：コードリーディング（完了）

### ✅ Phase 2：単相FLIPの実装（完了）
- ダムブレイクシミュレーション動作確認済み

### ✅ Phase 3：Phase-Field（密度場）の導入（完了）
- 2相パーティクル（水+空気、密度比1000:1）
- P2G + φ→β計算、界面クランプ、β下限ガード
- 可変密度ポアソン方程式（MIC(0)-PCG）
- 2相可視化（水=青、空気=灰）

### ✅ Phase 4：AMG圧力ソルバー + 最適化（完了）
実装内容:
1. **compact圧力システム** — PressureSystem構造体、CSR行列、getF/setF排除
2. **hypre/BoomerAMG + PCG** — 反復数84回→5-7回、収束失敗完全解消
3. **ソルバー切替機能** — `gSolverKind`: MIC0_PCG / HYPRE_AMG_PCG
4. **CSRパターンキャッシュ** — diagPos/xmPos等でtopo不変時はval更新のみ
5. **frozen preconditioner** — AMG Setup 8ステップに1回、反復数急増時は自動再setup
6. **TBB並列化** — P2G (enumerable_thread_specific)、G2P、advect、数値更新

パフォーマンス (128³, 1678万パーティクル):
| 指標 | Phase 3 初版 | Phase 4 最終版 |
|------|------------|--------------|
| 1ステップ時間 | 66秒 | 4.7秒 (topo不変) / 11秒 (topo変化) |
| PCG反復数 | 84-200+ | 5-7回 |
| PCG収束失敗 | 9/20ステップ | 0 |

### 🔲 Phase 5：Python連携（次の工程）
pybind11を使ってC++実装にPythonバインディングを追加。
可視化はPythonから行い（PyVistaなど）、重い計算はC++に任せる構成。

---

## 開発環境

| 項目 | 内容 |
|------|------|
| OS | Windows 10 Pro |
| シェル | MSYS2 MINGW64 |
| コンパイラ | GCC 15.2.0 |
| 並列ライブラリ | TBB |
| 圧力ソルバー | hypre 2.33.0 (BoomerAMG + PCG) |
| MPI | MS-MPI 10.1.1 |
| ビルドシステム | make（mkスクリプト） |
| リポジトリ | https://github.com/tum-pbs/MSBG |

### ビルド手順
```bash
cd build/
../mk
# または:
# PATH="/c/msys64/usr/bin:/c/msys64/mingw64/bin:$PATH" make -f ../makefile OBJE=o CPP_FLAGS="-std=gnu++17" ...
```

### 実行
```bash
PATH="/c/msys64/mingw64/bin:$PATH" ./msbg_demo.exe -c3 -r64 -b16   # 64³
PATH="/c/msys64/mingw64/bin:$PATH" ./msbg_demo.exe -c3 -r128 -b16  # 128³
```

### 依存パッケージ (MSYS2)
```
pacman -S mingw-w64-x86_64-hypre mingw-w64-x86_64-msmpi
```
+ MS-MPI ランタイム (msmpisetup.exe) の別途インストールが必要

### ファイル構成
```
MSBG/
├── src/
│   ├── flip_sim.cpp     ← Phase-Field FLIP 本体（~1050行）
│   ├── flip_sim.h       ← エントリポイント宣言
│   ├── main.cpp         ← -c3 で flip_dam_break() を呼び出し
│   ├── msbg.cpp/h       ← MSBGコアデータ構造
│   └── ...
├── build/               ← ビルド出力
├── makefile             ← -lHYPRE -lmsmpi 追加済み
├── mk                   ← ビルドスクリプト
├── PHASE3_DESIGN.md     ← Phase 3 設計書
└── CLAUDE_CODE_HANDOFF.md ← このファイル
```

### flip_sim.cpp の構成 (~1050行)
```
L1-30      ヘッダ、includes (hypre含む)
L31-50     定数、FlipParticle、グローバル
L51-130    ヘルパー (zero, index, getF/setF, getVC/setVC, isFluid, copy, interp)
L150-220   可視化 (saveParticleSlice: 2相色分け)
L220-435   particleToGrid (TBB並列P2G + φ→β + 界面クランプ)
L435-480   applyGravity
L456-590   PressureSystem構造体 + buildPressureSystem (パターンキャッシュ + TBB数値更新)
L590-680   compact PCG (MIC(0): buildMIC0Compact, applyMIC0Compact, csrSpmv)
L680-820   hypre AMG-PCG (永続オブジェクト + frozen preconditioner)
L820-880   pressureProjection (ソルバー切替 + 圧力勾配補正)
L880-1000  G2P (TBB並列), advect (TBB並列)
L1000-end  flip_dam_break エントリポイント (タイミング計測付き)
```

---

## ユーザーについて

- Pythonメインのエンジニア
- C++はこのプロジェクトで本格的に学び始めた初心者
- Codex CLI (OpenAI) をセカンドオピニオンとして活用するワークフローを好む
- 最終的にはPythonとC++を連携させたい
- Houdiniへの応用も視野に入れている
