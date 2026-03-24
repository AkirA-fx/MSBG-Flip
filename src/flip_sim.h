/******************************************************************************
 * flip_sim.h
 * 単相FLIPシミュレーション（ダムブレイク）
 * Phase 2 実装
 ******************************************************************************/
#ifndef FLIP_SIM_H
#define FLIP_SIM_H

// ダムブレイクシミュレーションのエントリポイント
// resolution : グリッド解像度（例: 64）
// blockSize  : MSBGブロックサイズ（16 or 32）
// nSteps     : シミュレーションステップ数
int flip_dam_break( int resolution, int blockSize, int nSteps );

#endif // FLIP_SIM_H
