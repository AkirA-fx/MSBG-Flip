/******************************************************************************
 * vdb_in.h  -  Phase 5b: Houdini -> MSBG VDB input (collider SDF)
 *
 * Houdini で書き出された OpenVDB ファイルを読み、シミュ空間に展開する。
 ******************************************************************************/
#ifndef VDB_IN_H
#define VDB_IN_H

#include <string>
#include <vector>

namespace VdbIn {

// VDB ファイルから signed distance field を読み、シミュ空間 (sx*sy*sz)
// に同サイズ float 配列として返す。
//   value < 0  : solid 内部
//   value == 0 : 表面
//   value > 0  : 外側 (流体可動領域)
//
// grid 選択ルール:
//   1. grid 名 "solid_sdf" を最優先
//   2. なければ最初の FloatGrid を使う
//
// VDB の voxel space とシミュ space は同じスケール (1 voxel = 1 cell) を仮定。
// background 値で sdfOut を初期化するので、narrow-band の外は外側扱い。
//
// 戻り値: 成功で true。失敗時 sdfOut 不変、false。
bool readColliderSDF(const std::string& path,
                     int sx, int sy, int sz,
                     std::vector<float>& sdfOut);

} // namespace VdbIn

#endif // VDB_IN_H
