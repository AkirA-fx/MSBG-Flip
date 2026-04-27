/******************************************************************************
 * camera_io.h  -  Phase 5a Step C: camera path I/O + frustum / LOD helpers
 *
 * Houdini 等から書き出されたカメラ列を読み込み、updateRefinementMap() から
 * 視錐台カリングと pixel-size LOD を行うためのユーティリティ。
 ******************************************************************************/
#ifndef CAMERA_IO_H
#define CAMERA_IO_H

#include <string>
#include <vector>

namespace CameraIO {

struct CameraFrame {
    int   frame = 0;
    float eye[3]    = {0,0,0};
    float lookAt[3] = {0,0,-1};
    float up[3]     = {0,1,0};
    float fovY      = 0.785398f;  // 45°
    float aspect    = 1.7777f;
    float nearZ     = 0.1f;
    float farZ      = 1e6f;
};

// 1 行 1 フレームの TSV/CSV-ish テキストを読む。
// フォーマット: frame ex ey ez  ax ay az  ux uy uz  fovY aspect near far
//   - 区切りは whitespace (タブ or スペース) どちらでも可
//   - 行頭 '#' はコメント
//   - 空行はスキップ
//   - 1ファイルに任意フレーム分を含められる (frame 番号は昇順前提)
// 戻り値: 読めたフレーム数。失敗時は 0 を返し frames は空のまま。
size_t readCameraPath(const std::string& path,
                      std::vector<CameraFrame>& frames);

// frame 番号 N に対して、frames から該当する CameraFrame を選ぶ。
// 完全一致が無ければ最後に N 以下のフレームを返す。
// frames が空なら nullptr。
const CameraFrame* findCameraForFrame(
    const std::vector<CameraFrame>& frames, int frame);

// world 座標 p が cam の視錐台内にあるか (6-plane test)。
bool inFrustum(const CameraFrame& cam, const float p[3]);

// world 座標 p について、ボクセルブロックの世界スケール blockSizeWorld と
// 画像高さ imageHeight を踏まえた「画面上の 1 ボクセルの px サイズ」と、
// それに対応する MSBG レベル (0..maxLevel) を返す。
//   pixelSizeWorld = 2 * dist * tan(fovY/2) / imageHeight
//   blockSizeWorld * 2^level <= pixelSizeWorld * lodScale なら粗くて OK
int levelFromPixelSize(const CameraFrame& cam,
                       const float p[3],
                       float blockSizeWorld,
                       int   imageHeight,
                       float lodScale,
                       int   maxLevel);

} // namespace CameraIO

#endif // CAMERA_IO_H
