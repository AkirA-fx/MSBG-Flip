/******************************************************************************
 * flip_sim.cpp  -  Phase-Field FLIP: MAC格子 + 可変密度 MIC(0)-PCG
 *
 * チャンネル:
 *   CH_VEC3_4       現在速度 (x=u面, y=v面, z=w面)
 *   CH_VEC3_2       投影前速度
 *   CH_VEC3_3       フェース β = 1/ρ (x=u面, y=v面, z=w面)
 *   CH_FLOAT_1      cell mass (全相合算)
 *   CH_FLOAT_2      PCG残差 r
 *   CH_FLOAT_3      前処理後 z = M^{-1} r
 *   CH_FLOAT_4      探索方向 d
 *   CH_FLOAT_6      q = A*d
 *   CH_DIVERGENCE
 *   CH_PRESSURE
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <tbb/tbb.h>
#include "msbg.h"
#include "bitmap.h"
#include "flip_sim.h"

// hypre (AMG preconditioner)
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

static const float GRAVITY     = -9.8f;
static const float DT          = 0.05f;
static const int   PPV         = 8;
static const float FLIP_ALPHA  = 0.95f;
static const float MASS_EPS    = 1e-5f;
static const float PCG_TOL     = 5e-4f;
static const int   PCG_MAXITER = 500;

// 2相パラメータ
static const float RHO_L       = 1000.0f;
static const float RHO_G       = 1.0f;
static const float ALPHA_PHI   = 1.0f;
static const float BETA_MIN    = 1e-6f;  // β下限ガード（ゼロ除算/数値不安定防止）

// Step 4/6: ソルバー選択
enum class PressureSolverKind { MIC0_PCG, HYPRE_AMG_PCG, MSBG_VCYCLE_PCG };
static PressureSolverKind gSolverKind = PressureSolverKind::MSBG_VCYCLE_PCG;

struct FlipParticle { Vec3Float pos, vel; int phase; /* 0=liquid 1=air */ };
static std::vector<FlipParticle> gParticles;
static std::vector<int>          gActiveBlocks;

//=== zero-clear =============================================================
static void zeroVecChannel( SBG::SparseGrid<Vec3Float> *sg )
{
    for( int bid = 0; bid < sg->nBlocks(); bid++ )
    {
        Vec3Float *d = sg->getBlockDataPtr(bid);
        if(!d) continue;
        const int n = sg->nVoxelsInBlock();
        for(int i=0;i<n;i++) d[i]=Vec3Float(0,0,0);
    }
}
static void zeroFloatChannel( SBG::SparseGrid<float> *sg )
{
    for( int bid = 0; bid < sg->nBlocks(); bid++ )
    {
        float *d = sg->getBlockDataPtr(bid);
        if(!d) continue;
        const int n = sg->nVoxelsInBlock();
        for(int i=0;i<n;i++) d[i]=0.f;
    }
}

//=== index helpers ==========================================================
static bool getBidVidF( SBG::SparseGrid<float> *sg,
                        int ix,int iy,int iz, int &bid,int &vid )
{
    const int bl=sg->bsxLog2(),bm=sg->bsx()-1,nbx=sg->nbx(),nby=sg->nby();
    if(ix<0||iy<0||iz<0||ix>=sg->sx()||iy>=sg->sy()||iz>=sg->sz()) return false;
    bid=(ix>>bl)+(iy>>bl)*nbx+(iz>>bl)*nbx*nby;
    const int bsx=sg->bsx();
    vid=(ix&bm)+(iy&bm)*bsx+(iz&bm)*bsx*bsx;
    return true;
}
static bool getBidVidV( SBG::SparseGrid<Vec3Float> *sg,
                        int ix,int iy,int iz, int &bid,int &vid )
{
    const int bl=sg->bsxLog2(),bm=sg->bsx()-1,nbx=sg->nbx(),nby=sg->nby();
    if(ix<0||iy<0||iz<0||ix>=sg->sx()||iy>=sg->sy()||iz>=sg->sz()) return false;
    bid=(ix>>bl)+(iy>>bl)*nbx+(iz>>bl)*nbx*nby;
    const int bsx=sg->bsx();
    vid=(ix&bm)+(iy&bm)*bsx+(iz&bm)*bsx*bsx;
    return true;
}
static float getF( SBG::SparseGrid<float> *sg,int ix,int iy,int iz )
{
    int bid,vid;
    if(!getBidVidF(sg,ix,iy,iz,bid,vid)) return 0.f;
    float *d=sg->getBlockDataPtr(bid); return d?d[vid]:0.f;
}
static void setF( SBG::SparseGrid<float> *sg,int ix,int iy,int iz,float v )
{
    int bid,vid;
    if(!getBidVidF(sg,ix,iy,iz,bid,vid)) return;
    float *d=sg->getBlockDataPtr(bid); if(d) d[vid]=v;
}
static float getVC( SBG::SparseGrid<Vec3Float> *sg,int ix,int iy,int iz,int c )
{
    int bid,vid;
    if(!getBidVidV(sg,ix,iy,iz,bid,vid)) return 0.f;
    Vec3Float *d=sg->getBlockDataPtr(bid);
    if(!d) return 0.f;
    return c==0?d[vid].x:(c==1?d[vid].y:d[vid].z);
}
static void setVC( SBG::SparseGrid<Vec3Float> *sg,int ix,int iy,int iz,int c,float v )
{
    int bid,vid;
    if(!getBidVidV(sg,ix,iy,iz,bid,vid)) return;
    Vec3Float *d=sg->getBlockDataPtr(bid); if(!d) return;
    if(c==0) d[vid].x=v; else if(c==1) d[vid].y=v; else d[vid].z=v;
}
static bool isFluid( SBG::SparseGrid<float> *sgMass,int ix,int iy,int iz )
{ return getF(sgMass,ix,iy,iz)>MASS_EPS; }

static void copyVecChannel( SBG::SparseGrid<Vec3Float> *dst,
                            SBG::SparseGrid<Vec3Float> *src )
{
    for(int bid=0;bid<src->nBlocks();bid++)
    {
        Vec3Float *s=src->getBlockDataPtr(bid),*d=dst->getBlockDataPtr(bid);
        if(!s||!d) continue;
        const int n=src->nVoxelsInBlock(); for(int i=0;i<n;i++) d[i]=s[i];
    }
}

//=== MAC interpolation ======================================================
static float interpComp( SBG::SparseGrid<Vec3Float> *sg,
                         float px,float py,float pz,
                         float ox,float oy,float oz,int comp )
{
    const float gx=px-ox,gy=py-oy,gz=pz-oz;
    const int ix0=(int)floorf(gx),iy0=(int)floorf(gy),iz0=(int)floorf(gz);
    const float fx=gx-ix0,fy=gy-iy0,fz=gz-iz0;
    float sum=0.f;
    for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
        sum+=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz)
             *getVC(sg,ix0+dx,iy0+dy,iz0+dz,comp);
    return sum;
}
static float interpU(SBG::SparseGrid<Vec3Float>*sg,float px,float py,float pz)
{ return interpComp(sg,px,py,pz,0.f,0.5f,0.5f,0); }
static float interpV(SBG::SparseGrid<Vec3Float>*sg,float px,float py,float pz)
{ return interpComp(sg,px,py,pz,0.5f,0.f,0.5f,1); }
static float interpW(SBG::SparseGrid<Vec3Float>*sg,float px,float py,float pz)
{ return interpComp(sg,px,py,pz,0.5f,0.5f,0.f,2); }

//=== visualization ==========================================================
static void saveParticleSlice(int step,int sx,int sy,int sz)
{
    const int ZOOM=4;
    BmpBitmap *B=BmpNewBitmap(sx*ZOOM,sy*ZOOM,BMP_RGB|BMP_CLEAR);
    if(!B){TRCERR(("BmpNewBitmap failed\n"));return;}
    std::vector<int> cntL(sx*sy,0), cntA(sx*sy,0);
    const float zCenter=sz*0.5f,zHalf=sz*0.15f;
    for(const FlipParticle &p:gParticles)
    {
        if(fabsf(p.pos.z-zCenter)>zHalf) continue;
        const int ix=(int)floorf(p.pos.x),iy=(int)floorf(p.pos.y);
        if(ix>=0&&ix<sx&&iy>=0&&iy<sy)
        { if(p.phase==0) cntL[ix+iy*sx]++; else cntA[ix+iy*sx]++; }
    }
    int maxL=1,maxA=1;
    for(int c:cntL) maxL=std::max(maxL,c);
    for(int c:cntA) maxA=std::max(maxA,c);
    for(int iy=0;iy<sy;iy++) for(int ix=0;ix<sx;ix++)
    {
        const int cl=cntL[ix+iy*sx], ca=cntA[ix+iy*sx];
        int col=0;
        if(cl>0)
        { const float t=(float)cl/maxL;
          col=BMP_MKRGB((int)(30+t*80),(int)(100+t*120),(int)(200+t*55)); }
        else if(ca>0)
        { const float t=(float)ca/maxA;
          col=BMP_MKRGB((int)(60+t*60),(int)(60+t*60),(int)(60+t*60)); }
        else continue;
        const int py0=(sy-1-iy)*ZOOM;
        for(int dy=0;dy<ZOOM;dy++) for(int dx=0;dx<ZOOM;dx++)
        { const int px=ix*ZOOM+dx,py=py0+dy;
          if(px<sx*ZOOM&&py>=0&&py<sy*ZOOM) B->data[px+B->sx*py]=col; }
    }
    char fname[256]; sprintf(fname,"c:/tmp/flip_frame_%04d.png",step);
    BmpSaveBitmapPNG(B,fname,NULL,0); BmpDeleteBitmap(&B);
    TRCP(("Saved %s\n",fname));
}

//=== §3.4 Adaptive Grid: refinement map =====================================
static void buildRefinementMapFromParticles(
    MSBG::MultiresSparseGrid *msbg,
    std::vector<int> &blockLevels )
{
    SBG::SparseGrid<Vec3Float> *sg0=msbg->sg0();
    const int nBlk=msbg->nBlocks();
    const int nLevels=msbg->getNumLevels();
    const int coarsest=nLevels-1;
    blockLevels.assign(nBlk,coarsest);

    // BFS: 液体ブロックからのblock距離を計算
    const int INF=0x7fffffff;
    std::vector<int> dist(nBlk,INF);
    std::queue<int> q;

    for(const FlipParticle &p:gParticles)
    {
        if(p.phase!=0) continue; // liquid only
        int ix=(int)floorf(p.pos.x), iy=(int)floorf(p.pos.y), iz=(int)floorf(p.pos.z);
        ix=std::max(0,std::min(ix,(int)sg0->sx()-1));
        iy=std::max(0,std::min(iy,(int)sg0->sy()-1));
        iz=std::max(0,std::min(iz,(int)sg0->sz()-1));
        int bx,by,bz;
        sg0->getBlockCoords(ix,iy,iz,bx,by,bz);
        if(!sg0->blockCoordsInRange(bx,by,bz)) continue;
        int bid=sg0->getBlockIndex(bx,by,bz);
        if(dist[bid]==0) continue;
        dist[bid]=0; q.push(bid);
    }

    // 26近傍で距離を伝播
    while(!q.empty())
    {
        int bid=q.front(); q.pop();
        int bx,by,bz;
        sg0->getBlockCoordsById(bid,bx,by,bz);
        int d=dist[bid];
        if(d>=coarsest+1) continue;
        for(int dz=-1;dz<=1;dz++) for(int dy=-1;dy<=1;dy++) for(int dx=-1;dx<=1;dx++)
        {
            if(dx==0&&dy==0&&dz==0) continue;
            int bx2=bx+dx,by2=by+dy,bz2=bz+dz;
            if(!sg0->blockCoordsInRange(bx2,by2,bz2)) continue;
            int bid2=sg0->getBlockIndex(bx2,by2,bz2);
            if(dist[bid2]<=d+1) continue;
            dist[bid2]=d+1; q.push(bid2);
        }
    }

    // 距離→レベル変換
    for(int bid=0;bid<nBlk;bid++)
    {
        int d=dist[bid];
        if(d<=1) blockLevels[bid]=0;
        else if(nLevels>=2 && d==2) blockLevels[bid]=1;
        else blockLevels[bid]=coarsest;
    }
    msbg->regularizeRefinementMap(blockLevels.data());

    int cnt[3]={0,0,0};
    for(int bid=0;bid<nBlk;bid++)
    { int lv=blockLevels[bid]; if(lv<3) cnt[lv]++; }
    TRCP(("  refinement: L0=%d L1=%d L2=%d / %d blocks\n",cnt[0],cnt[1],cnt[2],nBlk));
}

//=== 1. particles ============================================================
static void initParticles(int sx,int sy,int sz)
{
    const int damX=sx/2,damY=sy/2,nSub=2;
    const float step=1.f/nSub;
    gParticles.clear();
    gParticles.reserve(sx*sy*sz*PPV);
    for(int iz=0;iz<sz;iz++) for(int iy=0;iy<sy;iy++) for(int ix=0;ix<sx;ix++)
    {
        const int phase=(ix<damX&&iy<damY)?0:1; // 0=liquid 1=air
        for(int kz=0;kz<nSub;kz++) for(int ky=0;ky<nSub;ky++) for(int kx=0;kx<nSub;kx++)
        {
            FlipParticle p;
            p.pos=Vec3Float(ix+(kx+0.5f)*step,iy+(ky+0.5f)*step,iz+(kz+0.5f)*step);
            p.vel=Vec3Float(0,0,0);
            p.phase=phase;
            gParticles.push_back(p);
        }
    }
    int nL=0,nA=0;
    for(const FlipParticle &p:gParticles) { if(p.phase==0) nL++; else nA++; }
    TRCP(("initParticles: %d particles (liquid=%d air=%d) dam=%dx%dx%d\n",
          (int)gParticles.size(),nL,nA,damX,damY,sz));
}

//=== 2. MAC P2G + phase-field + beta ========================================
static void particleToGrid( SBG::SparseGrid<Vec3Float> *sgVel,
                            SBG::SparseGrid<float>     *sgMass,
                            SBG::SparseGrid<Vec3Float> *sgBeta )
{
    zeroVecChannel(sgVel); zeroFloatChannel(sgMass); zeroVecChannel(sgBeta);
    const int nvox=sgVel->nVoxelsInBlock();
    const int nElem=sgVel->nBlocks()*nvox;
    std::vector<float> wU(nElem,0.f),wV(nElem,0.f),wW(nElem,0.f);
    std::vector<float> rhoU(nElem,0.f),rhoV(nElem,0.f),rhoW(nElem,0.f);
    std::vector<int>   cellLiqCnt(nElem,0);

    // スレッドローカル散布バッファ
    struct P2GBuf {
        std::vector<float> vx,vy,vz,mass,wu,wv,ww,ru,rv,rw;
        std::vector<int>   liq;
        void init(int n){
            vx.assign(n,0.f); vy.assign(n,0.f); vz.assign(n,0.f);
            mass.assign(n,0.f);
            wu.assign(n,0.f); wv.assign(n,0.f); ww.assign(n,0.f);
            ru.assign(n,0.f); rv.assign(n,0.f); rw.assign(n,0.f);
            liq.assign(n,0);
        }
    };

    tbb::enumerable_thread_specific<P2GBuf> tlsBuf(
        [&](){ P2GBuf b; b.init(nElem); return b; });

    const size_t np=gParticles.size();
    const int bl=sgVel->bsxLog2(), bm=sgVel->bsx()-1;
    const int nbx=sgVel->nbx(), nby=sgVel->nby();
    const int bsx=sgVel->bsx();
    const int sxG=sgVel->sx(), syG=sgVel->sy(), szG=sgVel->sz();

    // 並列パーティクル散布
    tbb::parallel_for(tbb::blocked_range<size_t>(0,np),
        [&](const tbb::blocked_range<size_t> &range){
        P2GBuf &L=tlsBuf.local();

        // inline bid/vid計算（関数呼び出しオーバーヘッド削減）
        auto bv=[&](int ix,int iy,int iz,int &bid,int &vid)->bool{
            if(ix<0||iy<0||iz<0||ix>=sxG||iy>=syG||iz>=szG) return false;
            bid=(ix>>bl)+(iy>>bl)*nbx+(iz>>bl)*nbx*nby;
            vid=(ix&bm)+(iy&bm)*bsx+(iz&bm)*bsx*bsx;
            return true;
        };

        for(size_t pi=range.begin();pi<range.end();pi++)
        {
            const FlipParticle &p=gParticles[pi];
            const float mp=(p.phase==0)?RHO_L:RHO_G;

            // cell mass
            { int bid,vid;
              if(bv((int)floorf(p.pos.x),(int)floorf(p.pos.y),(int)floorf(p.pos.z),bid,vid))
              { int idx=bid*nvox+vid; L.mass[idx]+=1.f;
                if(p.phase==0) L.liq[idx]++; } }

            // U face (0, 0.5, 0.5)
            { float gx=p.pos.x,gy=p.pos.y-0.5f,gz=p.pos.z-0.5f;
              int ix0=(int)floorf(gx),iy0=(int)floorf(gy),iz0=(int)floorf(gz);
              float fx=gx-ix0,fy=gy-iy0,fz=gz-iz0;
              for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
              { float w=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz);
                int bid,vid;
                if(bv(ix0+dx,iy0+dy,iz0+dz,bid,vid))
                { int idx=bid*nvox+vid;
                  L.vx[idx]+=p.vel.x*w; L.wu[idx]+=w; L.ru[idx]+=mp*w; } } }

            // V face (0.5, 0, 0.5)
            { float gx=p.pos.x-0.5f,gy=p.pos.y,gz=p.pos.z-0.5f;
              int ix0=(int)floorf(gx),iy0=(int)floorf(gy),iz0=(int)floorf(gz);
              float fx=gx-ix0,fy=gy-iy0,fz=gz-iz0;
              for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
              { float w=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz);
                int bid,vid;
                if(bv(ix0+dx,iy0+dy,iz0+dz,bid,vid))
                { int idx=bid*nvox+vid;
                  L.vy[idx]+=p.vel.y*w; L.wv[idx]+=w; L.rv[idx]+=mp*w; } } }

            // W face (0.5, 0.5, 0)
            { float gx=p.pos.x-0.5f,gy=p.pos.y-0.5f,gz=p.pos.z;
              int ix0=(int)floorf(gx),iy0=(int)floorf(gy),iz0=(int)floorf(gz);
              float fx=gx-ix0,fy=gy-iy0,fz=gz-iz0;
              for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
              { float w=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz);
                int bid,vid;
                if(bv(ix0+dx,iy0+dy,iz0+dz,bid,vid))
                { int idx=bid*nvox+vid;
                  L.vz[idx]+=p.vel.z*w; L.ww[idx]+=w; L.rw[idx]+=mp*w; } } }
        }
    });

    // スレッドバッファ合算 → グリッドチャンネル + 重み配列
    for(auto &L:tlsBuf)
    {
        for(int i=0;i<nElem;i++)
        {
            wU[i]+=L.wu[i]; wV[i]+=L.wv[i]; wW[i]+=L.ww[i];
            rhoU[i]+=L.ru[i]; rhoV[i]+=L.rv[i]; rhoW[i]+=L.rw[i];
            cellLiqCnt[i]+=L.liq[i];
        }
    }
    // 速度・質量をグリッドに書き出し
    for(int bid=0;bid<sgVel->nBlocks();bid++)
    {
        Vec3Float *vd=sgVel->getBlockDataPtr(bid);
        float     *md=sgMass->getBlockDataPtr(bid);
        if(!vd) continue;
        const int off=bid*nvox;
        for(auto &L:tlsBuf)
        {
            for(int vid=0;vid<nvox;vid++)
            {
                vd[vid].x+=L.vx[off+vid];
                vd[vid].y+=L.vy[off+vid];
                vd[vid].z+=L.vz[off+vid];
                if(md) md[vid]+=L.mass[off+vid];
            }
        }
    }

    // phi 計算用パラメータ
    // rhoTilde0 = 均一分布でのカーネル重み合計 ≈ PPV（密度非依存）
    const float rhoTilde0=(float)PPV;
    const float etaPhi=logf(RHO_L/RHO_G);
    const float rhoTildeMin=etaPhi*RHO_G*rhoTilde0;
    const float phiDenom=ALPHA_PHI*rhoTilde0*RHO_L;

    // normalize velocity + compute beta + active blocks
    gActiveBlocks.clear();
    const int nx=sgMass->sx(),ny=sgMass->sy(),nz=sgMass->sz();
    for(int bid=0;bid<sgVel->nBlocks();bid++)
    {
        Vec3Float *vd=sgVel->getBlockDataPtr(bid);
        Vec3Float *bd=sgBeta->getBlockDataPtr(bid);
        float     *md=sgMass->getBlockDataPtr(bid);
        if(!vd) continue;
        const int off=bid*nvox; bool hasFluid=false;

        // block の x0,y0,z0 を計算
        const int nbx=sgMass->nbx(),nby=sgMass->nby(),bs=sgMass->bsx();
        const int bx=bid%nbx, by=(bid/nbx)%nby, bz=bid/(nbx*nby);
        const int x0=bx*bs, y0=by*bs, z0=bz*bs;

        for(int vid=0;vid<nvox;vid++)
        {
            // 速度正規化
            if(wU[off+vid]>MASS_EPS) vd[vid].x/=wU[off+vid]; else vd[vid].x=0.f;
            if(wV[off+vid]>MASS_EPS) vd[vid].y/=wV[off+vid]; else vd[vid].y=0.f;
            if(wW[off+vid]>MASS_EPS) vd[vid].z/=wW[off+vid]; else vd[vid].z=0.f;
            if(md&&md[vid]>MASS_EPS) hasFluid=true;

            // phi → beta 計算 (各フェース独立)
            if(bd)
            {
                const int lx=vid%bs, ly=(vid/bs)%bs, lz=vid/(bs*bs);
                const int ix=x0+lx, iy=y0+ly, iz=z0+lz;

                auto phiFromRho=[&](float rt)->float{
                    if(rt<rhoTildeMin) return 0.f;
                    float v=sqrtf(std::max(rt-rhoTildeMin,0.f)/phiDenom);
                    return std::min(v,1.f);
                };
                auto betaFromPhi=[](float phi)->float{
                    float rho=RHO_G+phi*(RHO_L-RHO_G);
                    return std::max(1.f/rho, BETA_MIN);
                };

                float phiU=phiFromRho(rhoU[off+vid]);
                float phiV=phiFromRho(rhoV[off+vid]);
                float phiW=phiFromRho(rhoW[off+vid]);

                // セル相判定: 0=liquid, 1=air, -1=empty/外
                auto cellPh=[&](int cx,int cy,int cz)->int{
                    int b2,v2;
                    if(!getBidVidF(sgMass,cx,cy,cz,b2,v2)) return -1;
                    float *md2=sgMass->getBlockDataPtr(b2);
                    if(!md2||md2[v2]<=MASS_EPS) return -1;
                    return (cellLiqCnt[b2*nvox+v2]*2>(int)(md2[v2]+0.5f))?0:1;
                };

                // 界面クランプ (論文§3.3):
                // 両側 liquid → phi=1, 両側 air → phi=0
                // U面: cell(ix-1,iy,iz) と cell(ix,iy,iz) の間
                { int pL=cellPh(ix-1,iy,iz), pR=cellPh(ix,iy,iz);
                  if(pL==0&&pR==0) phiU=1.f;
                  else if(pL==1&&pR==1) phiU=0.f; }
                // V面: cell(ix,iy-1,iz) と cell(ix,iy,iz) の間
                { int pB=cellPh(ix,iy-1,iz), pT=cellPh(ix,iy,iz);
                  if(pB==0&&pT==0) phiV=1.f;
                  else if(pB==1&&pT==1) phiV=0.f; }
                // W面: cell(ix,iy,iz-1) と cell(ix,iy,iz) の間
                { int pN=cellPh(ix,iy,iz-1), pF=cellPh(ix,iy,iz);
                  if(pN==0&&pF==0) phiW=1.f;
                  else if(pN==1&&pF==1) phiW=0.f; }

                bd[vid].x=betaFromPhi(phiU);
                bd[vid].y=betaFromPhi(phiV);
                bd[vid].z=betaFromPhi(phiW);
            }
        }
        if(hasFluid) gActiveBlocks.push_back(bid);
    }
}

//=== 3. gravity ==============================================================
static void applyGravity( SBG::SparseGrid<Vec3Float> *sgVel,
                          SBG::SparseGrid<Vec3Float> *sgVelOld,
                          SBG::SparseGrid<float>     *sgMass,
                          float dt )
{
    copyVecChannel(sgVelOld,sgVel);
    const int nx=sgMass->sx(),ny=sgMass->sy(),nz=sgMass->sz();
    for(int iz=0;iz<nz;iz++) for(int iy=0;iy<=ny;iy++) for(int ix=0;ix<nx;ix++)
    {
        const bool fLo=(iy>0 )&&isFluid(sgMass,ix,iy-1,iz);
        const bool fHi=(iy<ny)&&isFluid(sgMass,ix,iy,  iz);
        if(fLo||fHi) setVC(sgVel,ix,iy,iz,1,getVC(sgVel,ix,iy,iz,1)+GRAVITY*dt);
    }
    // solid BC
    for(int iy=0;iy<ny;iy++) for(int iz=0;iz<nz;iz++)
    { setVC(sgVel,0,iy,iz,0,0.f); setVC(sgVel,nx-1,iy,iz,0,0.f); }
    for(int ix=0;ix<nx;ix++) for(int iz=0;iz<nz;iz++)
        setVC(sgVel,ix,0,iz,1,0.f);
    for(int ix=0;ix<nx;ix++) for(int iy=0;iy<ny;iy++)
    { setVC(sgVel,ix,iy,0,2,0.f); setVC(sgVel,ix,iy,nz-1,2,0.f); }
}

//=== Phase 4: Compact Pressure System ========================================

// Step 1: データ構造
struct PressureSystem {
    int n = 0;                        // fluid cell 数（未知数数）
    int nx = 0, ny = 0, nz = 0;      // グリッド寸法
    std::vector<int> rowPtr;          // CSR行ポインタ [n+1]
    std::vector<int> colInd;          // CSR列インデックス
    std::vector<float> val;           // CSR値
    std::vector<int> cellToRow;       // linearized cell → compact row (-1 if not fluid)
    std::vector<int> rowToCell;       // compact row → linearized cell
    std::vector<float> rhs;           // 右辺ベクトル [n]
    std::vector<float> sol;           // 解ベクトル [n]
    // 各行の空間座標（MIC(0)の方向依存アクセス用）
    std::vector<short> rowIx, rowIy, rowIz;
    // 各行のβ値（MIC(0)用キャッシュ: x-面,y-面,z-面）
    std::vector<float> betaXm, betaYm, betaZm;
    // CSRパターンキャッシュ（valの位置を記録）
    std::vector<int> diagPos, xmPos, xpPos, ymPos, ypPos, zmPos, zpPos;
    int patternRevision = 0;
    bool topologyChanged = true;
    bool patternValid = false;
};

static PressureSystem gPSys;
static const int HYPRE_SETUP_INTERVAL = 8;

// Step 2: MSBG → CSR 行列構築 (パターン再利用 + TBB並列数値更新)

// CSRパターン構築（トポロジ変更時のみ実行）
static void rebuildPressurePattern(PressureSystem &sys)
{
    const int nx=sys.nx, ny=sys.ny, nz=sys.nz;
    auto lin=[=](int ix,int iy,int iz){ return ix+nx*(iy+ny*iz); };
    enum { XM,XP,YM,YP,ZM,ZP };
    struct Nbr { int col,dir; };

    sys.rowPtr.assign(sys.n+1,0);
    sys.colInd.clear(); sys.val.clear();
    sys.diagPos.assign(sys.n,-1);
    sys.xmPos.assign(sys.n,-1); sys.xpPos.assign(sys.n,-1);
    sys.ymPos.assign(sys.n,-1); sys.ypPos.assign(sys.n,-1);
    sys.zmPos.assign(sys.n,-1); sys.zpPos.assign(sys.n,-1);
    sys.colInd.reserve(sys.n*7); sys.val.reserve(sys.n*7);

    for(int row=0;row<sys.n;row++)
    {
        const int ix=sys.rowIx[row], iy=sys.rowIy[row], iz=sys.rowIz[row];
        Nbr nbrs[6]; int nNbr=0;
        auto addNbr=[&](int jx,int jy,int jz,int dir){
            if(jx<0||jx>=nx||jy<0||jy>=ny||jz<0||jz>=nz) return;
            int col=sys.cellToRow[lin(jx,jy,jz)];
            if(col>=0) nbrs[nNbr++]={col,dir};
        };
        addNbr(ix-1,iy,iz,XM); addNbr(ix+1,iy,iz,XP);
        addNbr(ix,iy-1,iz,YM); addNbr(ix,iy+1,iz,YP);
        addNbr(ix,iy,iz-1,ZM); addNbr(ix,iy,iz+1,ZP);
        for(int i=0;i<nNbr-1;i++) for(int j=i+1;j<nNbr;j++)
            if(nbrs[j].col<nbrs[i].col) std::swap(nbrs[i],nbrs[j]);

        int diagInserted=0;
        for(int i=0;i<nNbr;i++)
        {
            if(!diagInserted && nbrs[i].col>row){
                sys.diagPos[row]=(int)sys.colInd.size();
                sys.colInd.push_back(row); sys.val.push_back(0.f);
                diagInserted=1;
            }
            int p=(int)sys.colInd.size();
            sys.colInd.push_back(nbrs[i].col); sys.val.push_back(0.f);
            switch(nbrs[i].dir){
                case XM: sys.xmPos[row]=p; break; case XP: sys.xpPos[row]=p; break;
                case YM: sys.ymPos[row]=p; break; case YP: sys.ypPos[row]=p; break;
                case ZM: sys.zmPos[row]=p; break; case ZP: sys.zpPos[row]=p; break;
            }
        }
        if(!diagInserted){
            sys.diagPos[row]=(int)sys.colInd.size();
            sys.colInd.push_back(row); sys.val.push_back(0.f);
        }
        sys.rowPtr[row+1]=(int)sys.colInd.size();
    }
    sys.patternValid=true;
    ++sys.patternRevision;
}

// 数値更新（TBB並列、毎ステップ実行）
static void updatePressureNumerics(
    SBG::SparseGrid<Vec3Float> *sgBeta,
    SBG::SparseGrid<float>     *sgDiv,
    PressureSystem &sys )
{
    const int nx=sys.nx, ny=sys.ny, nz=sys.nz;
    sys.rhs.resize(sys.n);
    sys.betaXm.resize(sys.n); sys.betaYm.resize(sys.n); sys.betaZm.resize(sys.n);

    tbb::parallel_for(tbb::blocked_range<int>(0,sys.n),
        [&](const tbb::blocked_range<int> &r){
        for(int row=r.begin();row<r.end();row++)
        {
            const int ix=sys.rowIx[row], iy=sys.rowIy[row], iz=sys.rowIz[row];
            const float bxm=(ix  >0 )?getVC(sgBeta,ix,  iy,iz,0):0.f;
            const float bxp=(ix+1<nx)?getVC(sgBeta,ix+1,iy,iz,0):0.f;
            const float bym=(iy  >0 )?getVC(sgBeta,ix,iy,  iz,1):0.f;
            const float byp=(iy+1<ny)?getVC(sgBeta,ix,iy+1,iz,1):0.f;
            const float bzm=(iz  >0 )?getVC(sgBeta,ix,iy,iz,  2):0.f;
            const float bzp=(iz+1<nz)?getVC(sgBeta,ix,iy,iz+1,2):0.f;

            sys.rhs[row]=-getF(sgDiv,ix,iy,iz);
            sys.betaXm[row]=bxm; sys.betaYm[row]=bym; sys.betaZm[row]=bzm;

            sys.val[sys.diagPos[row]]=bxm+bxp+bym+byp+bzm+bzp;
            if(sys.xmPos[row]>=0) sys.val[sys.xmPos[row]]=-bxm;
            if(sys.xpPos[row]>=0) sys.val[sys.xpPos[row]]=-bxp;
            if(sys.ymPos[row]>=0) sys.val[sys.ymPos[row]]=-bym;
            if(sys.ypPos[row]>=0) sys.val[sys.ypPos[row]]=-byp;
            if(sys.zmPos[row]>=0) sys.val[sys.zmPos[row]]=-bzm;
            if(sys.zpPos[row]>=0) sys.val[sys.zpPos[row]]=-bzp;
        }
    });
}

static void buildPressureSystem(
    SBG::SparseGrid<float>     *sgMass,
    SBG::SparseGrid<Vec3Float> *sgBeta,
    SBG::SparseGrid<float>     *sgDiv,
    float dt,
    PressureSystem &sys )
{
    const int nx=sgMass->sx(), ny=sgMass->sy(), nz=sgMass->sz();
    sys.nx=nx; sys.ny=ny; sys.nz=nz;
    const int nCells=nx*ny*nz;
    auto lin=[=](int ix,int iy,int iz){ return ix+nx*(iy+ny*iz); };

    // fluid cell 列挙
    std::vector<int> newRowToCell;
    newRowToCell.reserve(sys.n>0?sys.n:nCells/4);
    std::vector<int> newCellToRow(nCells,-1);
    std::vector<short> newIx,newIy,newIz;

    for(int iz=0;iz<nz;iz++) for(int iy=0;iy<ny;iy++) for(int ix=0;ix<nx;ix++)
    {
        if(!isFluid(sgMass,ix,iy,iz)) continue;
        int row=(int)newRowToCell.size();
        newCellToRow[lin(ix,iy,iz)]=row;
        newRowToCell.push_back(lin(ix,iy,iz));
        newIx.push_back((short)ix); newIy.push_back((short)iy); newIz.push_back((short)iz);
    }

    // トポロジ変化判定（n一致ならパターン再利用）
    const bool topoSame=sys.patternValid &&
        (int)newRowToCell.size()==sys.n &&
        std::equal(newRowToCell.begin(),newRowToCell.end(),sys.rowToCell.begin());

    sys.topologyChanged=!topoSame;
    if(!topoSame)
    {
        sys.cellToRow.swap(newCellToRow);
        sys.rowToCell.swap(newRowToCell);
        sys.rowIx.swap(newIx); sys.rowIy.swap(newIy); sys.rowIz.swap(newIz);
        sys.n=(int)sys.rowToCell.size();
        sys.sol.assign(sys.n,0.f);
        rebuildPressurePattern(sys);
    }

    updatePressureNumerics(sgBeta,sgDiv,sys);

    TRCP(("  PressureSystem: n=%d nnz=%d (%.1f/row) topo=%s\n",
          sys.n,(int)sys.val.size(),(float)sys.val.size()/std::max(sys.n,1),
          topoSame?"reuse":"rebuild"));
}

// 解をMSBGグリッドに書き戻す
static void scatterPressureToGrid(
    const PressureSystem &sys,
    SBG::SparseGrid<float> *sgP )
{
    zeroFloatChannel(sgP);
    for(int row=0;row<sys.n;row++)
    {
        int ix=sys.rowIx[row], iy=sys.rowIy[row], iz=sys.rowIz[row];
        setF(sgP,ix,iy,iz,sys.sol[row]);
    }
}

// Step 3: Compact PCG (MIC(0)前処理付き)

// CSR SpMV: q = A * x
static void csrSpmv(const PressureSystem &A, const float *x, float *q)
{
    for(int i=0;i<A.n;i++)
    {
        float sum=0.f;
        for(int j=A.rowPtr[i];j<A.rowPtr[i+1];j++)
            sum+=A.val[j]*x[A.colInd[j]];
        q[i]=sum;
    }
}

// compact 内積
static double compactDot(const float *a, const float *b, int n)
{
    double s=0.0;
    for(int i=0;i<n;i++) s+=(double)a[i]*(double)b[i];
    return s;
}

// MIC(0) 構築 (compact版)
static void buildMIC0Compact(const PressureSystem &sys,
                              std::vector<float> &precon)
{
    precon.assign(sys.n,0.f);
    const float sigma=0.25f;
    const int nx=sys.nx, ny=sys.ny;
    auto lin=[=](int ix,int iy,int iz){ return ix+nx*(iy+ny*iz); };

    for(int row=0;row<sys.n;row++)
    {
        // 対角成分を CSR から取得
        float Adiag=0.f;
        for(int j=sys.rowPtr[row];j<sys.rowPtr[row+1];j++)
            if(sys.colInd[j]==row){ Adiag=sys.val[j]; break; }

        float e=Adiag;
        int ix=sys.rowIx[row], iy=sys.rowIy[row], iz=sys.rowIz[row];

        // x- 隣接
        if(ix>0){
            int r2=sys.cellToRow[lin(ix-1,iy,iz)];
            if(r2>=0){ float bx=sys.betaXm[row];
                       float p=precon[r2]; e-=(bx*p)*(bx*p); } }
        // y-
        if(iy>0){
            int r2=sys.cellToRow[lin(ix,iy-1,iz)];
            if(r2>=0){ float by=sys.betaYm[row];
                       float p=precon[r2]; e-=(by*p)*(by*p); } }
        // z-
        if(iz>0){
            int r2=sys.cellToRow[lin(ix,iy,iz-1)];
            if(r2>=0){ float bz=sys.betaZm[row];
                       float p=precon[r2]; e-=(bz*p)*(bz*p); } }

        precon[row]=1.f/sqrtf(std::max(e,sigma*std::max(Adiag,1e-10f)));
    }
}

// MIC(0) 適用 (compact版): z = M^{-1} r
static void applyMIC0Compact(const PressureSystem &sys,
                              const std::vector<float> &precon,
                              const float *r, float *z)
{
    const int n=sys.n, nx=sys.nx, ny=sys.ny;
    auto lin=[=](int ix,int iy,int iz){ return ix+nx*(iy+ny*iz); };
    std::vector<float> q(n);

    // 前進代入: L * q = r
    for(int row=0;row<n;row++)
    {
        int ix=sys.rowIx[row], iy=sys.rowIy[row], iz=sys.rowIz[row];
        float t=r[row];
        if(ix>0){ int r2=sys.cellToRow[lin(ix-1,iy,iz)];
                  if(r2>=0) t+=sys.betaXm[row]*precon[r2]*q[r2]; }
        if(iy>0){ int r2=sys.cellToRow[lin(ix,iy-1,iz)];
                  if(r2>=0) t+=sys.betaYm[row]*precon[r2]*q[r2]; }
        if(iz>0){ int r2=sys.cellToRow[lin(ix,iy,iz-1)];
                  if(r2>=0) t+=sys.betaZm[row]*precon[r2]*q[r2]; }
        q[row]=t*precon[row];
    }

    // 後退代入: L^T * z = q
    // betaXp/Yp/Zp は隣接行の betaXm/Ym/Zm を参照
    for(int row=n-1;row>=0;row--)
    {
        int ix=sys.rowIx[row], iy=sys.rowIy[row], iz=sys.rowIz[row];
        float pi=precon[row];
        float t=q[row];
        if(ix+1<sys.nx){ int r2=sys.cellToRow[lin(ix+1,iy,iz)];
                         if(r2>=0) t+=sys.betaXm[r2]*pi*z[r2]; }
        if(iy+1<sys.ny){ int r2=sys.cellToRow[lin(ix,iy+1,iz)];
                         if(r2>=0) t+=sys.betaYm[r2]*pi*z[r2]; }
        if(iz+1<sys.nz){ int r2=sys.cellToRow[lin(ix,iy,iz+1)];
                         if(r2>=0) t+=sys.betaZm[r2]*pi*z[r2]; }
        z[row]=t*pi;
    }
}

// Compact PCG 本体
static void solvePressureCompactPCG(PressureSystem &sys, float tol, int maxIter)
{
    const int n=sys.n;
    if(n==0) return;

    std::vector<float> &x=sys.sol;
    std::fill(x.begin(),x.end(),0.f);
    std::vector<float> r(n), z(n), d(n), q(n);
    std::vector<float> precon;

    // r = b (x=0 なので r=b)
    double rhsNormSq=0.0;
    for(int i=0;i<n;i++){ r[i]=sys.rhs[i]; rhsNormSq+=(double)r[i]*(double)r[i]; }
    if(rhsNormSq<=1e-30) return;

    buildMIC0Compact(sys,precon);
    applyMIC0Compact(sys,precon,r.data(),z.data());

    double rho=compactDot(r.data(),z.data(),n);
    for(int i=0;i<n;i++) d[i]=z[i];

    const double tolSq=(double)tol*(double)tol;
    int convergedIter=-1;

    for(int iter=0;iter<maxIter;iter++)
    {
        csrSpmv(sys,d.data(),q.data());
        const double dq=compactDot(d.data(),q.data(),n);
        if(dq<=1e-30||rho<=1e-30) break;

        const float alpha=(float)(rho/dq);
        for(int i=0;i<n;i++){ x[i]+=alpha*d[i]; r[i]-=alpha*q[i]; }

        const double rNormSq=compactDot(r.data(),r.data(),n);
        if(rNormSq<=tolSq*rhsNormSq){ convergedIter=iter+1; break; }

        applyMIC0Compact(sys,precon,r.data(),z.data());
        const double rhoNew=compactDot(r.data(),z.data(),n);
        if(rho<=1e-30) break;

        const float beta=(float)(rhoNew/rho);
        for(int i=0;i<n;i++) d[i]=z[i]+beta*d[i];
        rho=rhoNew;
    }

    const double finalRes=compactDot(r.data(),r.data(),n);
    if(convergedIter>0)
        { TRCP(("  MIC0-PCG converged in %d iter  |r|/|b|=%.2e\n",convergedIter,sqrt(finalRes/rhsNormSq))); }
    else
        { TRCP(("  MIC0-PCG: maxIter=%d reached  |r|/|b|=%.2e\n",maxIter,sqrt(finalRes/rhsNormSq))); }
}

// Step 5: Hypre AMG + PCG solver (永続オブジェクト再利用版)
static bool gHypreInitialized = false;
static HYPRE_IJMatrix  s_ijA  = NULL;
static HYPRE_IJVector  s_ijb  = NULL;
static HYPRE_IJVector  s_ijx  = NULL;
static HYPRE_Solver    s_amg  = NULL;
static HYPRE_Solver    s_pcg  = NULL;
static int             s_prevN = -1;
static int             s_prevPatternRevision = -1;
static int             s_setupCount = 0;
static int             s_stepsSinceSetup = HYPRE_SETUP_INTERVAL;

static void hypreCleanup()
{
    if(s_pcg){ HYPRE_ParCSRPCGDestroy(s_pcg); s_pcg=NULL; }
    if(s_amg){ HYPRE_BoomerAMGDestroy(s_amg); s_amg=NULL; }
    if(s_ijA){ HYPRE_IJMatrixDestroy(s_ijA);  s_ijA=NULL; }
    if(s_ijb){ HYPRE_IJVectorDestroy(s_ijb);  s_ijb=NULL; }
    if(s_ijx){ HYPRE_IJVectorDestroy(s_ijx);  s_ijx=NULL; }
    s_prevN=-1; s_prevPatternRevision=-1;
    s_stepsSinceSetup=HYPRE_SETUP_INTERVAL;
}

static void solvePressureHypreAMG(PressureSystem &sys, float tol, int maxIter)
{
    const int n = sys.n;
    if(n==0) return;

    if(!gHypreInitialized){
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if(!mpi_initialized){
            int argc_dummy = 0; char **argv_dummy = NULL;
            MPI_Init(&argc_dummy, &argv_dummy);
        }
        HYPRE_Init();
        gHypreInitialized = true;
    }

    MPI_Comm comm = MPI_COMM_WORLD;
    const bool needRebuild =
        (n != s_prevN) || (sys.patternRevision != s_prevPatternRevision);

    // トポロジが変わったらオブジェクトを再作成
    if(needRebuild)
    {
        hypreCleanup();
        const int ilower=0, iupper=n-1;

        HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &s_ijA);
        HYPRE_IJMatrixSetObjectType(s_ijA, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(s_ijA);

        HYPRE_IJVectorCreate(comm, ilower, iupper, &s_ijb);
        HYPRE_IJVectorCreate(comm, ilower, iupper, &s_ijx);
        HYPRE_IJVectorSetObjectType(s_ijb, HYPRE_PARCSR);
        HYPRE_IJVectorSetObjectType(s_ijx, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(s_ijb);
        HYPRE_IJVectorInitialize(s_ijx);

        // AMG preconditioner
        HYPRE_BoomerAMGCreate(&s_amg);
        HYPRE_BoomerAMGSetMaxLevels(s_amg, 25);
        HYPRE_BoomerAMGSetCoarsenType(s_amg, 6);
        HYPRE_BoomerAMGSetRelaxType(s_amg, 6);
        HYPRE_BoomerAMGSetNumSweeps(s_amg, 1);
        HYPRE_BoomerAMGSetStrongThreshold(s_amg, 0.25);
        HYPRE_BoomerAMGSetTol(s_amg, 0.0);
        HYPRE_BoomerAMGSetMaxIter(s_amg, 1);
        HYPRE_BoomerAMGSetPrintLevel(s_amg, 0);

        // PCG outer solver
        HYPRE_ParCSRPCGCreate(comm, &s_pcg);
        HYPRE_PCGSetMaxIter(s_pcg, maxIter);
        HYPRE_PCGSetTol(s_pcg, (double)tol);
        HYPRE_PCGSetTwoNorm(s_pcg, 1);
        HYPRE_PCGSetPrintLevel(s_pcg, 0);
        HYPRE_PCGSetPrecond(s_pcg,
                            (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                            (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                            s_amg);
        s_prevN = n;
        s_prevPatternRevision = sys.patternRevision;
    }

    // 行列値の更新
    for(int row=0; row<n; row++)
    {
        int ncols = sys.rowPtr[row+1] - sys.rowPtr[row];
        int *cols = &sys.colInd[sys.rowPtr[row]];
        std::vector<double> dvals(ncols);
        for(int j=0; j<ncols; j++)
            dvals[j] = (double)sys.val[sys.rowPtr[row]+j];
        HYPRE_IJMatrixSetValues(s_ijA, 1, &ncols, &row, cols, dvals.data());
    }
    HYPRE_IJMatrixAssemble(s_ijA);

    HYPRE_ParCSRMatrix parA;
    HYPRE_IJMatrixGetObject(s_ijA, (void**)&parA);

    // ベクトル値の更新
    {
        std::vector<int> indices(n);
        std::vector<double> bvals(n), xvals(n, 0.0);
        for(int i=0; i<n; i++){ indices[i]=i; bvals[i]=(double)sys.rhs[i]; }
        HYPRE_IJVectorSetValues(s_ijb, n, indices.data(), bvals.data());
        HYPRE_IJVectorSetValues(s_ijx, n, indices.data(), xvals.data());
    }
    HYPRE_IJVectorAssemble(s_ijb);
    HYPRE_IJVectorAssemble(s_ijx);

    HYPRE_ParVector parb, parx;
    HYPRE_IJVectorGetObject(s_ijb, (void**)&parb);
    HYPRE_IJVectorGetObject(s_ijx, (void**)&parx);

    // Frozen preconditioner: Setup は interval ごと or n が大幅変化時のみ
    const bool needSetup =
        needRebuild || (s_stepsSinceSetup >= HYPRE_SETUP_INTERVAL);
    if(needSetup){
        HYPRE_ParCSRPCGSetup(s_pcg, parA, parb, parx);
        ++s_setupCount;
        s_stepsSinceSetup = 0;
    }
    HYPRE_ParCSRPCGSolve(s_pcg, parA, parb, parx);
    ++s_stepsSinceSetup;

    // 結果取得
    HYPRE_Int numIter;
    double finalRelRes;
    HYPRE_PCGGetNumIterations(s_pcg, &numIter);
    HYPRE_PCGGetFinalRelativeResidualNorm(s_pcg, &finalRelRes);

    {
        std::vector<int> indices(n);
        std::vector<double> xvals(n);
        for(int i=0; i<n; i++) indices[i]=i;
        HYPRE_IJVectorGetValues(s_ijx, n, indices.data(), xvals.data());
        sys.sol.resize(n);
        for(int i=0; i<n; i++) sys.sol[i]=(float)xvals[i];
    }

    // 反復数急増時は次ステップで Setup 強制
    if(!needSetup && numIter>(HYPRE_Int)(0.8f*maxIter))
        s_stepsSinceSetup=HYPRE_SETUP_INTERVAL;

    { TRCP(("  AMG-PCG %d iter |r|/|b|=%.2e setup=%s\n",
            (int)numIter,finalRelRes,needSetup?"yes":"skip")); }
}

//=== Step 6: MSBG native V-cycle PCG solver ================================

// Callback context for FLIP → MSBG pressure solver
struct FlipPressureCtx {
    SBG::SparseGrid<Vec3Float> *sgVel;
    SBG::SparseGrid<float>     *sgMass;
    SBG::SparseGrid<Vec3Float> *sgBeta;
    float dt;
    int nx, ny, nz;
};

static MSBG::CellFlags flipCellTypeCB(void *user, int x, int y, int z)
{
    auto *ctx = (FlipPressureCtx*)user;
    if(x<0||y<0||z<0||x>=ctx->nx||y>=ctx->ny||z>=ctx->nz) return MSBG::CELL_SOLID;
    return isFluid(ctx->sgMass, x, y, z) ? 0 : MSBG::CELL_VOID;
}

static float flipFaceCoeffCB(void *user, int dir, int x, int y, int z)
{
    auto *ctx = (FlipPressureCtx*)user;
    return getVC(ctx->sgBeta, x, y, z, dir);
}

static float flipRhsCB(void *user, int x, int y, int z)
{
    auto *ctx = (FlipPressureCtx*)user;
    if(!isFluid(ctx->sgMass, x, y, z)) return 0.f;
    const float div =
        getVC(ctx->sgVel,x+1,y,  z,  0)-getVC(ctx->sgVel,x,y,z,0)+
        getVC(ctx->sgVel,x,  y+1,z,  1)-getVC(ctx->sgVel,x,y,z,1)+
        getVC(ctx->sgVel,x,  y,  z+1,2)-getVC(ctx->sgVel,x,y,z,2);
    return div / ctx->dt;
}

// Phase 2 legacy code removed — see git history (ed5d638, 7e444b0)
// buildMsbgPressureLevel0() and solvePressureMsbgPCG() were replaced
// by the MSBG-internal callback-based API (preparePressureSolveFLIP /
// solvePressureFLIP) in Phase 3.

#if 0  // --- BEGIN REMOVED Phase 2 legacy code ---
static void buildMsbgPressureLevel0_REMOVED(
    MSBG::MultiresSparseGrid *msbg,
    SBG::SparseGrid<Vec3Float> *sgVel,
    SBG::SparseGrid<float>     *sgMass,
    SBG::SparseGrid<Vec3Float> *sgBeta,
    float dt )
{
    using namespace MSBG;
    (void)sgBeta; // TODO: faceCoeff対応時に使用

    SBG::SparseGrid<float> *sgDiv = msbg->getFloatChannel(CH_DIVERGENCE, 0);
    SBG::SparseGrid<float> *sgP   = msbg->getFloatChannel(CH_PRESSURE, 0);
    const int nx = sgMass->sx(), ny = sgMass->sy(), nz = sgMass->sz();

    // Cell flags / Face area / Diagonal を全MG levelで初期化
    // processBlockLaplacian が getBlockDataPtr(bid) で直接参照するため
    // constant block (NULL) ではsegfaultする → 全blockにデータ割り当て必須
    const int nLev = msbg->getNumLevels();  // sparse levels

    // Helper lambda: sparse grid の全blockにデータを割り当てて値を埋める
    auto fillSG = [](SBG::SparseGrid<float> *sg, float val) {
        if(!sg) return;
        sg->prepareDataAccess(SBG::ACC_READ | SBG::ACC_WRITE);
        for(int bid = 0; bid < sg->nBlocks(); bid++) {
            float *d = sg->getBlockDataPtr(bid, 1, 0);
            if(!d) continue;
            for(int i = 0; i < sg->nVoxelsInBlock(); i++) d[i] = val;
        }
    };
    auto fillFlagsSG = [](SBG::SparseGrid<CellFlags> *sg) {
        if(!sg) return;
        sg->prepareDataAccess(SBG::ACC_READ | SBG::ACC_WRITE);
        for(int bid = 0; bid < sg->nBlocks(); bid++)
        {
            CellFlags *d = sg->getBlockDataPtr(bid, 1, 0);
            if(!d) continue;
            memset(d, 0, sg->nVoxelsInBlock() * sizeof(CellFlags));
        }
    };

    // Sparse levels: (level, levelMg) where level >= levelMg
    for(int lMg = 0; lMg < nLev; lMg++)
    {
        for(int level = lMg; level < nLev; level++)
        {
            auto *sgF = msbg->getFlagsChannel(CH_CELL_FLAGS, level, lMg);
            fillFlagsSG(sgF);
            for(int dir = 0; dir < 3; dir++)
                fillSG(msbg->getFaceAreaChannel(dir, level, lMg), 1.0f);
            fillSG(msbg->getFloatChannel(CH_DIAGONAL, level, lMg), 1.0f/6.0f);
        }
    }
    // V-cycle作業チャンネルをlevelMg=0で事前確保
    // （V-cycleの再帰はsetChannel/downsampleChannel経由で上位MGレベルのblockを確保する）
    {
        const int workChans[] = { CH_PRESSURE, CH_DIVERGENCE,
                                  CH_FLOAT_2, CH_FLOAT_3, CH_FLOAT_4,
                                  CH_FLOAT_6, CH_FLOAT_TMP_3 };
        for(int chan : workChans)
        {
            for(int level = 0; level < nLev; level++)
            {
                auto *sg = msbg->getFloatChannel(chan, level, 0);
                if(!sg) continue;
                sg->prepareDataAccess(SBG::ACC_READ | SBG::ACC_WRITE);
                for(int bid = 0; bid < sg->nBlocks(); bid++)
                    sg->getBlockDataPtr(bid, 1, 0);
            }
        }
    }

    // Divergence (MAC) と pressure を初期化
    zeroFloatChannel(sgDiv);
    zeroFloatChannel(sgP);

    for(int iz=0;iz<nz;iz++) for(int iy=0;iy<ny;iy++) for(int ix=0;ix<nx;ix++)
    {
        if(!isFluid(sgMass,ix,iy,iz)) continue;
        const float div =
            getVC(sgVel,ix+1,iy,  iz,  0)-getVC(sgVel,ix,iy,iz,0)+
            getVC(sgVel,ix,  iy+1,iz,  1)-getVC(sgVel,ix,iy,iz,1)+
            getVC(sgVel,ix,  iy,  iz+1,2)-getVC(sgVel,ix,iy,iz,2);
        setF(sgDiv,ix,iy,iz, div/dt);
    }

    // _blocksRelax リスト構築: relax() が使用するblock一覧
    // isRelaxationBlock() の条件: !(BLK_NO_FLUID|BLK_FIXED) && level<=levelMg
    for(int lMg = 0; lMg < nLev; lMg++)
    {
        msbg->_blocksRelax[lMg].clear();
        for(int bid = 0; bid < msbg->nBlocks(); bid++)
        {
            MSBG::BlockInfo *bi = msbg->getBlockInfo(bid, lMg);
            if(!bi) continue;
            if(msbg->isRelaxationBlock(lMg, bi))
                msbg->_blocksRelax[lMg].push_back(bid);
        }
    }
}

// MSBG V-cycle PCG solver: PCG外側 + V-cycle preconditioner
static void solvePressureMsbgPCG(
    MSBG::MultiresSparseGrid *msbg,
    float tol, int maxIter )
{
    using namespace MSBG;

    // チャンネル割当:
    //   CH_PRESSURE     = x (solution)
    //   CH_DIVERGENCE   = b (rhs, level 0のみ)
    //   CH_FLOAT_2      = r (residual)
    //   CH_FLOAT_3      = z (preconditioned residual)
    //   CH_FLOAT_4      = d (search direction)
    //   CH_FLOAT_6      = q (A*d)
    //   CH_FLOAT_TMP_PS = tmp (relax swap buffer)
    //   CH_CG_P, CH_CG_Q = V-cycle内部作業用（coarseレベル）

    const int chX   = CH_PRESSURE;
    const int chB   = CH_DIVERGENCE;
    const int chR   = CH_FLOAT_2;       // residual
    const int chZ   = CH_FLOAT_3;       // preconditioned residual
    const int chD   = CH_FLOAT_4;       // search direction
    const int chQ   = CH_FLOAT_6;       // q = A*d
    const int chTmp = CH_FLOAT_TMP_3;   // relax temp (別CH_FLOAT_2=chRとの衝突回避)

    // Note: prepare() は cell flags を上書きするため呼ばない
    // cell flags / face area / diagonal は buildMsbgPressureLevel0 で設定済み

    // x = 0 (already zero from buildMsbgPressureLevel0)

    // r = b - A*x = b (since x=0)
    msbg->copyChannel(chB, chR, 1, -1, 0);

    // Compute |b|^2
    long double bNormSq = 0.0;
    {
        SBG::SparseGrid<float> *sgR = msbg->getFloatChannel(chR, 0);
        for(int bid = 0; bid < sgR->nBlocks(); bid++)
        {
            float *d = sgR->getBlockDataPtr(bid);
            if(!d) continue;
            for(int i = 0; i < sgR->nVoxelsInBlock(); i++)
                bNormSq += (double)d[i] * (double)d[i];
        }
    }
    if(bNormSq <= 1e-30) return;

    // z = M^{-1} r  (1 V-cycle as preconditioner)
    msbg->setChannel(0.0, chZ, -1, 0);
    msbg->vCycle(0, chZ, chR, chQ, chTmp, 2, 2, 24);

    // d = z
    msbg->copyChannel(chZ, chD, 1, -1, 0);

    // rho = r . z
    long double rho = 0.0;
    {
        SBG::SparseGrid<float> *sgR = msbg->getFloatChannel(chR, 0);
        SBG::SparseGrid<float> *sgZ = msbg->getFloatChannel(chZ, 0);
        for(int bid = 0; bid < sgR->nBlocks(); bid++)
        {
            float *dr = sgR->getBlockDataPtr(bid);
            float *dz = sgZ->getBlockDataPtr(bid);
            if(!dr || !dz) continue;
            for(int i = 0; i < sgR->nVoxelsInBlock(); i++)
                rho += (double)dr[i] * (double)dz[i];
        }
    }

    const double tolSq = (double)tol * (double)tol;
    int convergedIter = -1;

    for(int iter = 0; iter < maxIter; iter++)
    {
        // q = A * d
        msbg->multiplyLaplacianMatrixOpt(
            0, 0, chD, CH_NULL, 0.0f, chQ, nullptr, nullptr);

        // alpha = rho / (d . q)
        long double dq = 0.0;
        {
            SBG::SparseGrid<float> *sgD = msbg->getFloatChannel(chD, 0);
            SBG::SparseGrid<float> *sgQ = msbg->getFloatChannel(chQ, 0);
            for(int bid = 0; bid < sgD->nBlocks(); bid++)
            {
                float *dd = sgD->getBlockDataPtr(bid);
                float *dq_ = sgQ->getBlockDataPtr(bid);
                if(!dd || !dq_) continue;
                for(int i = 0; i < sgD->nVoxelsInBlock(); i++)
                    dq += (double)dd[i] * (double)dq_[i];
            }
        }
        if(dq <= 1e-30 || rho <= 1e-30) break;

        const float alpha = (float)((double)rho / (double)dq);

        // x += alpha * d,  r -= alpha * q
        long double rNormSq = 0.0;
        {
            SBG::SparseGrid<float> *sgX = msbg->getFloatChannel(chX, 0);
            SBG::SparseGrid<float> *sgR = msbg->getFloatChannel(chR, 0);
            SBG::SparseGrid<float> *sgD = msbg->getFloatChannel(chD, 0);
            SBG::SparseGrid<float> *sgQ = msbg->getFloatChannel(chQ, 0);
            for(int bid = 0; bid < sgX->nBlocks(); bid++)
            {
                float *dx = sgX->getBlockDataPtr(bid);
                float *dr = sgR->getBlockDataPtr(bid);
                float *dd = sgD->getBlockDataPtr(bid);
                float *dq_ = sgQ->getBlockDataPtr(bid);
                if(!dx || !dr) continue;
                for(int i = 0; i < sgX->nVoxelsInBlock(); i++)
                {
                    if(dd) dx[i] += alpha * dd[i];
                    if(dq_) dr[i] -= alpha * dq_[i];
                    rNormSq += (double)dr[i] * (double)dr[i];
                }
            }
        }

        if(rNormSq <= tolSq * bNormSq)
        {
            convergedIter = iter + 1;
            break;
        }

        // z = M^{-1} r  (V-cycle preconditioner)
        msbg->setChannel(0.0, chZ, -1, 0);
        msbg->vCycle(0, chZ, chR, chQ, chTmp, 2, 2, 24);

        // rhoNew = r . z
        long double rhoNew = 0.0;
        {
            SBG::SparseGrid<float> *sgR = msbg->getFloatChannel(chR, 0);
            SBG::SparseGrid<float> *sgZ = msbg->getFloatChannel(chZ, 0);
            for(int bid = 0; bid < sgR->nBlocks(); bid++)
            {
                float *dr = sgR->getBlockDataPtr(bid);
                float *dz = sgZ->getBlockDataPtr(bid);
                if(!dr || !dz) continue;
                for(int i = 0; i < sgR->nVoxelsInBlock(); i++)
                    rhoNew += (double)dr[i] * (double)dz[i];
            }
        }
        if(rho <= 1e-30) break;

        const float beta = (float)((double)rhoNew / (double)rho);

        // d = z + beta * d
        {
            SBG::SparseGrid<float> *sgZ = msbg->getFloatChannel(chZ, 0);
            SBG::SparseGrid<float> *sgD = msbg->getFloatChannel(chD, 0);
            for(int bid = 0; bid < sgZ->nBlocks(); bid++)
            {
                float *dz = sgZ->getBlockDataPtr(bid);
                float *dd = sgD->getBlockDataPtr(bid);
                if(!dz || !dd) continue;
                for(int i = 0; i < sgZ->nVoxelsInBlock(); i++)
                    dd[i] = dz[i] + beta * dd[i];
            }
        }

        rho = rhoNew;
    }

    {
        long double finalRes = 0.0;
        SBG::SparseGrid<float> *sgR = msbg->getFloatChannel(chR, 0);
        for(int bid = 0; bid < sgR->nBlocks(); bid++)
        {
            float *dr = sgR->getBlockDataPtr(bid);
            if(!dr) continue;
            for(int i = 0; i < sgR->nVoxelsInBlock(); i++)
                finalRes += (double)dr[i] * (double)dr[i];
        }

        if(convergedIter > 0)
            { TRCP(("  MG-PCG converged in %d iter  |r|/|b|=%.2e\n",
                     convergedIter, sqrt((double)finalRes/(double)bNormSq))); }
        else
            { TRCP(("  MG-PCG: maxIter=%d reached  |r|/|b|=%.2e\n",
                     maxIter, sqrt((double)finalRes/(double)bNormSq))); }
    }
}
#endif  // --- END REMOVED Phase 2 legacy code ---

//=== 4. 圧力投影 ============================================================
static void pressureProjection( MSBG::MultiresSparseGrid *msbg,
                                SBG::SparseGrid<Vec3Float> *sgVel,
                                SBG::SparseGrid<float>     *sgMass,
                                SBG::SparseGrid<float>     *sgDiv,
                                SBG::SparseGrid<float>     *sgP,
                                SBG::SparseGrid<float>     *sgR,
                                SBG::SparseGrid<float>     *sgZ,
                                SBG::SparseGrid<float>     *sgD,
                                SBG::SparseGrid<float>     *sgQ,
                                SBG::SparseGrid<Vec3Float> *sgBeta,
                                float dt )
{
    (void)sgR; (void)sgZ; (void)sgD; (void)sgQ; // compact版では未使用
    const int nx=sgMass->sx(),ny=sgMass->sy(),nz=sgMass->sz();

    if(gSolverKind == PressureSolverKind::MSBG_VCYCLE_PCG)
    {
        // MSBG内部V-cycle PCG: callback経由でFLIPデータを注入
        FlipPressureCtx ctx = { sgVel, sgMass, sgBeta, dt, nx, ny, nz };
        MSBG::FlipPressureCallbacks cb;
        cb.sampleCellType  = &flipCellTypeCB;
        cb.sampleFaceCoeff = &flipFaceCoeffCB;
        cb.sampleRhs       = &flipRhsCB;
        cb.user = &ctx;

        msbg->preparePressureSolveFLIP(cb);

        MSBG::FlipPressureSolveParams sp;
        msbg->solvePressureFLIP(sp);
        // 結果は CH_PRESSURE に直接書かれている（sgP と同一）
    }
    else
    {
        zeroFloatChannel(sgDiv); zeroFloatChannel(sgP);

        // MAC 発散
        for(int iz=0;iz<nz;iz++) for(int iy=0;iy<ny;iy++) for(int ix=0;ix<nx;ix++)
        {
            if(!isFluid(sgMass,ix,iy,iz)) continue;
            const float div=
                getVC(sgVel,ix+1,iy,  iz,  0)-getVC(sgVel,ix,iy,iz,0)+
                getVC(sgVel,ix,  iy+1,iz,  1)-getVC(sgVel,ix,iy,iz,1)+
                getVC(sgVel,ix,  iy,  iz+1,2)-getVC(sgVel,ix,iy,iz,2);
            setF(sgDiv,ix,iy,iz,div/dt);
        }

        // compact 圧力系を構築
        buildPressureSystem(sgMass,sgBeta,sgDiv,dt,gPSys);

        // ソルバー選択
        if(gSolverKind==PressureSolverKind::HYPRE_AMG_PCG)
            solvePressureHypreAMG(gPSys,PCG_TOL,PCG_MAXITER);
        else
            solvePressureCompactPCG(gPSys,PCG_TOL,PCG_MAXITER);

        // 解をグリッドに書き戻す
        scatterPressureToGrid(gPSys,sgP);
    }

    // 圧力勾配補正 U (可変密度: u -= dt * beta * dp/dx)
    for(int iz=0;iz<nz;iz++) for(int iy=0;iy<ny;iy++) for(int ix=0;ix<=nx;ix++)
    {
        if(ix==0||ix==nx){setVC(sgVel,ix,iy,iz,0,0.f);continue;}
        bool fL=isFluid(sgMass,ix-1,iy,iz),fR=isFluid(sgMass,ix,iy,iz);
        if(!fL&&!fR){setVC(sgVel,ix,iy,iz,0,0.f);continue;}
        float pL=fL?getF(sgP,ix-1,iy,iz):0.f,pR=fR?getF(sgP,ix,iy,iz):0.f;
        float beta=getVC(sgBeta,ix,iy,iz,0);
        setVC(sgVel,ix,iy,iz,0,getVC(sgVel,ix,iy,iz,0)-dt*beta*(pR-pL));
    }
    // 圧力勾配補正 V
    for(int iz=0;iz<nz;iz++) for(int iy=0;iy<=ny;iy++) for(int ix=0;ix<nx;ix++)
    {
        if(iy==0){setVC(sgVel,ix,iy,iz,1,0.f);continue;}
        bool fB=(iy>0)&&isFluid(sgMass,ix,iy-1,iz),fT=(iy<ny)&&isFluid(sgMass,ix,iy,iz);
        if(!fB&&!fT){setVC(sgVel,ix,iy,iz,1,0.f);continue;}
        float pB=fB?getF(sgP,ix,iy-1,iz):0.f,pT=fT?getF(sgP,ix,iy,iz):0.f;
        float beta=getVC(sgBeta,ix,iy,iz,1);
        setVC(sgVel,ix,iy,iz,1,getVC(sgVel,ix,iy,iz,1)-dt*beta*(pT-pB));
    }
    // 圧力勾配補正 W
    for(int iz=0;iz<=nz;iz++) for(int iy=0;iy<ny;iy++) for(int ix=0;ix<nx;ix++)
    {
        if(iz==0||iz==nz){setVC(sgVel,ix,iy,iz,2,0.f);continue;}
        bool fN=isFluid(sgMass,ix,iy,iz-1),fF=isFluid(sgMass,ix,iy,iz);
        if(!fN&&!fF){setVC(sgVel,ix,iy,iz,2,0.f);continue;}
        float pN=fN?getF(sgP,ix,iy,iz-1):0.f,pF=fF?getF(sgP,ix,iy,iz):0.f;
        float beta=getVC(sgBeta,ix,iy,iz,2);
        setVC(sgVel,ix,iy,iz,2,getVC(sgVel,ix,iy,iz,2)-dt*beta*(pF-pN));
    }
}

//=== 5. G2P =================================================================
// MSBG level-aware MAC補間: 粒子位置からblockのlevelを考慮
// (Step 2: 現在は全block level=0で自前trilinearと同一結果)
static float interpCompMSBG( MSBG::MultiresSparseGrid *msbg,
                              SBG::SparseGrid<Vec3Float> *sg,
                              float px,float py,float pz,
                              float ox,float oy,float oz,int comp )
{
    // TODO: adaptive時はmsbg->getBlockInfo(bid)->levelに応じて
    //       適切なlevelのSparseGridから読む。現在はlevel=0固定。
    (void)msbg;
    const float gx=px-ox,gy=py-oy,gz=pz-oz;
    const int ix0=(int)floorf(gx),iy0=(int)floorf(gy),iz0=(int)floorf(gz);
    const float fx=gx-ix0,fy=gy-iy0,fz=gz-iz0;
    float sum=0.f;
    for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
        sum+=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz)
             *getVC(sg,ix0+dx,iy0+dy,iz0+dz,comp);
    return sum;
}

static void gridToParticle( MSBG::MultiresSparseGrid *msbg,
                            SBG::SparseGrid<Vec3Float> *sgVel,
                            SBG::SparseGrid<Vec3Float> *sgVelOld )
{
    const float alpha=FLIP_ALPHA;
    const size_t np=gParticles.size();
    tbb::parallel_for(tbb::blocked_range<size_t>(0,np),
        [&](const tbb::blocked_range<size_t> &range){
        for(size_t i=range.begin();i<range.end();i++)
        {
            FlipParticle &p=gParticles[i];
            float picU=interpCompMSBG(msbg,sgVel,   p.pos.x,p.pos.y,p.pos.z,0.f,0.5f,0.5f,0);
            float picV=interpCompMSBG(msbg,sgVel,   p.pos.x,p.pos.y,p.pos.z,0.5f,0.f,0.5f,1);
            float picW=interpCompMSBG(msbg,sgVel,   p.pos.x,p.pos.y,p.pos.z,0.5f,0.5f,0.f,2);
            float oldU=interpCompMSBG(msbg,sgVelOld,p.pos.x,p.pos.y,p.pos.z,0.f,0.5f,0.5f,0);
            float oldV=interpCompMSBG(msbg,sgVelOld,p.pos.x,p.pos.y,p.pos.z,0.5f,0.f,0.5f,1);
            float oldW=interpCompMSBG(msbg,sgVelOld,p.pos.x,p.pos.y,p.pos.z,0.5f,0.5f,0.f,2);
            p.vel.x=(1.f-alpha)*picU+alpha*(p.vel.x+(picU-oldU));
            p.vel.y=(1.f-alpha)*picV+alpha*(p.vel.y+(picV-oldV));
            p.vel.z=(1.f-alpha)*picW+alpha*(p.vel.z+(picW-oldW));
        }
    });
}

//=== 6. advect ==============================================================
static void advectParticles(int sx,int sy,int sz,float dt)
{
    const float eps=1e-4f,xMax=(float)sx-eps,yMax=(float)sy-eps,zMax=(float)sz-eps;
    const size_t np=gParticles.size();
    tbb::parallel_for(tbb::blocked_range<size_t>(0,np),
        [&](const tbb::blocked_range<size_t> &range){
        for(size_t i=range.begin();i<range.end();i++)
        {
            FlipParticle &p=gParticles[i];
            p.pos.x+=dt*p.vel.x; p.pos.y+=dt*p.vel.y; p.pos.z+=dt*p.vel.z;
            if(p.pos.x<0.f) {p.pos.x=0.f; p.vel.x=std::max(p.vel.x,0.f);}
            if(p.pos.x>xMax){p.pos.x=xMax;p.vel.x=std::min(p.vel.x,0.f);}
            if(p.pos.y<0.f) {p.pos.y=0.f; p.vel.y=std::max(p.vel.y,0.f);}
            if(p.pos.y>yMax){p.pos.y=yMax;}
            if(p.pos.z<0.f) {p.pos.z=0.f; p.vel.z=std::max(p.vel.z,0.f);}
            if(p.pos.z>zMax){p.pos.z=zMax;p.vel.z=std::min(p.vel.z,0.f);}
        }
    });
}

//=== entry point ============================================================
int flip_dam_break( int resolution, int blockSize, int nSteps )
{
    using namespace MSBG;
    TRCP(("=== FLIP Dam Break (PF-FLIP, 2-phase) ===\n"));
    TRCP(("resolution=%d blockSize=%d nSteps=%d dt=%.4f\n",resolution,blockSize,nSteps,DT));
    TRCP(("RHO_L=%.0f RHO_G=%.1f ratio=%.0f:1\n",RHO_L,RHO_G,RHO_L/RHO_G));

    const int sx=ALIGN(resolution,blockSize),sy=sx,sz=sx;

    // マルチ解像度対応: OPT_SINGLE_LEVEL を外し、3レベルで作成
    // (Step 1: 全ブロック level=0 で既存動作を維持)
    MultiresSparseGrid *msbg=MultiresSparseGrid::create(
        "FLIP_MAC",sx,sy,sz,blockSize,-1,0,-1,
        0,         // OPT_SINGLE_LEVEL を外す
        NULL,NULL,
        3          // 3 resolution levels
    );
    if(!msbg){TRCERR(("create() failed\n"));return 1;}

    TRCP(("MSBG: nLevels=%d nMgLevels=%d nBlocks=%d\n",
          msbg->getNumLevels(),msbg->getNumMgLevels(),msbg->nBlocks()));

    msbg->setDomainBoundarySpec_(DBC_SOLID,DBC_SOLID,DBC_SOLID,DBC_OPEN,DBC_SOLID,DBC_SOLID);

    // チャンネル準備ヘルパー
    auto prepareAllChannels=[&](){
        msbg->prepareDataAccess(CH_VEC3_4);
        msbg->prepareDataAccess(CH_VEC3_2);
        msbg->prepareDataAccess(CH_VEC3_3);
        msbg->prepareDataAccess(CH_FLOAT_1);
        msbg->prepareDataAccess(CH_FLOAT_2);
        msbg->prepareDataAccess(CH_FLOAT_3);
        msbg->prepareDataAccess(CH_FLOAT_4);
        msbg->prepareDataAccess(CH_FLOAT_6);
        msbg->prepareDataAccess(CH_DIVERGENCE);
        msbg->prepareDataAccess(CH_PRESSURE);
        // Step 6: V-cycle solver用追加チャンネル
        msbg->prepareDataAccess(CH_FLOAT_TMP_3); // relax temp
    };
    auto bindChannels=[&](
        SBG::SparseGrid<Vec3Float>*&sgVel, SBG::SparseGrid<Vec3Float>*&sgVelOld,
        SBG::SparseGrid<Vec3Float>*&sgBeta, SBG::SparseGrid<float>*&sgMass,
        SBG::SparseGrid<float>*&sgR, SBG::SparseGrid<float>*&sgZ,
        SBG::SparseGrid<float>*&sgD, SBG::SparseGrid<float>*&sgQ,
        SBG::SparseGrid<float>*&sgDiv, SBG::SparseGrid<float>*&sgP)->bool
    {
        sgVel   =msbg->getVecChannel(  CH_VEC3_4,      0);
        sgVelOld=msbg->getVecChannel(  CH_VEC3_2,      0);
        sgBeta  =msbg->getVecChannel(  CH_VEC3_3,      0);
        sgMass  =msbg->getFloatChannel(CH_FLOAT_1,     0);
        sgR     =msbg->getFloatChannel(CH_FLOAT_2,     0);
        sgZ     =msbg->getFloatChannel(CH_FLOAT_3,     0);
        sgD     =msbg->getFloatChannel(CH_FLOAT_4,     0);
        sgQ     =msbg->getFloatChannel(CH_FLOAT_6,     0);
        sgDiv   =msbg->getFloatChannel(CH_DIVERGENCE,  0);
        sgP     =msbg->getFloatChannel(CH_PRESSURE,    0);
        return sgVel&&sgVelOld&&sgBeta&&sgMass&&sgR&&sgZ&&sgD&&sgQ&&sgDiv&&sgP;
    };
    auto touchAllBlocks=[&](SBG::SparseGrid<Vec3Float>*sgVel,
        SBG::SparseGrid<Vec3Float>*sgVelOld, SBG::SparseGrid<Vec3Float>*sgBeta,
        SBG::SparseGrid<float>*sgMass, SBG::SparseGrid<float>*sgR,
        SBG::SparseGrid<float>*sgZ, SBG::SparseGrid<float>*sgD,
        SBG::SparseGrid<float>*sgQ, SBG::SparseGrid<float>*sgDiv,
        SBG::SparseGrid<float>*sgP)
    {
        for(int bid=0;bid<sgVel->nBlocks();bid++)
        { sgVel->getBlockDataPtr(bid,1,1); sgVelOld->getBlockDataPtr(bid,1,1);
          sgBeta->getBlockDataPtr(bid,1,1);
          sgMass->getBlockDataPtr(bid,1,1); sgR->getBlockDataPtr(bid,1,1);
          sgZ->getBlockDataPtr(bid,1,1); sgD->getBlockDataPtr(bid,1,1);
          sgQ->getBlockDataPtr(bid,1,1); sgDiv->getBlockDataPtr(bid,1,1);
          sgP->getBlockDataPtr(bid,1,1);
          // Step 6: relax temp
          SBG::SparseGrid<float> *sgTmp3=msbg->getFloatChannel(CH_FLOAT_TMP_3,0);
          if(sgTmp3) sgTmp3->getBlockDataPtr(bid,1,1); }
    };

    // 粒子初期化
    initParticles(sx,sy,sz);

    // §3.4: 初回 refinement map 構築
    std::vector<int> refinementMap;
    buildRefinementMapFromParticles(msbg,refinementMap);
    msbg->setRefinementMap(refinementMap.data(),NULL,-1,NULL,false,true);
    prepareAllChannels();

    SBG::SparseGrid<Vec3Float> *sgVel,*sgVelOld,*sgBeta;
    SBG::SparseGrid<float> *sgMass,*sgR,*sgZ,*sgD,*sgQ,*sgDiv,*sgP;
    if(!bindChannels(sgVel,sgVelOld,sgBeta,sgMass,sgR,sgZ,sgD,sgQ,sgDiv,sgP))
    { TRCERR(("Null channel pointer!\n")); MultiresSparseGrid::destroy(msbg); return 1; }
    touchAllBlocks(sgVel,sgVelOld,sgBeta,sgMass,sgR,sgZ,sgD,sgQ,sgDiv,sgP);

    TRCP(("Grid: %dx%dx%d blocks=%d blockSize=%d\n",sx,sy,sz,(int)sgVel->nBlocks(),blockSize));

    UtTimer tm, tm2;
    for(int step=0;step<nSteps;step++)
    {
        TIMER_START(&tm);

        // §3.4: refinement map を毎step更新
        {
            std::vector<int> newMap;
            buildRefinementMapFromParticles(msbg,newMap);
            if(newMap!=refinementMap)
            {
                refinementMap.swap(newMap);
                msbg->setRefinementMap(refinementMap.data(),NULL,-1,NULL,false,true);
                prepareAllChannels();
                if(!bindChannels(sgVel,sgVelOld,sgBeta,sgMass,sgR,sgZ,sgD,sgQ,sgDiv,sgP))
                { TRCERR(("Null channel after regrid\n")); break; }
                touchAllBlocks(sgVel,sgVelOld,sgBeta,sgMass,sgR,sgZ,sgD,sgQ,sgDiv,sgP);
            }
        }

        TIMER_START(&tm2);
        particleToGrid(    sgVel,sgMass,sgBeta);
        TIMER_STOP(&tm2); double t_p2g=TIMER_DIFF_MS(&tm2);
        TIMER_START(&tm2);
        applyGravity(      sgVel,sgVelOld,sgMass,DT);
        TIMER_STOP(&tm2); double t_grav=TIMER_DIFF_MS(&tm2);
        TIMER_START(&tm2);
        pressureProjection(msbg,sgVel,sgMass,sgDiv,sgP,sgR,sgZ,sgD,sgQ,sgBeta,DT);
        TIMER_STOP(&tm2); double t_press=TIMER_DIFF_MS(&tm2);
        TIMER_START(&tm2);
        gridToParticle(    msbg,sgVel,sgVelOld);
        TIMER_STOP(&tm2); double t_g2p=TIMER_DIFF_MS(&tm2);
        TIMER_START(&tm2);
        advectParticles(   sx,sy,sz,DT);
        TIMER_STOP(&tm2); double t_adv=TIMER_DIFF_MS(&tm2);
        TIMER_STOP(&tm);
        TRCP(("step %3d/%d  particles=%d  activeBlocks=%d  %.3f sec\n",
              step+1,nSteps,(int)gParticles.size(),(int)gActiveBlocks.size(),
              (double)TIMER_DIFF_MS(&tm)/1000.0));
        TRCP(("  P2G=%.1f Grav=%.1f Press=%.1f G2P=%.1f Adv=%.1f ms\n",
              t_p2g,t_grav,t_press,t_g2p,t_adv));
        saveParticleSlice(step,sx,sy,sz);
    }

    TRCP(("=== Done ===\n"));
    hypreCleanup();
    MultiresSparseGrid::destroy(msbg);
    gParticles.clear();
    return 0;
}
