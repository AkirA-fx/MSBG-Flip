/******************************************************************************
 * flip_sim.cpp  -  Phase-Field FLIP: MAC格子 + 可変密度 MIC(0)-PCG
 *
 * Phase 0: FlipSimulation class implementation
 ******************************************************************************/
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <string.h>
#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <tbb/tbb.h>
#include "msbg.h"
#include "bitmap.h"
#include "flip_sim.h"

// hypre (AMG preconditioner)
#include "HYPRE.h"
#include "HYPRE_IJ_mv.h"
#include "HYPRE_parcsr_ls.h"
#include "HYPRE_krylov.h"

//=== HypreState definition ===================================================
static const int HYPRE_SETUP_INTERVAL = 8;

struct HypreState {
    bool initialized = false;
    HYPRE_IJMatrix  ijA  = NULL;
    HYPRE_IJVector  ijb  = NULL;
    HYPRE_IJVector  ijx  = NULL;
    HYPRE_Solver    amg  = NULL;
    HYPRE_Solver    pcg  = NULL;
    int  prevN = -1;
    int  prevPatternRevision = -1;
    int  setupCount = 0;
    int  stepsSinceSetup = HYPRE_SETUP_INTERVAL;

    ~HypreState() { cleanup(); }

    void cleanup()
    {
        if(pcg){ HYPRE_ParCSRPCGDestroy(pcg); pcg=NULL; }
        if(amg){ HYPRE_BoomerAMGDestroy(amg); amg=NULL; }
        if(ijA){ HYPRE_IJMatrixDestroy(ijA);  ijA=NULL; }
        if(ijb){ HYPRE_IJVectorDestroy(ijb);  ijb=NULL; }
        if(ijx){ HYPRE_IJVectorDestroy(ijx);  ijx=NULL; }
        prevN=-1; prevPatternRevision=-1;
        stepsSinceSetup=HYPRE_SETUP_INTERVAL;
    }
};

//=== Anonymous namespace: pure utility functions =============================
namespace {

inline bool finite3(float x, float y, float z)
{ return std::isfinite(x) && std::isfinite(y) && std::isfinite(z); }

void zeroVecChannel( SBG::SparseGrid<Vec3Float> *sg )
{
    for( int bid = 0; bid < sg->nBlocks(); bid++ )
    {
        Vec3Float *d = sg->getBlockDataPtr(bid);
        if(!d) continue;
        const int n = sg->nVoxelsInBlock();
        for(int i=0;i<n;i++) d[i]=Vec3Float(0,0,0);
    }
}

void zeroFloatChannel( SBG::SparseGrid<float> *sg )
{
    for( int bid = 0; bid < sg->nBlocks(); bid++ )
    {
        float *d = sg->getBlockDataPtr(bid);
        if(!d) continue;
        const int n = sg->nVoxelsInBlock();
        for(int i=0;i<n;i++) d[i]=0.f;
    }
}

bool getBidVidF( SBG::SparseGrid<float> *sg,
                 int ix,int iy,int iz, int &bid,int &vid )
{
    const int bl=sg->bsxLog2(),bm=sg->bsx()-1,nbx=sg->nbx(),nby=sg->nby();
    if(ix<0||iy<0||iz<0||ix>=sg->sx()||iy>=sg->sy()||iz>=sg->sz()) return false;
    bid=(ix>>bl)+(iy>>bl)*nbx+(iz>>bl)*nbx*nby;
    const int bsx=sg->bsx();
    vid=(ix&bm)+(iy&bm)*bsx+(iz&bm)*bsx*bsx;
    return true;
}

bool getBidVidV( SBG::SparseGrid<Vec3Float> *sg,
                 int ix,int iy,int iz, int &bid,int &vid )
{
    const int bl=sg->bsxLog2(),bm=sg->bsx()-1,nbx=sg->nbx(),nby=sg->nby();
    if(ix<0||iy<0||iz<0||ix>=sg->sx()||iy>=sg->sy()||iz>=sg->sz()) return false;
    bid=(ix>>bl)+(iy>>bl)*nbx+(iz>>bl)*nbx*nby;
    const int bsx=sg->bsx();
    vid=(ix&bm)+(iy&bm)*bsx+(iz&bm)*bsx*bsx;
    return true;
}

float getF( SBG::SparseGrid<float> *sg,int ix,int iy,int iz )
{
    int bid,vid;
    if(!getBidVidF(sg,ix,iy,iz,bid,vid)) return 0.f;
    float *d=sg->getBlockDataPtr(bid); return d?d[vid]:0.f;
}

void setF( SBG::SparseGrid<float> *sg,int ix,int iy,int iz,float v )
{
    int bid,vid;
    if(!getBidVidF(sg,ix,iy,iz,bid,vid)) return;
    float *d=sg->getBlockDataPtr(bid); if(d) d[vid]=v;
}

float getVC( SBG::SparseGrid<Vec3Float> *sg,int ix,int iy,int iz,int c )
{
    int bid,vid;
    if(!getBidVidV(sg,ix,iy,iz,bid,vid)) return 0.f;
    Vec3Float *d=sg->getBlockDataPtr(bid);
    if(!d) return 0.f;
    return c==0?d[vid].x:(c==1?d[vid].y:d[vid].z);
}

void setVC( SBG::SparseGrid<Vec3Float> *sg,int ix,int iy,int iz,int c,float v )
{
    int bid,vid;
    if(!getBidVidV(sg,ix,iy,iz,bid,vid)) return;
    Vec3Float *d=sg->getBlockDataPtr(bid); if(!d) return;
    if(c==0) d[vid].x=v; else if(c==1) d[vid].y=v; else d[vid].z=v;
}

bool isFluid( SBG::SparseGrid<float> *sgMass,int ix,int iy,int iz,float eps )
{ return getF(sgMass,ix,iy,iz)>eps; }

void copyVecChannel( SBG::SparseGrid<Vec3Float> *dst,
                     SBG::SparseGrid<Vec3Float> *src )
{
    for(int bid=0;bid<src->nBlocks();bid++)
    {
        Vec3Float *s=src->getBlockDataPtr(bid),*d=dst->getBlockDataPtr(bid);
        if(!s||!d) continue;
        const int n=src->nVoxelsInBlock(); for(int i=0;i<n;i++) d[i]=s[i];
    }
}

float interpComp( SBG::SparseGrid<Vec3Float> *sg,
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

// Valid-weight trilinear interpolation for advection RK2.
// Returns interpolated velocity at (px,py,pz) using only grid points
// that have block data allocated AND fluid support in sgMass.
// *validWeightOut receives the total valid interpolation weight (0..1).
float interpCompValidFine( SBG::SparseGrid<Vec3Float> *sgVel,
                           SBG::SparseGrid<float> *sgMass,
                           float px,float py,float pz,
                           float ox,float oy,float oz,int comp,
                           float massEps, float *validWeightOut )
{
    const float gx=px-ox,gy=py-oy,gz=pz-oz;
    const int ix0=(int)floorf(gx),iy0=(int)floorf(gy),iz0=(int)floorf(gz);
    const float fx=gx-ix0,fy=gy-iy0,fz=gz-iz0;

    float sum=0.f, wsum=0.f;
    for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
    {
        const float w=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz);
        const int ix=ix0+dx, iy=iy0+dy, iz=iz0+dz;
        // Check block data exists
        int bid,vid;
        if(!getBidVidV(sgVel,ix,iy,iz,bid,vid)) continue;
        Vec3Float *d=sgVel->getBlockDataPtr(bid);
        if(!d) continue;
        // Check fluid support: at least one adjacent cell has mass
        bool hasSupport = true;
        if(sgMass) {
            if(comp==0) hasSupport = isFluid(sgMass,ix-1,iy,iz,massEps)||isFluid(sgMass,ix,iy,iz,massEps);
            else if(comp==1) hasSupport = isFluid(sgMass,ix,iy-1,iz,massEps)||isFluid(sgMass,ix,iy,iz,massEps);
            else hasSupport = isFluid(sgMass,ix,iy,iz-1,massEps)||isFluid(sgMass,ix,iy,iz,massEps);
        }
        if(!hasSupport) continue;
        float val=(comp==0)?d[vid].x:(comp==1?d[vid].y:d[vid].z);
        sum  += w*val;
        wsum += w;
    }
    if(validWeightOut) *validWeightOut = wsum;
    return (wsum>1e-6f) ? (sum/wsum) : 0.f;
}

// CSR SpMV: q = A * x
void csrSpmv(const PressureSystem &A, const float *x, float *q)
{
    for(int i=0;i<A.n;i++)
    {
        float sum=0.f;
        for(int j=A.rowPtr[i];j<A.rowPtr[i+1];j++)
            sum+=A.val[j]*x[A.colInd[j]];
        q[i]=sum;
    }
}

double compactDot(const float *a, const float *b, int n)
{
    double s=0.0;
    for(int i=0;i<n;i++) s+=(double)a[i]*(double)b[i];
    return s;
}

//=== MSBG V-cycle callback context ==========================================
struct FlipPressureCtx {
    SBG::SparseGrid<Vec3Float> *sgVel;
    SBG::SparseGrid<float>     *sgMass;
    SBG::SparseGrid<Vec3Float> *sgBeta;
    float dt;
    float massEps;
    int nx, ny, nz;
};

MSBG::CellFlags flipCellTypeCB(void *user, int x, int y, int z)
{
    auto *ctx = (FlipPressureCtx*)user;
    if(x<0 || x>=ctx->nx) return MSBG::CELL_SOLID;
    if(z<0 || z>=ctx->nz) return MSBG::CELL_SOLID;
    if(y<0)               return MSBG::CELL_SOLID;
    if(y>=ctx->ny)        return MSBG::CELL_VOID;  // open top
    return isFluid(ctx->sgMass, x, y, z, ctx->massEps) ? 0 : MSBG::CELL_VOID;
}

float flipFaceCoeffCB(void *user, int dir, int x, int y, int z)
{
    auto *ctx = (FlipPressureCtx*)user;
    return getVC(ctx->sgBeta, x, y, z, dir);
}

float flipRhsCB(void *user, int x, int y, int z)
{
    auto *ctx = (FlipPressureCtx*)user;
    if(!isFluid(ctx->sgMass, x, y, z, ctx->massEps)) return 0.f;
    const float div =
        getVC(ctx->sgVel,x+1,y,  z,  0)-getVC(ctx->sgVel,x,y,z,0)+
        getVC(ctx->sgVel,x,  y+1,z,  1)-getVC(ctx->sgVel,x,y,z,1)+
        getVC(ctx->sgVel,x,  y,  z+1,2)-getVC(ctx->sgVel,x,y,z,2);
    return div / ctx->dt;
}

} // anonymous namespace

//=== FlipGridBundle ==========================================================

void FlipGridBundle::prepareChannels()
{
    using namespace MSBG;
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
    msbg->prepareDataAccess(CH_FLOAT_TMP_3);
}

bool FlipGridBundle::rebind()
{
    using namespace MSBG;
    vel      = msbg->getVecChannel(  CH_VEC3_4,      0);
    velOld   = msbg->getVecChannel(  CH_VEC3_2,      0);
    beta     = msbg->getVecChannel(  CH_VEC3_3,      0);
    mass     = msbg->getFloatChannel(CH_FLOAT_1,     0);
    r        = msbg->getFloatChannel(CH_FLOAT_2,     0);
    z        = msbg->getFloatChannel(CH_FLOAT_3,     0);
    d        = msbg->getFloatChannel(CH_FLOAT_4,     0);
    q        = msbg->getFloatChannel(CH_FLOAT_6,     0);
    div      = msbg->getFloatChannel(CH_DIVERGENCE,  0);
    pressure = msbg->getFloatChannel(CH_PRESSURE,    0);
    tmpRelax = msbg->getFloatChannel(CH_FLOAT_TMP_3, 0);
    return vel && velOld && beta && mass && r && z && d && q && div && pressure;
}

void FlipGridBundle::touchAllBlocks()
{
    for(int bid=0; bid<msbg->nBlocks(); bid++)
    {
        vel->getBlockDataPtr(bid,1,1);
        velOld->getBlockDataPtr(bid,1,1);
        beta->getBlockDataPtr(bid,1,1);
        mass->getBlockDataPtr(bid,1,1);
        r->getBlockDataPtr(bid,1,1);
        z->getBlockDataPtr(bid,1,1);
        d->getBlockDataPtr(bid,1,1);
        q->getBlockDataPtr(bid,1,1);
        div->getBlockDataPtr(bid,1,1);
        pressure->getBlockDataPtr(bid,1,1);
        if(tmpRelax) tmpRelax->getBlockDataPtr(bid,1,1);
    }
}

//=== FlipDebugOutput =========================================================

void FlipDebugOutput::saveParticleSlice(const FlipState& state, int step,
                                        int sx, int sy, int sz)
{
    const int ZOOM=4;
    BmpBitmap *B=BmpNewBitmap(sx*ZOOM,sy*ZOOM,BMP_RGB|BMP_CLEAR);
    if(!B){TRCERR(("BmpNewBitmap failed\n"));return;}
    std::vector<int> cntL(sx*sy,0), cntA(sx*sy,0);
    const float zCenter=sz*0.5f,zHalf=sz*0.15f;
    for(const FlipParticle &p : state.particles)
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

//=== FlipSimulation: constructor / destructor ================================

FlipSimulation::FlipSimulation(MSBG::MultiresSparseGrid& msbg, FlipConfig cfg)
    : cfg_(cfg), hypre_(std::make_unique<HypreState>())
{
    grid_.msbg = &msbg;
    auto *sg0 = msbg.sg0();
    grid_.sx = sg0->sx();
    grid_.sy = sg0->sy();
    grid_.sz = sg0->sz();
}

FlipSimulation::~FlipSimulation()
{
    // HypreState destructor handles cleanup via unique_ptr
}

//=== FlipSimulation: initialization ==========================================

void FlipSimulation::initializeParticles()
{
    const int sx=grid_.sx, sy=grid_.sy, sz=grid_.sz;
    const int damX=sx/2,damY=sy/2,nSub=2;
    const float step=1.f/nSub;
    state_.particles.clear();
    state_.particles.reserve(sx*sy*sz*cfg_.particlesPerVoxel);
    for(int iz=0;iz<sz;iz++) for(int iy=0;iy<sy;iy++) for(int ix=0;ix<sx;ix++)
    {
        const int phase=(ix<damX&&iy<damY)?0:1;
        for(int kz=0;kz<nSub;kz++) for(int ky=0;ky<nSub;ky++) for(int kx=0;kx<nSub;kx++)
        {
            FlipParticle p;
            p.pos=Vec3Float(ix+(kx+0.5f)*step,iy+(ky+0.5f)*step,iz+(kz+0.5f)*step);
            p.vel=Vec3Float(0,0,0);
            p.phase=phase;
            state_.particles.push_back(p);
        }
    }
    int nL=0,nA=0;
    for(const FlipParticle &p : state_.particles) { if(p.phase==0) nL++; else nA++; }
    TRCP(("initParticles: %d particles (liquid=%d air=%d) dam=%dx%dx%d\n",
          (int)state_.particles.size(),nL,nA,damX,damY,sz));
}

bool FlipSimulation::updateRefinementMap()
{
    auto *msbg = grid_.msbg;
    SBG::SparseGrid<Vec3Float> *sg0=msbg->sg0();
    const int nBlk=msbg->nBlocks();
    const int nLevels=msbg->getNumLevels();
    const int coarsest=nLevels-1;

    std::vector<int> newMap(nBlk,coarsest);

    // BFS: 液体ブロックからのblock距離を計算
    const int INF=0x7fffffff;
    std::vector<int> dist(nBlk,INF);
    std::queue<int> q;

    for(const FlipParticle &p : state_.particles)
    {
        if(p.phase!=0) continue;
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
        if(d<=1) newMap[bid]=0;
        else if(nLevels>=2 && d==2) newMap[bid]=1;
        else newMap[bid]=coarsest;
    }
    msbg->regularizeRefinementMap(newMap.data());

    int cnt[3]={0,0,0};
    for(int bid=0;bid<nBlk;bid++)
    { int lv=newMap[bid]; if(lv<3) cnt[lv]++; }
    TRCP(("  refinement: L0=%d L1=%d L2=%d / %d blocks\n",cnt[0],cnt[1],cnt[2],nBlk));

    bool changed = (newMap != state_.refinementMap);
    if(changed)
        state_.refinementMap.swap(newMap);
    return changed;
}

float FlipSimulation::computeDt() const
{
    auto *sgVel = grid_.vel;
    auto *msbg  = grid_.msbg;
    const int bl0=sgVel->bsxLog2();
    const int nbx0=sgVel->nbx(), nby0=sgVel->nby();
    const int sxG0=sgVel->sx(), syG0=sgVel->sy(), szG0=sgVel->sz();

    float maxInvDt=0.f;
    for(const FlipParticle &p : state_.particles)
    {
        int pix=std::max(0,std::min((int)floorf(p.pos.x),sxG0-1));
        int piy=std::max(0,std::min((int)floorf(p.pos.y),syG0-1));
        int piz=std::max(0,std::min((int)floorf(p.pos.z),szG0-1));
        int pbid=(pix>>bl0)+(piy>>bl0)*nbx0+(piz>>bl0)*nbx0*nby0;
        int plevel=msbg->getBlockLevel(pbid);
        float dx=(float)(1<<plevel);
        float speed=sqrtf(p.vel.x*p.vel.x+p.vel.y*p.vel.y+p.vel.z*p.vel.z);
        maxInvDt=std::max(maxInvDt, speed/dx);
    }

    const float dt=(maxInvDt>1e-6f)
        ? std::min(cfg_.dtMax, std::max(cfg_.dtMin, cfg_.cflNumber/maxInvDt))
        : cfg_.dtMax;
    return dt;
}

//=== FlipSimulation: P2G =====================================================

void FlipSimulation::particleToGrid()
{
    auto *sgVel  = grid_.vel;
    auto *sgMass = grid_.mass;
    auto *sgBeta = grid_.beta;
    const float RHO_L = cfg_.rhoL;
    const float RHO_G = cfg_.rhoG;
    const float MASS_EPS = cfg_.massEps;
    const float BETA_MIN = cfg_.betaMin;
    const int   PPV = cfg_.particlesPerVoxel;

    zeroVecChannel(sgVel); zeroFloatChannel(sgMass); zeroVecChannel(sgBeta);

    const int nvox=sgVel->nVoxelsInBlock();

    // ブロック単位のスレッドローカル散布バッファ（メモリ効率改善）
    struct P2GBlockBuf {
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
    const size_t estimatedActiveBlocks =
        !state_.activeBlocks.empty()
            ? state_.activeBlocks.size()
            : std::max<size_t>(1, (size_t)sgVel->nBlocks() / 4);

    struct P2GBuf {
        std::unordered_map<int, P2GBlockBuf> blocks;
        explicit P2GBuf(size_t reserveBlocks=0) {
            if(reserveBlocks>0) blocks.reserve(reserveBlocks);
        }
        P2GBlockBuf &getBlock(int bid, int nv) {
            auto it = blocks.find(bid);
            if(it != blocks.end()) return it->second;
            P2GBlockBuf buf; buf.init(nv);
            auto res = blocks.emplace(bid, std::move(buf));
            return res.first->second;
        }
    };

    tbb::enumerable_thread_specific<P2GBuf> tlsBuf(
        [&](){ return P2GBuf(estimatedActiveBlocks); });

    const size_t np=state_.particles.size();
    const int bl=sgVel->bsxLog2(), bm=sgVel->bsx()-1;
    const int nbx=sgVel->nbx(), nby=sgVel->nby();
    const int bsx=sgVel->bsx();
    const int sxG=sgVel->sx(), syG=sgVel->sy(), szG=sgVel->sz();
    const auto &particles = state_.particles;

    // 並列パーティクル散布
    tbb::parallel_for(tbb::blocked_range<size_t>(0,np),
        [&](const tbb::blocked_range<size_t> &range){
        P2GBuf &L=tlsBuf.local();

        auto bv=[&](int ix,int iy,int iz,int &bid,int &vid)->bool{
            if(ix<0||iy<0||iz<0||ix>=sxG||iy>=syG||iz>=szG) return false;
            bid=(ix>>bl)+(iy>>bl)*nbx+(iz>>bl)*nbx*nby;
            vid=(ix&bm)+(iy&bm)*bsx+(iz&bm)*bsx*bsx;
            return true;
        };

        for(size_t pi=range.begin();pi<range.end();pi++)
        {
            const FlipParticle &p=particles[pi];
            const float mp=(p.phase==0)?RHO_L:RHO_G;

            // cell mass
            { int bid,vid;
              if(bv((int)floorf(p.pos.x),(int)floorf(p.pos.y),(int)floorf(p.pos.z),bid,vid))
              { P2GBlockBuf &B=L.getBlock(bid,nvox);
                B.mass[vid]+=1.f;
                if(p.phase==0) B.liq[vid]++; } }

            // U face (0, 0.5, 0.5)
            { float gx=p.pos.x,gy=p.pos.y-0.5f,gz=p.pos.z-0.5f;
              int ix0=(int)floorf(gx),iy0=(int)floorf(gy),iz0=(int)floorf(gz);
              float fx=gx-ix0,fy=gy-iy0,fz=gz-iz0;
              for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
              { float w=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz);
                int bid,vid;
                if(bv(ix0+dx,iy0+dy,iz0+dz,bid,vid))
                { P2GBlockBuf &B=L.getBlock(bid,nvox);
                  B.vx[vid]+=p.vel.x*w; B.wu[vid]+=w; B.ru[vid]+=mp*w; } } }

            // V face (0.5, 0, 0.5)
            { float gx=p.pos.x-0.5f,gy=p.pos.y,gz=p.pos.z-0.5f;
              int ix0=(int)floorf(gx),iy0=(int)floorf(gy),iz0=(int)floorf(gz);
              float fx=gx-ix0,fy=gy-iy0,fz=gz-iz0;
              for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
              { float w=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz);
                int bid,vid;
                if(bv(ix0+dx,iy0+dy,iz0+dz,bid,vid))
                { P2GBlockBuf &B=L.getBlock(bid,nvox);
                  B.vy[vid]+=p.vel.y*w; B.wv[vid]+=w; B.rv[vid]+=mp*w; } } }

            // W face (0.5, 0.5, 0)
            { float gx=p.pos.x-0.5f,gy=p.pos.y-0.5f,gz=p.pos.z;
              int ix0=(int)floorf(gx),iy0=(int)floorf(gy),iz0=(int)floorf(gz);
              float fx=gx-ix0,fy=gy-iy0,fz=gz-iz0;
              for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
              { float w=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz);
                int bid,vid;
                if(bv(ix0+dx,iy0+dy,iz0+dz,bid,vid))
                { P2GBlockBuf &B=L.getBlock(bid,nvox);
                  B.vz[vid]+=p.vel.z*w; B.ww[vid]+=w; B.rw[vid]+=mp*w; } } }
        }
    });

    // スレッドバッファ合算（ブロック単位）
    struct P2GMergedBlock {
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
    std::unordered_map<int, P2GMergedBlock> merged;
    merged.reserve(estimatedActiveBlocks);
    for(auto &L:tlsBuf)
    {
        for(auto &kv : L.blocks)
        {
            const int bid=kv.first;
            const P2GBlockBuf &src=kv.second;
            P2GMergedBlock &dst=merged[bid];
            if(dst.vx.empty()) dst.init(nvox);
            for(int vid=0;vid<nvox;vid++)
            {
                dst.vx[vid]+=src.vx[vid]; dst.vy[vid]+=src.vy[vid]; dst.vz[vid]+=src.vz[vid];
                dst.mass[vid]+=src.mass[vid];
                dst.wu[vid]+=src.wu[vid]; dst.wv[vid]+=src.wv[vid]; dst.ww[vid]+=src.ww[vid];
                dst.ru[vid]+=src.ru[vid]; dst.rv[vid]+=src.rv[vid]; dst.rw[vid]+=src.rw[vid];
                dst.liq[vid]+=src.liq[vid];
            }
        }
    }

    // 速度・質量をグリッドに書き出し
    for(auto &kv : merged)
    {
        const int bid=kv.first;
        const P2GMergedBlock &B=kv.second;
        Vec3Float *vd=sgVel->getBlockDataPtr(bid);
        float     *md=sgMass->getBlockDataPtr(bid);
        if(!vd) continue;
        for(int vid=0;vid<nvox;vid++)
        {
            vd[vid].x+=B.vx[vid];
            vd[vid].y+=B.vy[vid];
            vd[vid].z+=B.vz[vid];
            if(md) md[vid]+=B.mass[vid];
        }
    }

    // phi 計算用パラメータ
    const float rhoTilde0=(float)PPV;
    const float etaPhi=logf(RHO_L/RHO_G);
    const float rhoTildeMin=etaPhi*RHO_G*rhoTilde0;
    const float phiDenom=cfg_.alphaPhi*rhoTilde0*RHO_L;

    // normalize velocity + compute beta + active blocks
    state_.activeBlocks.clear();
    for(int bid=0;bid<sgVel->nBlocks();bid++)
    {
        Vec3Float *vd=sgVel->getBlockDataPtr(bid);
        Vec3Float *bd=sgBeta->getBlockDataPtr(bid);
        float     *md=sgMass->getBlockDataPtr(bid);
        if(!vd) continue;
        bool hasFluid=false;

        auto mit=merged.find(bid);
        const bool hasMerged=(mit!=merged.end());

        const int nbxM=sgMass->nbx(),nbyM=sgMass->nby(),bs=sgMass->bsx();
        const int bxB=bid%nbxM, byB=(bid/nbxM)%nbyM, bzB=bid/(nbxM*nbyM);
        const int x0=bxB*bs, y0=byB*bs, z0=bzB*bs;

        for(int vid=0;vid<nvox;vid++)
        {
            const float wuV=hasMerged?mit->second.wu[vid]:0.f;
            const float wvV=hasMerged?mit->second.wv[vid]:0.f;
            const float wwV=hasMerged?mit->second.ww[vid]:0.f;
            if(wuV>MASS_EPS) vd[vid].x/=wuV; else vd[vid].x=0.f;
            if(wvV>MASS_EPS) vd[vid].y/=wvV; else vd[vid].y=0.f;
            if(wwV>MASS_EPS) vd[vid].z/=wwV; else vd[vid].z=0.f;
            if(md&&md[vid]>MASS_EPS) hasFluid=true;

            if(bd)
            {
                const int lx=vid%bs, ly=(vid/bs)%bs, lz=vid/(bs*bs);
                const int ix=x0+lx, iy=y0+ly, iz=z0+lz;

                auto phiFromRho=[&](float rt)->float{
                    if(rt<rhoTildeMin) return 0.f;
                    float v=sqrtf(std::max(rt-rhoTildeMin,0.f)/phiDenom);
                    return std::min(v,1.f);
                };
                auto betaFromPhi=[&](float phi)->float{
                    float rho=RHO_G+phi*(RHO_L-RHO_G);
                    return std::max(1.f/rho, BETA_MIN);
                };

                const float ruV=hasMerged?mit->second.ru[vid]:0.f;
                const float rvV=hasMerged?mit->second.rv[vid]:0.f;
                const float rwV=hasMerged?mit->second.rw[vid]:0.f;
                float phiU=phiFromRho(ruV);
                float phiV=phiFromRho(rvV);
                float phiW=phiFromRho(rwV);

                // セル相判定: 0=liquid, 1=air, -1=empty/外
                auto cellPh=[&](int cx,int cy,int cz)->int{
                    int b2,v2;
                    if(!getBidVidF(sgMass,cx,cy,cz,b2,v2)) return -1;
                    float *md2=sgMass->getBlockDataPtr(b2);
                    if(!md2||md2[v2]<=MASS_EPS) return -1;
                    auto mit2=merged.find(b2);
                    if(mit2==merged.end()) return -1;
                    return (mit2->second.liq[v2]*2>(int)(md2[v2]+0.5f))?0:1;
                };

                // 界面クランプ (論文§3.3)
                { int pL=cellPh(ix-1,iy,iz), pR=cellPh(ix,iy,iz);
                  if(pL==0&&pR==0) phiU=1.f;
                  else if(pL==1&&pR==1) phiU=0.f; }
                { int pB=cellPh(ix,iy-1,iz), pT=cellPh(ix,iy,iz);
                  if(pB==0&&pT==0) phiV=1.f;
                  else if(pB==1&&pT==1) phiV=0.f; }
                { int pN=cellPh(ix,iy,iz-1), pF=cellPh(ix,iy,iz);
                  if(pN==0&&pF==0) phiW=1.f;
                  else if(pN==1&&pF==1) phiW=0.f; }

                bd[vid].x=betaFromPhi(phiU);
                bd[vid].y=betaFromPhi(phiV);
                bd[vid].z=betaFromPhi(phiW);
            }
        }
        if(hasFluid) state_.activeBlocks.push_back(bid);
    }
}

//=== FlipSimulation: gravity =================================================

void FlipSimulation::applyGravity(float dt)
{
    auto *sgVel    = grid_.vel;
    auto *sgVelOld = grid_.velOld;
    auto *sgMass   = grid_.mass;
    const float MASS_EPS = cfg_.massEps;

    copyVecChannel(sgVelOld,sgVel);
    const int nx=sgMass->sx(),ny=sgMass->sy(),nz=sgMass->sz();
    for(int iz=0;iz<nz;iz++) for(int iy=0;iy<=ny;iy++) for(int ix=0;ix<nx;ix++)
    {
        const bool fLo=(iy>0 )&&isFluid(sgMass,ix,iy-1,iz,MASS_EPS);
        const bool fHi=(iy<ny)&&isFluid(sgMass,ix,iy,  iz,MASS_EPS);
        if(fLo||fHi) setVC(sgVel,ix,iy,iz,1,getVC(sgVel,ix,iy,iz,1)+cfg_.gravity*dt);
    }
    // solid BC
    for(int iy=0;iy<ny;iy++) for(int iz=0;iz<nz;iz++)
    { setVC(sgVel,0,iy,iz,0,0.f); setVC(sgVel,nx-1,iy,iz,0,0.f); }
    for(int ix=0;ix<nx;ix++) for(int iz=0;iz<nz;iz++)
        setVC(sgVel,ix,0,iz,1,0.f);
    for(int ix=0;ix<nx;ix++) for(int iy=0;iy<ny;iy++)
    { setVC(sgVel,ix,iy,0,2,0.f); setVC(sgVel,ix,iy,nz-1,2,0.f); }
}

//=== FlipSimulation: pressure solver internals ===============================

void FlipSimulation::rebuildPressurePattern()
{
    PressureSystem &sys = pressure_;
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

void FlipSimulation::updatePressureNumerics()
{
    PressureSystem &sys = pressure_;
    auto *sgBeta = grid_.beta;
    auto *sgDiv  = grid_.div;
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

            // Negated because compact matrix uses -L (positive definite),
            // while sgDiv stores +div/dt.  MSBG path computes RHS separately
            // via flipRhsCB with matching +L convention.
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

void FlipSimulation::buildPressureSystem(float dt)
{
    PressureSystem &sys = pressure_;
    auto *sgMass = grid_.mass;
    auto *sgDiv  = grid_.div;
    const float MASS_EPS = cfg_.massEps;
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
        if(!isFluid(sgMass,ix,iy,iz,MASS_EPS)) continue;
        int row=(int)newRowToCell.size();
        newCellToRow[lin(ix,iy,iz)]=row;
        newRowToCell.push_back(lin(ix,iy,iz));
        newIx.push_back((short)ix); newIy.push_back((short)iy); newIz.push_back((short)iz);
    }

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
        rebuildPressurePattern();
    }

    // compute divergence for compact solver path
    zeroFloatChannel(sgDiv);
    for(int iz=0;iz<nz;iz++) for(int iy=0;iy<ny;iy++) for(int ix=0;ix<nx;ix++)
    {
        if(!isFluid(sgMass,ix,iy,iz,MASS_EPS)) continue;
        const float div=
            getVC(grid_.vel,ix+1,iy,  iz,  0)-getVC(grid_.vel,ix,iy,iz,0)+
            getVC(grid_.vel,ix,  iy+1,iz,  1)-getVC(grid_.vel,ix,iy,iz,1)+
            getVC(grid_.vel,ix,  iy,  iz+1,2)-getVC(grid_.vel,ix,iy,iz,2);
        setF(sgDiv,ix,iy,iz,div/dt);
    }

    updatePressureNumerics();

    TRCP(("  PressureSystem: n=%d nnz=%d (%.1f/row) topo=%s\n",
          sys.n,(int)sys.val.size(),(float)sys.val.size()/std::max(sys.n,1),
          topoSame?"reuse":"rebuild"));
}

void FlipSimulation::scatterPressureToGrid()
{
    PressureSystem &sys = pressure_;
    auto *sgP = grid_.pressure;
    zeroFloatChannel(sgP);
    for(int row=0;row<sys.n;row++)
    {
        int ix=sys.rowIx[row], iy=sys.rowIy[row], iz=sys.rowIz[row];
        setF(sgP,ix,iy,iz,sys.sol[row]);
    }
}

//=== MIC(0) preconditioner (compact) ========================================

void FlipSimulation::buildMIC0Compact(std::vector<float> &precon)
{
    PressureSystem &sys = pressure_;
    precon.assign(sys.n,0.f);
    const float sigma=0.25f;
    const int nx=sys.nx, ny=sys.ny;
    auto lin=[=](int ix,int iy,int iz){ return ix+nx*(iy+ny*iz); };

    for(int row=0;row<sys.n;row++)
    {
        float Adiag=0.f;
        for(int j=sys.rowPtr[row];j<sys.rowPtr[row+1];j++)
            if(sys.colInd[j]==row){ Adiag=sys.val[j]; break; }

        float e=Adiag;
        int ix=sys.rowIx[row], iy=sys.rowIy[row], iz=sys.rowIz[row];

        if(ix>0){
            int r2=sys.cellToRow[lin(ix-1,iy,iz)];
            if(r2>=0){ float bx=sys.betaXm[row];
                       float p=precon[r2]; e-=(bx*p)*(bx*p); } }
        if(iy>0){
            int r2=sys.cellToRow[lin(ix,iy-1,iz)];
            if(r2>=0){ float by=sys.betaYm[row];
                       float p=precon[r2]; e-=(by*p)*(by*p); } }
        if(iz>0){
            int r2=sys.cellToRow[lin(ix,iy,iz-1)];
            if(r2>=0){ float bz=sys.betaZm[row];
                       float p=precon[r2]; e-=(bz*p)*(bz*p); } }

        precon[row]=1.f/sqrtf(std::max(e,sigma*std::max(Adiag,1e-10f)));
    }
}

void FlipSimulation::applyMIC0Compact(const std::vector<float> &precon,
                                       const float *r, float *z)
{
    PressureSystem &sys = pressure_;
    const int n=sys.n, nx=sys.nx, ny=sys.ny;
    auto lin=[=](int ix,int iy,int iz){ return ix+nx*(iy+ny*iz); };
    std::vector<float> q(n);

    // 前進代入
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

    // 後退代入
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

//=== Compact PCG solver =====================================================

void FlipSimulation::solvePressureCompactPCG()
{
    PressureSystem &sys = pressure_;
    const int n=sys.n;
    if(n==0) return;

    std::vector<float> &x=sys.sol;
    std::fill(x.begin(),x.end(),0.f);
    std::vector<float> r(n), z(n), d(n), q(n);
    std::vector<float> precon;

    double rhsNormSq=0.0;
    for(int i=0;i<n;i++){ r[i]=sys.rhs[i]; rhsNormSq+=(double)r[i]*(double)r[i]; }
    if(rhsNormSq<=1e-30) return;

    buildMIC0Compact(precon);
    applyMIC0Compact(precon,r.data(),z.data());

    double rho=compactDot(r.data(),z.data(),n);
    for(int i=0;i<n;i++) d[i]=z[i];

    const double tolSq=(double)cfg_.pcgTol*(double)cfg_.pcgTol;
    int convergedIter=-1;

    for(int iter=0;iter<cfg_.pcgMaxIter;iter++)
    {
        csrSpmv(sys,d.data(),q.data());
        const double dq=compactDot(d.data(),q.data(),n);
        if(dq<=1e-30||rho<=1e-30) break;

        const float alpha=(float)(rho/dq);
        for(int i=0;i<n;i++){ x[i]+=alpha*d[i]; r[i]-=alpha*q[i]; }

        const double rNormSq=compactDot(r.data(),r.data(),n);
        if(rNormSq<=tolSq*rhsNormSq){ convergedIter=iter+1; break; }

        applyMIC0Compact(precon,r.data(),z.data());
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
        { TRCP(("  MIC0-PCG: maxIter=%d reached  |r|/|b|=%.2e\n",cfg_.pcgMaxIter,sqrt(finalRes/rhsNormSq))); }
}

//=== Hypre AMG+PCG solver ===================================================

void FlipSimulation::solvePressureHypreAMG()
{
    PressureSystem &sys = pressure_;
    HypreState &hy = *hypre_;
    const int n = sys.n;
    if(n==0) return;

    if(!hy.initialized){
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if(!mpi_initialized){
            int argc_dummy = 0; char **argv_dummy = NULL;
            MPI_Init(&argc_dummy, &argv_dummy);
        }
        HYPRE_Init();
        hy.initialized = true;
    }

    MPI_Comm comm = MPI_COMM_WORLD;
    const bool needRebuild =
        (n != hy.prevN) || (sys.patternRevision != hy.prevPatternRevision);

    if(needRebuild)
    {
        hy.cleanup();
        hy.initialized = true; // cleanup resets state but HYPRE is still initialized
        const int ilower=0, iupper=n-1;

        HYPRE_IJMatrixCreate(comm, ilower, iupper, ilower, iupper, &hy.ijA);
        HYPRE_IJMatrixSetObjectType(hy.ijA, HYPRE_PARCSR);
        HYPRE_IJMatrixInitialize(hy.ijA);

        HYPRE_IJVectorCreate(comm, ilower, iupper, &hy.ijb);
        HYPRE_IJVectorCreate(comm, ilower, iupper, &hy.ijx);
        HYPRE_IJVectorSetObjectType(hy.ijb, HYPRE_PARCSR);
        HYPRE_IJVectorSetObjectType(hy.ijx, HYPRE_PARCSR);
        HYPRE_IJVectorInitialize(hy.ijb);
        HYPRE_IJVectorInitialize(hy.ijx);

        HYPRE_BoomerAMGCreate(&hy.amg);
        HYPRE_BoomerAMGSetMaxLevels(hy.amg, 25);
        HYPRE_BoomerAMGSetCoarsenType(hy.amg, 6);
        HYPRE_BoomerAMGSetRelaxType(hy.amg, 6);
        HYPRE_BoomerAMGSetNumSweeps(hy.amg, 1);
        HYPRE_BoomerAMGSetStrongThreshold(hy.amg, 0.25);
        HYPRE_BoomerAMGSetTol(hy.amg, 0.0);
        HYPRE_BoomerAMGSetMaxIter(hy.amg, 1);
        HYPRE_BoomerAMGSetPrintLevel(hy.amg, 0);

        HYPRE_ParCSRPCGCreate(comm, &hy.pcg);
        HYPRE_PCGSetMaxIter(hy.pcg, cfg_.pcgMaxIter);
        HYPRE_PCGSetTol(hy.pcg, (double)cfg_.pcgTol);
        HYPRE_PCGSetTwoNorm(hy.pcg, 1);
        HYPRE_PCGSetPrintLevel(hy.pcg, 0);
        HYPRE_PCGSetPrecond(hy.pcg,
                            (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSolve,
                            (HYPRE_PtrToSolverFcn)HYPRE_BoomerAMGSetup,
                            hy.amg);
        hy.prevN = n;
        hy.prevPatternRevision = sys.patternRevision;
    }

    // 行列値の更新
    for(int row=0; row<n; row++)
    {
        int ncols = sys.rowPtr[row+1] - sys.rowPtr[row];
        int *cols = &sys.colInd[sys.rowPtr[row]];
        std::vector<double> dvals(ncols);
        for(int j=0; j<ncols; j++)
            dvals[j] = (double)sys.val[sys.rowPtr[row]+j];
        HYPRE_IJMatrixSetValues(hy.ijA, 1, &ncols, &row, cols, dvals.data());
    }
    HYPRE_IJMatrixAssemble(hy.ijA);

    HYPRE_ParCSRMatrix parA;
    HYPRE_IJMatrixGetObject(hy.ijA, (void**)&parA);

    {
        std::vector<int> indices(n);
        std::vector<double> bvals(n), xvals(n, 0.0);
        for(int i=0; i<n; i++){ indices[i]=i; bvals[i]=(double)sys.rhs[i]; }
        HYPRE_IJVectorSetValues(hy.ijb, n, indices.data(), bvals.data());
        HYPRE_IJVectorSetValues(hy.ijx, n, indices.data(), xvals.data());
    }
    HYPRE_IJVectorAssemble(hy.ijb);
    HYPRE_IJVectorAssemble(hy.ijx);

    HYPRE_ParVector parb, parx;
    HYPRE_IJVectorGetObject(hy.ijb, (void**)&parb);
    HYPRE_IJVectorGetObject(hy.ijx, (void**)&parx);

    const bool needSetup =
        needRebuild || (hy.stepsSinceSetup >= HYPRE_SETUP_INTERVAL);
    if(needSetup){
        HYPRE_ParCSRPCGSetup(hy.pcg, parA, parb, parx);
        ++hy.setupCount;
        hy.stepsSinceSetup = 0;
    }
    HYPRE_ParCSRPCGSolve(hy.pcg, parA, parb, parx);
    ++hy.stepsSinceSetup;

    HYPRE_Int numIter;
    double finalRelRes;
    HYPRE_PCGGetNumIterations(hy.pcg, &numIter);
    HYPRE_PCGGetFinalRelativeResidualNorm(hy.pcg, &finalRelRes);

    {
        std::vector<int> indices(n);
        std::vector<double> xvals(n);
        for(int i=0; i<n; i++) indices[i]=i;
        HYPRE_IJVectorGetValues(hy.ijx, n, indices.data(), xvals.data());
        sys.sol.resize(n);
        for(int i=0; i<n; i++) sys.sol[i]=(float)xvals[i];
    }

    if(!needSetup && numIter>(HYPRE_Int)(0.8f*cfg_.pcgMaxIter))
        hy.stepsSinceSetup=HYPRE_SETUP_INTERVAL;

    { TRCP(("  AMG-PCG %d iter |r|/|b|=%.2e setup=%s\n",
            (int)numIter,finalRelRes,needSetup?"yes":"skip")); }
}

//=== FlipSimulation: pressure projection =====================================

void FlipSimulation::pressureProjection(float dt)
{
    auto *msbg   = grid_.msbg;
    auto *sgVel  = grid_.vel;
    auto *sgMass = grid_.mass;
    auto *sgP    = grid_.pressure;
    auto *sgBeta = grid_.beta;
    const float MASS_EPS = cfg_.massEps;
    const int nx=sgMass->sx(),ny=sgMass->sy(),nz=sgMass->sz();

    if(cfg_.solverKind == PressureSolverKind::MSBG_VCYCLE_PCG)
    {
        FlipPressureCtx ctx = { sgVel, sgMass, sgBeta, dt, MASS_EPS, nx, ny, nz };
        MSBG::FlipPressureCallbacks cb;
        cb.sampleCellType  = &flipCellTypeCB;
        cb.sampleFaceCoeff = &flipFaceCoeffCB;
        cb.sampleRhs       = &flipRhsCB;
        cb.user = &ctx;

        msbg->preparePressureSolveFLIP(cb);

        MSBG::FlipPressureSolveParams sp;
        sp.tol     = cfg_.pcgTol;
        sp.maxIter = cfg_.pcgMaxIter;
        msbg->solvePressureFLIP(sp);
    }
    else
    {
        buildPressureSystem(dt);

        if(cfg_.solverKind==PressureSolverKind::HYPRE_AMG_PCG)
            solvePressureHypreAMG();
        else
            solvePressureCompactPCG();

        scatterPressureToGrid();
    }

    // 圧力勾配補正 U
    for(int iz=0;iz<nz;iz++) for(int iy=0;iy<ny;iy++) for(int ix=0;ix<=nx;ix++)
    {
        if(ix==0||ix==nx){setVC(sgVel,ix,iy,iz,0,0.f);continue;}
        bool fL=isFluid(sgMass,ix-1,iy,iz,MASS_EPS),fR=isFluid(sgMass,ix,iy,iz,MASS_EPS);
        if(!fL&&!fR){setVC(sgVel,ix,iy,iz,0,0.f);continue;}
        float pL=fL?getF(sgP,ix-1,iy,iz):0.f,pR=fR?getF(sgP,ix,iy,iz):0.f;
        float beta=getVC(sgBeta,ix,iy,iz,0);
        { float v=getVC(sgVel,ix,iy,iz,0)-dt*beta*(pR-pL);
          setVC(sgVel,ix,iy,iz,0,std::isfinite(v)?v:0.f); }
    }
    // 圧力勾配補正 V
    for(int iz=0;iz<nz;iz++) for(int iy=0;iy<=ny;iy++) for(int ix=0;ix<nx;ix++)
    {
        if(iy==0){setVC(sgVel,ix,iy,iz,1,0.f);continue;}
        bool fB=(iy>0)&&isFluid(sgMass,ix,iy-1,iz,MASS_EPS),fT=(iy<ny)&&isFluid(sgMass,ix,iy,iz,MASS_EPS);
        if(!fB&&!fT){setVC(sgVel,ix,iy,iz,1,0.f);continue;}
        float pB=fB?getF(sgP,ix,iy-1,iz):0.f,pT=fT?getF(sgP,ix,iy,iz):0.f;
        float beta=getVC(sgBeta,ix,iy,iz,1);
        { float v=getVC(sgVel,ix,iy,iz,1)-dt*beta*(pT-pB);
          setVC(sgVel,ix,iy,iz,1,std::isfinite(v)?v:0.f); }
    }
    // 圧力勾配補正 W
    for(int iz=0;iz<=nz;iz++) for(int iy=0;iy<ny;iy++) for(int ix=0;ix<nx;ix++)
    {
        if(iz==0||iz==nz){setVC(sgVel,ix,iy,iz,2,0.f);continue;}
        bool fN=isFluid(sgMass,ix,iy,iz-1,MASS_EPS),fF=isFluid(sgMass,ix,iy,iz,MASS_EPS);
        if(!fN&&!fF){setVC(sgVel,ix,iy,iz,2,0.f);continue;}
        float pN=fN?getF(sgP,ix,iy,iz-1):0.f,pF=fF?getF(sgP,ix,iy,iz):0.f;
        float beta=getVC(sgBeta,ix,iy,iz,2);
        { float v=getVC(sgVel,ix,iy,iz,2)-dt*beta*(pF-pN);
          setVC(sgVel,ix,iy,iz,2,std::isfinite(v)?v:0.f); }
    }
}

//=== FlipSimulation: G2P =====================================================

Vec3Float FlipSimulation::interpVec3MSBG(SBG::SparseGrid<Vec3Float> *sg,
                                          float px,float py,float pz,
                                          int chanId) const
{
    auto *msbg = grid_.msbg;
    const int nx=sg->sx(), ny=sg->sy(), nz=sg->sz();

    struct SampleInfo {
        float gx,gy,gz; int ix0,iy0,iz0; float fx,fy,fz; int bidC;
    };
    auto makeSample=[&](float ox,float oy,float oz)->SampleInfo{
        SampleInfo s;
        s.gx=px-ox; s.gy=py-oy; s.gz=pz-oz;
        s.ix0=(int)floorf(s.gx); s.iy0=(int)floorf(s.gy); s.iz0=(int)floorf(s.gz);
        s.fx=s.gx-s.ix0; s.fy=s.gy-s.iy0; s.fz=s.gz-s.iz0;
        const int cx=std::max(0,std::min(s.ix0,nx-1));
        const int cy=std::max(0,std::min(s.iy0,ny-1));
        const int cz=std::max(0,std::min(s.iz0,nz-1));
        int bxC,byC,bzC;
        sg->getBlockCoords(cx,cy,cz,bxC,byC,bzC);
        s.bidC=sg->getBlockIndex(bxC,byC,bzC);
        return s;
    };

    const SampleInfo su=makeSample(0.f,0.5f,0.5f);
    const SampleInfo sv=makeSample(0.5f,0.f,0.5f);
    const SampleInfo sw=makeSample(0.5f,0.5f,0.f);

    // If staggered offsets land in different blocks, fall back to per-component
    if(su.bidC!=sv.bidC || su.bidC!=sw.bidC)
    {
        return Vec3Float(
            interpCompMSBG(sg,px,py,pz,0.f,0.5f,0.5f,0,chanId),
            interpCompMSBG(sg,px,py,pz,0.5f,0.f,0.5f,1,chanId),
            interpCompMSBG(sg,px,py,pz,0.5f,0.5f,0.f,2,chanId));
    }

    const int level=msbg->getBlockLevel(su.bidC);

    auto interpFine=[&](const SampleInfo &s,int comp)->float{
        float sum=0.f;
        for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
            sum+=(dx?s.fx:1.f-s.fx)*(dy?s.fy:1.f-s.fy)*(dz?s.fz:1.f-s.fz)
                 *getVC(sg,s.ix0+dx,s.iy0+dy,s.iz0+dz,comp);
        return sum;
    };

    if(level==0)
        return Vec3Float(interpFine(su,0),interpFine(sv,1),interpFine(sw,2));

    auto *sgLev=msbg->getVecChannel(chanId,level,0);
    if(!sgLev) sgLev=sg;
    auto *sgDist=msbg->getDistFineCoarseChannel(level);

    auto interpCoarse=[&](const SampleInfo &s,int comp)->float{
        const float scale=1.f/(float)(1<<level);
        const float cgx=s.gx*scale, cgy=s.gy*scale, cgz=s.gz*scale;
        const int cix0=(int)floorf(cgx), ciy0=(int)floorf(cgy), ciz0=(int)floorf(cgz);
        const float cfx=cgx-cix0, cfy=cgy-ciy0, cfz=cgz-ciz0;
        float sum=0.f;
        for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
        {
            int ix2=cix0+dx, iy2=ciy0+dy, iz2=ciz0+dz;
            int b2,v2; float val=0.f;
            if(getBidVidV(sgLev,ix2,iy2,iz2,b2,v2))
            { Vec3Float *d=sgLev->getBlockDataPtr(b2);
              if(d) val=(comp==0?d[v2].x:(comp==1?d[v2].y:d[v2].z)); }
            sum+=(dx?cfx:1.f-cfx)*(dy?cfy:1.f-cfy)*(dz?cfz:1.f-cfz)*val;
        }
        return sum;
    };

    // Coarse-fine blending using distFineCoarse (matches MSBG core approach)
    auto interpBlended=[&](const SampleInfo &s,int comp)->float{
        const float fineVal   = interpFine(s,comp);
        const float coarseVal = interpCoarse(s,comp);

        if(!sgDist) return coarseVal;

        const float scale=1.f/(float)(1<<level);
        float pos[3] = { s.gx*scale, s.gy*scale, s.gz*scale };
        const float dist = sgDist->interpolateToFloat<SBG::IP_LINEAR>(
            pos, SBG::OPT_IPBC_NEUMAN) / 1024.0f;

        const int stenWidth = 2;
        const float alphaFine = (float)MtLinstep(
            (stenWidth/2.f - 0.f) * MT_SQRT3F * 1.01f,
            0.99f * (msbg->dTransRes() + 2.f) / MT_SQRT3F,
            dist);

        return alphaFine*fineVal + (1.f-alphaFine)*coarseVal;
    };

    return Vec3Float(interpBlended(su,0),interpBlended(sv,1),interpBlended(sw,2));
}

float FlipSimulation::interpCompMSBG(SBG::SparseGrid<Vec3Float> *sg,
                                      float px,float py,float pz,
                                      float ox,float oy,float oz,int comp,
                                      int chanId) const
{
    auto *msbg = grid_.msbg;
    const float gx=px-ox,gy=py-oy,gz=pz-oz;
    const int ix0=(int)floorf(gx),iy0=(int)floorf(gy),iz0=(int)floorf(gz);
    const float fx=gx-ix0,fy=gy-iy0,fz=gz-iz0;

    const int nx=sg->sx(), ny=sg->sy(), nz=sg->sz();

    const int cx=std::max(0,std::min(ix0,nx-1));
    const int cy=std::max(0,std::min(iy0,ny-1));
    const int cz=std::max(0,std::min(iz0,nz-1));
    int bxC,byC,bzC;
    sg->getBlockCoords(cx,cy,cz,bxC,byC,bzC);
    const int bidC=sg->getBlockIndex(bxC,byC,bzC);
    const int level=msbg->getBlockLevel(bidC);

    if(level==0)
    {
        float sum=0.f;
        for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
            sum+=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz)
                 *getVC(sg,ix0+dx,iy0+dy,iz0+dz,comp);
        return sum;
    }

    auto *sgLev=msbg->getVecChannel(chanId,level,0);
    if(!sgLev) sgLev=sg;

    auto interpFineComp=[&]()->float{
        float s=0.f;
        for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
            s+=(dx?fx:1.f-fx)*(dy?fy:1.f-fy)*(dz?fz:1.f-fz)
               *getVC(sg,ix0+dx,iy0+dy,iz0+dz,comp);
        return s;
    };

    const float scale=1.f/(float)(1<<level);
    const float cgx=gx*scale, cgy=gy*scale, cgz=gz*scale;
    const int cix0=(int)floorf(cgx), ciy0=(int)floorf(cgy), ciz0=(int)floorf(cgz);
    const float cfx=cgx-cix0, cfy=cgy-ciy0, cfz=cgz-ciz0;

    float coarseVal=0.f;
    for(int dz=0;dz<=1;dz++) for(int dy=0;dy<=1;dy++) for(int dx=0;dx<=1;dx++)
    {
        int ix2=cix0+dx, iy2=ciy0+dy, iz2=ciz0+dz;
        int b2,v2;
        float val=0.f;
        if(getBidVidV(sgLev,ix2,iy2,iz2,b2,v2))
        {
            Vec3Float *d=sgLev->getBlockDataPtr(b2);
            if(d) val=(comp==0?d[v2].x:(comp==1?d[v2].y:d[v2].z));
        }
        coarseVal+=(dx?cfx:1.f-cfx)*(dy?cfy:1.f-cfy)*(dz?cfz:1.f-cfz)*val;
    }

    // Coarse-fine blending using distFineCoarse (matches MSBG core)
    auto *sgDist=msbg->getDistFineCoarseChannel(level);
    if(!sgDist) return coarseVal;

    float pos[3] = { cgx, cgy, cgz };
    const float dist = sgDist->interpolateToFloat<SBG::IP_LINEAR>(
        pos, SBG::OPT_IPBC_NEUMAN) / 1024.0f;

    const int stenWidth = 2;
    const float alphaFine = (float)MtLinstep(
        (stenWidth/2.f - 0.f) * MT_SQRT3F * 1.01f,
        0.99f * (msbg->dTransRes() + 2.f) / MT_SQRT3F,
        dist);

    return alphaFine*interpFineComp() + (1.f-alphaFine)*coarseVal;
}

void FlipSimulation::gridToParticle()
{
    auto *sgVel    = grid_.vel;
    auto *sgVelOld = grid_.velOld;
    const float alpha = cfg_.flipAlpha;
    const size_t np = state_.particles.size();

    tbb::parallel_for(tbb::blocked_range<size_t>(0,np),
        [&](const tbb::blocked_range<size_t> &range){
        for(size_t i=range.begin();i<range.end();i++)
        {
            FlipParticle &p=state_.particles[i];
            const Vec3Float pic = interpVec3MSBG(sgVel,    p.pos.x,p.pos.y,p.pos.z,MSBG::CH_VEC3_4);
            const Vec3Float old = interpVec3MSBG(sgVelOld, p.pos.x,p.pos.y,p.pos.z,MSBG::CH_VEC3_2);
            p.vel.x=(1.f-alpha)*pic.x + alpha*(p.vel.x + (pic.x-old.x));
            p.vel.y=(1.f-alpha)*pic.y + alpha*(p.vel.y + (pic.y-old.y));
            p.vel.z=(1.f-alpha)*pic.z + alpha*(p.vel.z + (pic.z-old.z));
        }
    });
}

//=== FlipSimulation: advection ===============================================

void FlipSimulation::advectParticles(float dt)
{
    auto *sgVel  = grid_.vel;
    auto *sgMass = grid_.mass;
    const float MASS_EPS = cfg_.massEps;
    const int sx=grid_.sx, sy=grid_.sy, sz=grid_.sz;
    const float eps=1e-4f,xMax=(float)sx-eps,yMax=(float)sy-eps,zMax=(float)sz-eps;
    const size_t np=state_.particles.size();

    tbb::parallel_for(tbb::blocked_range<size_t>(0,np),
        [&](const tbb::blocked_range<size_t> &range){
        for(size_t i=range.begin();i<range.end();i++)
        {
            FlipParticle &p=state_.particles[i];
            if(!finite3(p.pos.x,p.pos.y,p.pos.z)||!finite3(p.vel.x,p.vel.y,p.vel.z))
            { p.vel.x=p.vel.y=p.vel.z=0.f;
              p.pos.x=std::clamp(p.pos.x,0.f,xMax);
              p.pos.y=std::clamp(p.pos.y,0.f,yMax);
              p.pos.z=std::clamp(p.pos.z,0.f,zMax);
              if(!std::isfinite(p.pos.x))p.pos.x=0.5f;
              if(!std::isfinite(p.pos.y))p.pos.y=0.5f;
              if(!std::isfinite(p.pos.z))p.pos.z=0.5f; continue; }

            // RK2 Midpoint: k1 = particle velocity (already pressure-projected
            // via G2P), k2 = grid velocity at midpoint with valid-weight fallback.
            // k1 uses p.vel directly (avoids sparse grid empty-block issues).
            const Vec3Float k1 = p.vel;
            const Vec3Float mid(
                std::clamp(p.pos.x + 0.5f*dt*k1.x, 0.f, xMax),
                std::clamp(p.pos.y + 0.5f*dt*k1.y, 0.f, yMax),
                std::clamp(p.pos.z + 0.5f*dt*k1.z, 0.f, zMax));

            // Sample grid velocity at midpoint; fall back to k1 where no data
            float wu=0.f,wv=0.f,ww=0.f;
            Vec3Float k2(
                interpCompValidFine(sgVel,sgMass,mid.x,mid.y,mid.z,0.f,0.5f,0.5f,0,MASS_EPS,&wu),
                interpCompValidFine(sgVel,sgMass,mid.x,mid.y,mid.z,0.5f,0.f,0.5f,1,MASS_EPS,&wv),
                interpCompValidFine(sgVel,sgMass,mid.x,mid.y,mid.z,0.5f,0.5f,0.f,2,MASS_EPS,&ww));
            if(wu<0.01f) k2.x=k1.x;
            if(wv<0.01f) k2.y=k1.y;
            if(ww<0.01f) k2.z=k1.z;

            p.pos.x += dt*k2.x;
            p.pos.y += dt*k2.y;
            p.pos.z += dt*k2.z;

            // Boundary clamp + free-slip BC
            if(p.pos.x<0.f) {p.pos.x=0.f;  p.vel.x=std::max(p.vel.x,0.f);}
            if(p.pos.x>xMax){p.pos.x=xMax; p.vel.x=std::min(p.vel.x,0.f);}
            if(p.pos.y<0.f) {p.pos.y=0.f;  p.vel.y=std::max(p.vel.y,0.f);}
            if(p.pos.y>yMax){p.pos.y=yMax; p.vel.y=std::min(p.vel.y,0.f);}
            if(p.pos.z<0.f) {p.pos.z=0.f;  p.vel.z=std::max(p.vel.z,0.f);}
            if(p.pos.z>zMax){p.pos.z=zMax; p.vel.z=std::min(p.vel.z,0.f);}
        }
    });
}

//=== FlipSimulation: run =====================================================

int FlipSimulation::runDamBreak(int nSteps)
{
    using namespace MSBG;
    auto *msbg = grid_.msbg;
    const int sx=grid_.sx, sy=grid_.sy, sz=grid_.sz;

    TRCP(("=== FLIP Dam Break (PF-FLIP, 2-phase) ===\n"));
    TRCP(("resolution=%d blockSize=%d nSteps=%d dt_max=%.4f CFL=%.1f\n",
          sx,(int)msbg->sg0()->bsx(),nSteps,cfg_.dtMax,cfg_.cflNumber));
    TRCP(("RHO_L=%.0f RHO_G=%.1f ratio=%.0f:1\n",cfg_.rhoL,cfg_.rhoG,cfg_.rhoL/cfg_.rhoG));

    msbg->setDomainBoundarySpec_(DBC_SOLID,DBC_SOLID,DBC_SOLID,DBC_OPEN,DBC_SOLID,DBC_SOLID);

    TRCP(("MSBG: nLevels=%d nMgLevels=%d nBlocks=%d\n",
          msbg->getNumLevels(),msbg->getNumMgLevels(),msbg->nBlocks()));

    // 粒子初期化
    initializeParticles();

    // §3.4: 初回 refinement map 構築
    updateRefinementMap();
    msbg->setRefinementMap(state_.refinementMap.data(),NULL,-1,NULL,false,true);
    grid_.prepareChannels();

    if(!grid_.rebind())
    { TRCERR(("Null channel pointer!\n")); return 1; }
    grid_.touchAllBlocks();

    TRCP(("Grid: %dx%dx%d blocks=%d blockSize=%d\n",
          sx,sy,sz,(int)grid_.vel->nBlocks(),(int)msbg->sg0()->bsx()));

    UtTimer tm, tm2;
    for(int step=0;step<nSteps;step++)
    {
        TIMER_START(&tm);

        // §3.4: refinement map を毎step更新
        if(updateRefinementMap())
        {
            msbg->setRefinementMap(state_.refinementMap.data(),NULL,-1,NULL,false,true);
            grid_.prepareChannels();
            if(!grid_.rebind())
            { TRCERR(("Null channel after regrid\n")); break; }
            grid_.touchAllBlocks();
        }

        const float dt = computeDt();

        TIMER_START(&tm2);
        particleToGrid();
        TIMER_STOP(&tm2); double t_p2g=TIMER_DIFF_MS(&tm2);
        TIMER_START(&tm2);
        applyGravity(dt);
        TIMER_STOP(&tm2); double t_grav=TIMER_DIFF_MS(&tm2);
        TIMER_START(&tm2);
        pressureProjection(dt);
        TIMER_STOP(&tm2); double t_press=TIMER_DIFF_MS(&tm2);
        TIMER_START(&tm2);
        gridToParticle();
        TIMER_STOP(&tm2); double t_g2p=TIMER_DIFF_MS(&tm2);
        TIMER_START(&tm2);
        advectParticles(dt);
        TIMER_STOP(&tm2); double t_adv=TIMER_DIFF_MS(&tm2);
        TIMER_STOP(&tm);

        state_.step = step + 1;
        state_.time += dt;

        TRCP(("step %3d/%d  particles=%d  activeBlocks=%d  %.3f sec  dt=%.4f\n",
              step+1,nSteps,(int)state_.particles.size(),(int)state_.activeBlocks.size(),
              (double)TIMER_DIFF_MS(&tm)/1000.0, dt));
        TRCP(("  P2G=%.1f Grav=%.1f Press=%.1f G2P=%.1f Adv=%.1f ms\n",
              t_p2g,t_grav,t_press,t_g2p,t_adv));

        if(cfg_.enableDebugSlice)
            FlipDebugOutput::saveParticleSlice(state_,step,sx,sy,sz);
    }

    TRCP(("=== Done ===\n"));
    return 0;
}

int FlipSimulation::runStandaloneDamBreak(int resolution, int blockSize, int nSteps)
{
    using namespace MSBG;
    const int sx = ALIGN(resolution, blockSize), sy=sx, sz=sx;

    MultiresSparseGrid *msbg = MultiresSparseGrid::create(
        "FLIP_MAC",sx,sy,sz,blockSize,-1,0,-1,
        0, NULL, NULL, 3
    );
    if(!msbg){TRCERR(("create() failed\n"));return 1;}

    FlipSimulation sim(*msbg);
    int rc = sim.runDamBreak(nSteps);

    MultiresSparseGrid::destroy(msbg);
    return rc;
}

//=== Backward-compatible entry point =========================================

int flip_dam_break(int resolution, int blockSize, int nSteps)
{
    return FlipSimulation::runStandaloneDamBreak(resolution, blockSize, nSteps);
}
