/*
* This file is part of the VanitySearch distribution (https://github.com/JeanLucPons/VanitySearch).
* Copyright (c) 2019 Jean Luc PONS.
*
* This program is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, version 3.
*
* This program is distributed in the hope that it will be useful, but
* WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
* General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with this program. If not, see <http://www.gnu.org/licenses/>.
*/

// ---------------------------------------------------------------------------------
// 256(+64) bits integer CUDA libray for SECPK1
// ---------------------------------------------------------------------------------

// We need 1 extra block for ModInv
#define NBBLOCK 5
#define BIFULLSIZE 40

// Assembly directives
#define UADDO(c, a, b) asm volatile ("add.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADDC(c, a, b) asm volatile ("addc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define UADD(c, a, b) asm volatile ("addc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define UADDO1(c, a) asm volatile ("add.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADDC1(c, a) asm volatile ("addc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define UADD1(c, a) asm volatile ("addc.u64 %0, %0, %1;" : "+l"(c) : "l"(a));

#define USUBO(c, a, b) asm volatile ("sub.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUBC(c, a, b) asm volatile ("subc.cc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b) : "memory" );
#define USUB(c, a, b) asm volatile ("subc.u64 %0, %1, %2;" : "=l"(c) : "l"(a), "l"(b));

#define USUBO1(c, a) asm volatile ("sub.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUBC1(c, a) asm volatile ("subc.cc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) : "memory" );
#define USUB1(c, a) asm volatile ("subc.u64 %0, %0, %1;" : "+l"(c) : "l"(a) );

#define UMULLO(lo,a, b) asm volatile ("mul.lo.u64 %0, %1, %2;" : "=l"(lo) : "l"(a), "l"(b));
#define UMULHI(hi,a, b) asm volatile ("mul.hi.u64 %0, %1, %2;" : "=l"(hi) : "l"(a), "l"(b));
#define MADDO(r,a,b,c) asm volatile ("mad.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADDC(r,a,b,c) asm volatile ("madc.hi.cc.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c) : "memory" );
#define MADD(r,a,b,c) asm volatile ("madc.hi.u64 %0, %1, %2, %3;" : "=l"(r) : "l"(a), "l"(b), "l"(c));

// ---------------------------------------------------------------------------------------
// This Code Change by Alek76 02.10.2023 
// ---------------------------------------------------------------------------------------

// 64bits lsb negative inverse of P (mod 2^64)
#define MM64 0xD838091DD2253531ULL

// 64bits lsb negative inverse of O (mod 2^64)
//#define MM64o 0x4B0DFF665588B13FULL

// ---------------------------------------------------------------------------------------

#define _IsPositive(x) (((int64_t)(x[4]))>=0LL)
#define _IsNegative(x) (((int64_t)(x[4]))<0LL)
#define _IsEqual(a,b)  ((a[4] == b[4]) && (a[3] == b[3]) && (a[2] == b[2]) && (a[1] == b[1]) && (a[0] == b[0]))
#define _IsZero(a)     ((a[4] | a[3] | a[2] | a[1] | a[0]) == 0ULL)
#define _IsOne(a)      ((a[4] == 0ULL) && (a[3] == 0ULL) && (a[2] == 0ULL) && (a[1] == 0ULL) && (a[0] == 1ULL))

#define IDX threadIdx.x

// ---------------------------------------------------------------------------------------

#define Add2(r, a, b) {\
  UADDO(r[0], a[0], b[0]); \
  UADDC(r[1], a[1], b[1]); \
  UADDC(r[2], a[2], b[2]); \
  UADDC(r[3], a[3], b[3]); \
  UADD(r[4], a[4], b[4]);}

// ---------------------------------------------------------------------------------------

#define Add1(r,a) { \
  UADDO1(r[0], a[0]); \
  UADDC1(r[1], a[1]); \
  UADDC1(r[2], a[2]); \
  UADDC1(r[3], a[3]); \
  UADD1(r[4], a[4]);}

// ---------------------------------------------------------------------------------------

#define AddP(r) { \
  UADDO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  UADDC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  UADDC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  UADD1(r[4], 0ULL);}

// ---------------------------------------------------------------------------------------

#define SubP(r) { \
  USUBO1(r[0], 0xFFFFFFFEFFFFFC2FULL); \
  USUBC1(r[1], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[2], 0xFFFFFFFFFFFFFFFFULL); \
  USUBC1(r[3], 0xFFFFFFFFFFFFFFFFULL); \
  USUB1(r[4], 0ULL);}

// ---------------------------------------------------------------------------------------

#define Sub2(r,a,b)  {\
  USUBO(r[0], a[0], b[0]); \
  USUBC(r[1], a[1], b[1]); \
  USUBC(r[2], a[2], b[2]); \
  USUBC(r[3], a[3], b[3]); \
  USUB(r[4], a[4], b[4]);}

// ---------------------------------------------------------------------------------------

#define Sub1(r,a) {\
  USUBO1(r[0], a[0]); \
  USUBC1(r[1], a[1]); \
  USUBC1(r[2], a[2]); \
  USUBC1(r[3], a[3]); \
  USUB1(r[4], a[4]);}

// ---------------------------------------------------------------------------------------

#define Neg(r) {\
USUBO(r[0],0ULL,r[0]); \
USUBC(r[1],0ULL,r[1]); \
USUBC(r[2],0ULL,r[2]); \
USUBC(r[3],0ULL,r[3]); \
USUB(r[4],0ULL,r[4]); }

// ---------------------------------------------------------------------------------------

#define Mult2(r, a, b) {\
  UMULLO(r[0],a[0],b); \
  UMULLO(r[1],a[1],b); \
  MADDO(r[1], a[0],b,r[1]); \
  UMULLO(r[2],a[2], b); \
  MADDC(r[2], a[1], b, r[2]); \
  UMULLO(r[3],a[3], b); \
  MADDC(r[3], a[2], b, r[3]); \
  UMULLO(r[4],a[4], b); \
  MADD(r[4], a[3], b, r[4]);}

// ---------------------------------------------------------------------------------------

#define UMult(r, a, b) {\
  UMULLO(r[0],a[0],b); \
  UMULLO(r[1],a[1],b); \
  MADDO(r[1], a[0],b,r[1]); \
  UMULLO(r[2],a[2], b); \
  MADDC(r[2], a[1], b, r[2]); \
  UMULLO(r[3],a[3], b); \
  MADDC(r[3], a[2], b, r[3]); \
  MADD(r[4], a[3], b, 0ULL);}

// ---------------------------------------------------------------------------------------

#define Load(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3]; \
  (r)[4] = (a)[4];}

// ---------------------------------------------------------------------------------------

#define _LoadI64(r, a) {\
  (r)[0] = a; \
  (r)[1] = a>>63; \
  (r)[2] = (r)[1]; \
  (r)[3] = (r)[1]; \
  (r)[4] = (r)[1];}
// ---------------------------------------------------------------------------------------

#define Load256(r, a) {\
  (r)[0] = (a)[0]; \
  (r)[1] = (a)[1]; \
  (r)[2] = (a)[2]; \
  (r)[3] = (a)[3];}

// ---------------------------------------------------------------------------------------
/*
#define Load256A(r, a) {\
  (r)[0] = (a)[IDX]; \
  (r)[1] = (a)[IDX+blockDim.x]; \
  (r)[2] = (a)[IDX+2*blockDim.x]; \
  (r)[3] = (a)[IDX+3*blockDim.x];}
*/
// ---------------------------------------------------------------------------------------
/*
#define Store256A(r, a) {\
  (r)[IDX] = (a)[0]; \
  (r)[IDX+blockDim.x] = (a)[1]; \
  (r)[IDX+2*blockDim.x] = (a)[2]; \
  (r)[IDX+3*blockDim.x] = (a)[3];}
*/
// ---------------------------------------------------------------------------------------

__device__ void LoadKangaroos(uint64_t *a,uint64_t px[GPU_GRP_SIZE][4],uint64_t py[GPU_GRP_SIZE][4],uint64_t dist[GPU_GRP_SIZE][4]) {

  for(int g = 0; g<GPU_GRP_SIZE; g++) {
    
    uint64_t *x64 = (uint64_t *)px[g];
    uint64_t *y64 = (uint64_t *)py[g];
    uint64_t *d64 = (uint64_t *)dist[g];
    uint32_t stride = g * 12 * blockDim.x;

    x64[0] = (a)[IDX + 0 * blockDim.x + stride];
    x64[1] = (a)[IDX + 1 * blockDim.x + stride];
    x64[2] = (a)[IDX + 2 * blockDim.x + stride];
    x64[3] = (a)[IDX + 3 * blockDim.x + stride];

    y64[0] = (a)[IDX + 4 * blockDim.x + stride];
    y64[1] = (a)[IDX + 5 * blockDim.x + stride];
    y64[2] = (a)[IDX + 6 * blockDim.x + stride];
    y64[3] = (a)[IDX + 7 * blockDim.x + stride];

    d64[0] = (a)[IDX + 8 * blockDim.x + stride];
    d64[1] = (a)[IDX + 9 * blockDim.x + stride];
    d64[2] = (a)[IDX + 10 * blockDim.x + stride];
    d64[3] = (a)[IDX + 11 * blockDim.x + stride];

  }

}

// ---------------------------------------------------------------------------------------

__device__ void StoreKangaroos(uint64_t *a,uint64_t px[GPU_GRP_SIZE][4],uint64_t py[GPU_GRP_SIZE][4],uint64_t dist[GPU_GRP_SIZE][4]) {

  for(int g = 0; g < GPU_GRP_SIZE; g++) {
    uint64_t *x64 = (uint64_t *)px[g];
    uint64_t *y64 = (uint64_t *)py[g];
    uint64_t *d64 = (uint64_t *)dist[g];
    uint32_t stride = g * 12 * blockDim.x;
    (a)[IDX + 0 * blockDim.x + stride] = x64[0];
    (a)[IDX + 1 * blockDim.x + stride] = x64[1];
    (a)[IDX + 2 * blockDim.x + stride] = x64[2];
    (a)[IDX + 3 * blockDim.x + stride] = x64[3];
    (a)[IDX + 4 * blockDim.x + stride] = y64[0];
    (a)[IDX + 5 * blockDim.x + stride] = y64[1];
    (a)[IDX + 6 * blockDim.x + stride] = y64[2];
    (a)[IDX + 7 * blockDim.x + stride] = y64[3];
    (a)[IDX + 8 * blockDim.x + stride] = d64[0];
    (a)[IDX + 9 * blockDim.x + stride] = d64[1];
    (a)[IDX + 10 * blockDim.x + stride] = d64[2];
    (a)[IDX + 11 * blockDim.x + stride] = d64[3];
  }

}

// ---------------------------------------------------------------------------------------

__device__ void ShiftR62(uint64_t *r) {

  r[0] = (r[1] << 2) | (r[0] >> 62);
  r[1] = (r[2] << 2) | (r[1] >> 62);
  r[2] = (r[3] << 2) | (r[2] >> 62);
  r[3] = (r[4] << 2) | (r[3] >> 62);
  // With sign extent
  r[4] = (int64_t)(r[4]) >> 62;

}

// ---------------------------------------------------------------------------------------

__device__ void IMult(uint64_t* r,uint64_t* a,int64_t b) {

  uint64_t t[NBBLOCK];

  // Make b positive
  int64_t msk = b >> 63;
  int64_t nmsk = ~msk;
  b = ((-b) & msk) | (b & ~msk);
  USUBO(t[0],a[0] & nmsk,a[0] & msk);
  USUBC(t[1],a[1] & nmsk,a[1] & msk);
  USUBC(t[2],a[2] & nmsk,a[2] & msk);
  USUBC(t[3],a[3] & nmsk,a[3] & msk);
  USUB(t[4],a[4] & nmsk,a[4] & msk);
  Mult2(r,t,b)

}


// ---------------------------------------------------------------------------------------

__device__ void MulP(uint64_t *r, uint64_t a) {

  uint64_t ah;
  uint64_t al;

  UMULLO(al, a, 0x1000003D1ULL);
  UMULHI(ah, a, 0x1000003D1ULL);

  USUBO(r[0], 0ULL, al);
  USUBC(r[1], 0ULL, ah);
  USUBC(r[2], 0ULL, 0ULL);
  USUBC(r[3], 0ULL, 0ULL);
  USUB(r[4], a, 0ULL);

}

// ---------------------------------------------------------------------------------------

__device__ void ModNeg256(uint64_t* r,uint64_t* a) {

  uint64_t t[4];
  USUBO(t[0],0ULL,a[0]);
  USUBC(t[1],0ULL,a[1]);
  USUBC(t[2],0ULL,a[2]);
  USUBC(t[3],0ULL,a[3]);
  UADDO(r[0],t[0],0xFFFFFFFEFFFFFC2FULL);
  UADDC(r[1],t[1],0xFFFFFFFFFFFFFFFFULL);
  UADDC(r[2],t[2],0xFFFFFFFFFFFFFFFFULL);
  UADD(r[3],t[3],0xFFFFFFFFFFFFFFFFULL);

}

// ---------------------------------------------------------------------------------------

__device__ void ModNeg256(uint64_t* r) {

  uint64_t t[4];
  USUBO(t[0],0ULL,r[0]);
  USUBC(t[1],0ULL,r[1]);
  USUBC(t[2],0ULL,r[2]);
  USUBC(t[3],0ULL,r[3]);
  UADDO(r[0],t[0],0xFFFFFFFEFFFFFC2FULL);
  UADDC(r[1],t[1],0xFFFFFFFFFFFFFFFFULL);
  UADDC(r[2],t[2],0xFFFFFFFFFFFFFFFFULL);
  UADD(r[3],t[3],0xFFFFFFFFFFFFFFFFULL);

}

// ---------------------------------------------------------------------------------------

__device__ void ModSub256(uint64_t* r, uint64_t* a, uint64_t* b) {

  uint64_t t;
  uint64_t T[4];
  USUBO(r[0],a[0],b[0]);
  USUBC(r[1],a[1],b[1]);
  USUBC(r[2],a[2],b[2]);
  USUBC(r[3],a[3],b[3]);
  USUB(t,0ULL,0ULL);
  T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
  T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
  T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
  T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
  UADDO1(r[0],T[0]);
  UADDC1(r[1],T[1]);
  UADDC1(r[2],T[2]);
  UADD1(r[3],T[3]);

}

// ---------------------------------------------------------------------------------------

__device__ void ModSub256(uint64_t* r, uint64_t* b) {

  uint64_t t;
  uint64_t T[4];
  USUBO(r[0],r[0],b[0]);
  USUBC(r[1],r[1],b[1]);
  USUBC(r[2],r[2],b[2]);
  USUBC(r[3],r[3],b[3]);
  USUB(t,0ULL,0ULL);
  T[0] = 0xFFFFFFFEFFFFFC2FULL & t;
  T[1] = 0xFFFFFFFFFFFFFFFFULL & t;
  T[2] = 0xFFFFFFFFFFFFFFFFULL & t;
  T[3] = 0xFFFFFFFFFFFFFFFFULL & t;
  UADDO1(r[0],T[0]);
  UADDC1(r[1],T[1]);
  UADDC1(r[2],T[2]);
  UADD1(r[3],T[3]);

}

// ---------------------------------------------------------------------------------------
// This Code Change by Alek76 02.10.2023 
// ---------------------------------------------------------------------------------------
//__device__ __constant__ uint64_t _O[] = { 0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL };

// No use constant memory (slow memory).

__device__ void ModAdd256Order(uint64_t *r, uint64_t *b) {

  uint64_t t[5];
  uint64_t c;
  UADDO(t[0], r[0], b[0]);
  UADDC(t[1], r[1], b[1]);
  UADDC(t[2], r[2], b[2]);
  UADDC(t[3], r[3], b[3]);
  UADD(t[4], 0ULL, 0ULL);
  USUBO(r[0], t[0], 0xBFD25E8CD0364141ULL);//USUBO(r[0], t[0], _O[0]);
  USUBC(r[1], t[1], 0xBAAEDCE6AF48A03BULL);//USUBC(r[1], t[1], _O[1]);
  USUBC(r[2], t[2], 0xFFFFFFFFFFFFFFFEULL);//USUBC(r[2], t[2], _O[2]);
  USUBC(r[3], t[3], 0xFFFFFFFFFFFFFFFFULL);//USUBC(r[3], t[3], _O[3]);
  USUB(c, t[4], 0ULL);
  if((int64_t)c < 0) {
    Load256(r, t);
  }

}

// ---------------------------------------------------------------------------------------
#define SWAP_ADD(x,y) x+=y;y-=x;
#define SWAP_SUB(x,y) x-=y;y+=x;
#define SWAP_NEG(tmp,x,y) tmp = x; x = y; y = -tmp;
#define MSK62 0x3FFFFFFFFFFFFFFF

__device__ void _DivStep62(int64_t u0,int64_t v0,
  int64_t* eta,
  int64_t* uu,int64_t* uv,
  int64_t* vu,int64_t* vv) {


  // u' = (uu*u + uv*v) >> bitCount
  // v' = (vu*u + vv*v) >> bitCount

  int64_t  bitCount;

#if 0

  // Former divstep62
  // Do not use eta, u and v have an exponential decay in worst case 
  // but with low probability to reach this worst case complexity

  int64_t  nb0;
  bitCount = 62;

  while(true) {

    // zeros = log2(z & -z)
    int64_t z = v0 | (UINT64_MAX << bitCount);
    float f = (float)(z & -z);
    int zeros = (*(uint32_t*)(&f) >> 23) - 127;
    v0 >>= zeros;
    *uu <<= zeros;
    *uv <<= zeros;
    bitCount -= zeros;

    /*
    while(IS_EVEN(v0) && (bitCount > 0)) {

      bitCount--;
      v0 >>= 1;
      *uu <<= 1;
      *uv <<= 1;

    }
    */

    if(bitCount <= 0)
      break;

    nb0 = (v0 + u0) & 0x3;
    if(nb0 == 0) {
      SWAP_ADD(*vv,*uv);
      SWAP_ADD(*vu,*uu);
      SWAP_ADD(v0,u0);
    }
    else {
      SWAP_SUB(*vv,*uv);
      SWAP_SUB(*vu,*uu);
      SWAP_SUB(v0,u0);
    }

  }


#endif

#if 1

  int64_t x,y,z;
  bitCount = 62;

  // divstep62 var time implementation by Peter Dettman
  // (see https://github.com/bitcoin-core/secp256k1/pull/767)

  while(true) {

    // Use a sentinel bit to count zeros only up to bitCount
    z = v0 | (UINT64_MAX << bitCount);

#ifdef NOFASTCTZ
    int zeros = __ffsll(z) - 1;
#else
    // zeros = log2(z & -z) is faster than __ffsll()
    float f = (float)(z & -z);
    int zeros = (*(uint32_t*)(&f) >> 23) - 127;
#endif

    v0 >>= zeros;
    *uu <<= zeros;
    *uv <<= zeros;
    *eta -= zeros;
    bitCount -= zeros;

    if(bitCount <= 0)
      break;

    if(*eta < 0) {
      *eta = -*eta;
      SWAP_NEG(x,u0,v0);
      SWAP_NEG(y,*uu,*vu);
      SWAP_NEG(z,*uv,*vv);
    }

    v0 += u0;
    *vv += *uv;
    *vu += *uu;

  }

#endif


}

__device__ __noinline__ void _ModInv(uint64_t* R) {

  // Compute modular inverse of R mop P (using 320bits signed integer)
  // 0 < this < P  , P must be odd
  // Return 0 if no inverse

  int64_t  uu,uv,vu,vv;
  uint64_t r0,s0;
  int64_t  eta = -1;

  uint64_t u[NBBLOCK];
  uint64_t v[NBBLOCK];
  uint64_t r[NBBLOCK];
  uint64_t s[NBBLOCK];
  uint64_t t1[NBBLOCK];
  uint64_t t2[NBBLOCK];
  uint64_t t3[NBBLOCK];
  uint64_t t4[NBBLOCK];

  u[0] = 0xFFFFFFFEFFFFFC2F;
  u[1] = 0xFFFFFFFFFFFFFFFF;
  u[2] = 0xFFFFFFFFFFFFFFFF;
  u[3] = 0xFFFFFFFFFFFFFFFF;
  u[4] = 0;
  Load(v,R);

  // Delayed right shift 62bits
  // Do not maintain a matrix for r and s, the number of 
  // 'added P' can be easily calculated

  // Fist step (r,s)=(0,1) ----------------------------

  uu = 1; uv = 0;
  vu = 0; vv = 1;

  _DivStep62((int64_t)u[0],(int64_t)v[0],&eta,&uu,&uv,&vu,&vv);

  // Now update BigInt variables

  // u = (uu*u + uv*v)
  // v = (vu*u + vv*v)
  IMult(t1,u,uu);
  IMult(t2,v,uv);
  IMult(t3,u,vu);
  IMult(t4,v,vv);
  Add2(u,t1,t2);
  Add2(v,t3,t4);

  _LoadI64(t2,uv);
  _LoadI64(t4,vv);

  // Compute multiple of P to add to s and r to make them multiple of 2^62
  r0 = (t2[0] * MM64) & MSK62;
  s0 = (t4[0] * MM64) & MSK62;
  MulP(r,r0);
  Add1(r,t2);
  MulP(s,s0);
  Add1(s,t4);

  // Right shift all variables by 62bits
  ShiftR62(u);
  ShiftR62(v);
  ShiftR62(r);
  ShiftR62(s);

  // DivStep loop -------------------------------

  while(true) {

    uu = 1; uv = 0;
    vu = 0; vv = 1;

    _DivStep62((int64_t)u[0],(int64_t)v[0],&eta,&uu,&uv,&vu,&vv);

    // Now update BigInt variables

    // u = (uu*u + uv*v)
    // v = (vu*u + vv*v)
    IMult(t1,u,uu);
    IMult(t2,v,uv);
    IMult(t3,u,vu);
    IMult(t4,v,vv);
    Add2(u,t1,t2);
    Add2(v,t3,t4);

    // Right shift (u,v) by 62bits
    ShiftR62(u);
    ShiftR62(v);

    IMult(t1,r,uu);
    IMult(t2,s,uv);

    if(_IsZero(v)) {

      // Last step
      // s not needed
      r0 = ((t1[0] + t2[0]) * MM64) & MSK62;
      MulP(r,r0);
      Add1(r,t1);
      Add1(r,t2);
      ShiftR62(r);
      break;

    } else {

      // r = (uu*r + uv*s + r0*P)
      // s = (vu*r + vv*s + s0*P)

      IMult(t3,r,vu);
      IMult(t4,s,vv);

      // Compute multiple of P to add to s to make it multiple of 2^62
      r0 = ((t1[0] + t2[0]) * MM64) & MSK62;
      s0 = ((t3[0] + t4[0]) * MM64) & MSK62;
      MulP(r,r0);
      Add1(r,t1);
      Add1(r,t2);

      // s = (vu*r + vv*s + s0*P)
      MulP(s,s0);
      Add1(s,t3);
      Add1(s,t4);

      // Right shift (r,s) by 62bits
      ShiftR62(r);
      ShiftR62(s);

    }

  }

  // u ends with -1 or 1
  if(_IsNegative(u)) {
    Neg(u);
    Neg(r);
  }

  if(!_IsOne(u)) {
    // No inverse
    R[0] = 0ULL;
    R[1] = 0ULL;
    R[2] = 0ULL;
    R[3] = 0ULL;
    R[4] = 0ULL;
    return;
  }

  while(_IsNegative(r))
    AddP(r);
  while(!_IsNegative(r))
    SubP(r);
  AddP(r);

  Load(R,r);

  /*
  int64_t msk = (int64_t)(u[4]) >> 63;
  int64_t nmsk = ~msk;
  USUBO(r[0],r[0] & nmsk,r[0] & msk);
  USUBC(r[1],r[1] & nmsk,r[1] & msk);
  USUBC(r[2],r[2] & nmsk,r[2] & msk);
  USUBC(r[3],r[3] & nmsk,r[3] & msk);
  USUB(r[4],r[4] & nmsk,r[4] & msk);
  Add16P(r);
  // Reduce from 320 to 256
  uint64_t ah;
  uint64_t al;
  UMULLO(al,r[4],0x1000003D1ULL);
  UMULHI(ah,r[4],0x1000003D1ULL);
  UADDO(R[0],r[0],al);
  UADDC(R[1],r[1],ah);
  UADDC(R[2],r[2],0ULL);
  UADD(R[3],r[3],0ULL);
  */

}

// ---------------------------------------------------------------------------------------
// Compute a*b*(mod n)
// a and b must be lower than n
// ---------------------------------------------------------------------------------------

__device__ void _ModMult(uint64_t *r, uint64_t *a, uint64_t *b) {

  uint64_t r512[8];
  uint64_t t[NBBLOCK];
  uint64_t ah, al;

  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;

  // 256*256 multiplier
  UMult(r512, a, b[0]);
  UMult(t, a, b[1]);
  UADDO1(r512[1], t[0]);
  UADDC1(r512[2], t[1]);
  UADDC1(r512[3], t[2]);
  UADDC1(r512[4], t[3]);
  UADD1(r512[5], t[4]);
  UMult(t, a, b[2]);
  UADDO1(r512[2], t[0]);
  UADDC1(r512[3], t[1]);
  UADDC1(r512[4], t[2]);
  UADDC1(r512[5], t[3]);
  UADD1(r512[6], t[4]);
  UMult(t, a, b[3]);
  UADDO1(r512[3], t[0]);
  UADDC1(r512[4], t[1]);
  UADDC1(r512[5], t[2]);
  UADDC1(r512[6], t[3]);
  UADD1(r512[7], t[4]);

  // Reduce from 512 to 320
  UMult(t, (r512 + 4), 0x1000003D1ULL);
  UADDO1(r512[0], t[0]);
  UADDC1(r512[1], t[1]);
  UADDC1(r512[2], t[2]);
  UADDC1(r512[3], t[3]);

  // Reduce from 320 to 256
  UADD1(t[4], 0ULL);
  UMULLO(al, t[4], 0x1000003D1ULL);
  UMULHI(ah, t[4], 0x1000003D1ULL);
  UADDO(r[0], r512[0], al);
  UADDC(r[1], r512[1], ah);
  UADDC(r[2], r512[2], 0ULL);
  UADD(r[3], r512[3], 0ULL);

}


__device__ void _ModMult(uint64_t *r, uint64_t *a) {

  uint64_t r512[8];
  uint64_t t[NBBLOCK];
  uint64_t ah, al;
  r512[5] = 0;
  r512[6] = 0;
  r512[7] = 0;

  // 256*256 multiplier
  UMult(r512, a, r[0]);
  UMult(t, a, r[1]);
  UADDO1(r512[1], t[0]);
  UADDC1(r512[2], t[1]);
  UADDC1(r512[3], t[2]);
  UADDC1(r512[4], t[3]);
  UADD1(r512[5], t[4]);
  UMult(t, a, r[2]);
  UADDO1(r512[2], t[0]);
  UADDC1(r512[3], t[1]);
  UADDC1(r512[4], t[2]);
  UADDC1(r512[5], t[3]);
  UADD1(r512[6], t[4]);
  UMult(t, a, r[3]);
  UADDO1(r512[3], t[0]);
  UADDC1(r512[4], t[1]);
  UADDC1(r512[5], t[2]);
  UADDC1(r512[6], t[3]);
  UADD1(r512[7], t[4]);

  // Reduce from 512 to 320
  UMult(t, (r512 + 4), 0x1000003D1ULL);
  UADDO1(r512[0], t[0]);
  UADDC1(r512[1], t[1]);
  UADDC1(r512[2], t[2]);
  UADDC1(r512[3], t[3]);

  // Reduce from 320 to 256
  UADD1(t[4], 0ULL);
  UMULLO(al, t[4], 0x1000003D1ULL);
  UMULHI(ah, t[4], 0x1000003D1ULL);
  UADDO(r[0], r512[0], al);
  UADDC(r[1], r512[1], ah);
  UADDC(r[2], r512[2], 0ULL);
  UADD(r[3], r512[3], 0ULL);

}

__device__ void _ModSqr(uint64_t *rp, const uint64_t *up) {

  uint64_t r512[8];

  uint64_t u10, u11;

  uint64_t r0;
  uint64_t r1;
  uint64_t r3;
  uint64_t r4;

  uint64_t t1;
  uint64_t t2;


  //k=0
  UMULLO(r512[0], up[0], up[0]);
  UMULHI(r1, up[0], up[0]);

  //k=1
  UMULLO(r3, up[0], up[1]);
  UMULHI(r4, up[0], up[1]);
  UADDO1(r3, r3);
  UADDC1(r4, r4);
  UADD(t1, 0x0ULL, 0x0ULL);
  UADDO1(r3, r1);
  UADDC1(r4, 0x0ULL);
  UADD1(t1, 0x0ULL);
  r512[1] = r3;

  //k=2
  UMULLO(r0, up[0], up[2]);
  UMULHI(r1, up[0], up[2]);
  UADDO1(r0, r0);
  UADDC1(r1, r1);
  UADD(t2, 0x0ULL, 0x0ULL);
  UMULLO(u10, up[1], up[1]);
  UMULHI(u11, up[1], up[1]);
  UADDO1(r0, u10);
  UADDC1(r1, u11);
  UADD1(t2, 0x0ULL);
  UADDO1(r0, r4);
  UADDC1(r1, t1);
  UADD1(t2, 0x0ULL);

  r512[2] = r0;

  //k=3
  UMULLO(r3, up[0], up[3]);
  UMULHI(r4, up[0], up[3]);
  UMULLO(u10, up[1], up[2]);
  UMULHI(u11, up[1], up[2]);
  UADDO1(r3, u10);
  UADDC1(r4, u11);
  UADD(t1, 0x0ULL, 0x0ULL);
  t1 += t1;
  UADDO1(r3, r3);
  UADDC1(r4, r4);
  UADD1(t1, 0x0ULL);
  UADDO1(r3, r1);
  UADDC1(r4, t2);
  UADD1(t1, 0x0ULL);

  r512[3] = r3;

  //k=4
  UMULLO(r0, up[1], up[3]);
  UMULHI(r1, up[1], up[3]);
  UADDO1(r0, r0);
  UADDC1(r1, r1);
  UADD(t2, 0x0ULL, 0x0ULL);
  UMULLO(u10, up[2], up[2]);
  UMULHI(u11, up[2], up[2]);
  UADDO1(r0, u10);
  UADDC1(r1, u11);
  UADD1(t2, 0x0ULL);
  UADDO1(r0, r4);
  UADDC1(r1, t1);
  UADD1(t2, 0x0ULL);

  r512[4] = r0;

  //k=5
  UMULLO(r3, up[2], up[3]);
  UMULHI(r4, up[2], up[3]);
  UADDO1(r3, r3);
  UADDC1(r4, r4);
  UADD(t1, 0x0ULL, 0x0ULL);
  UADDO1(r3, r1);
  UADDC1(r4, t2);
  UADD1(t1, 0x0ULL);

  r512[5] = r3;

  //k=6
  UMULLO(r0, up[3], up[3]);
  UMULHI(r1, up[3], up[3]);
  UADDO1(r0, r4);
  UADD1(r1, t1);
  r512[6] = r0;

  //k=7
  r512[7] = r1;

#if 1

  // Reduce from 512 to 320
  UMULLO(r0, r512[4], 0x1000003D1ULL);
  UMULLO(r1, r512[5], 0x1000003D1ULL);
  MADDO(r1, r512[4], 0x1000003D1ULL, r1);
  UMULLO(t2, r512[6], 0x1000003D1ULL);
  MADDC(t2, r512[5], 0x1000003D1ULL, t2);
  UMULLO(r3, r512[7], 0x1000003D1ULL);
  MADDC(r3, r512[6], 0x1000003D1ULL, r3);
  MADD(r4, r512[7], 0x1000003D1ULL, 0ULL);

  UADDO1(r512[0], r0);
  UADDC1(r512[1], r1);
  UADDC1(r512[2], t2);
  UADDC1(r512[3], r3);

  // Reduce from 320 to 256
  UADD1(r4, 0ULL);
  UMULLO(u10, r4, 0x1000003D1ULL);
  UMULHI(u11, r4, 0x1000003D1ULL);
  UADDO(rp[0], r512[0], u10);
  UADDC(rp[1], r512[1], u11);
  UADDC(rp[2], r512[2], 0ULL);
  UADD(rp[3], r512[3], 0ULL);

#else

  uint64_t z1, z2, z3, z4, z5, z6, z7, z8;

  UMULLO(z3, r512[5], 0x1000003d1ULL);
  UMULHI(z4, r512[5], 0x1000003d1ULL);
  UMULLO(z5, r512[6], 0x1000003d1ULL);
  UMULHI(z6, r512[6], 0x1000003d1ULL);
  UMULLO(z7, r512[7], 0x1000003d1ULL);
  UMULHI(z8, r512[7], 0x1000003d1ULL);
  UMULLO(z1, r512[4], 0x1000003d1ULL);
  UMULHI(z2, r512[4], 0x1000003d1ULL);
  UADDO1(z1, r512[0]);
  UADD1(z2, 0x0ULL);


  UADDO1(z2, r512[1]);
  UADDC1(z4, r512[2]);
  UADDC1(z6, r512[3]);
  UADD1(z8, 0x0ULL);

  UADDO1(z3, z2);
  UADDC1(z5, z4);
  UADDC1(z7, z6);
  UADD1(z8, 0x0ULL);

  UMULLO(u10, z8, 0x1000003d1ULL);
  UMULHI(u11, z8, 0x1000003d1ULL);
  UADDO1(z1, u10);
  UADDC1(z3, u11);
  UADDC1(z5, 0x0ULL);
  UADD1(z7, 0x0ULL);

  rp[0] = z1;
  rp[1] = z3;
  rp[2] = z5;
  rp[3] = z7;

#endif

}

// ---------------------------------------------------------------------------------------
// Compute all ModInv of the group
// ---------------------------------------------------------------------------------------

//__device__ __noinline__ void _ModInvGrouped(uint64_t r[GPU_GRP_SIZE][4]) {
__device__ __forceinline__ void _ModInvGrouped(uint64_t r[GPU_GRP_SIZE][4]) {

  uint64_t subp[GPU_GRP_SIZE][4];
  uint64_t newValue[4];
  uint64_t inverse[5];

  Load256(subp[0], r[0]);
  for (uint32_t i = 1; i < GPU_GRP_SIZE; i++) {
    _ModMult(subp[i], subp[i - 1], r[i]);
  }

  // We need 320bit signed int for ModInv
  Load256(inverse, subp[GPU_GRP_SIZE - 1]);
  inverse[4] = 0;
  _ModInv(inverse);

  for (uint32_t i = GPU_GRP_SIZE - 1; i > 0; i--) {
    _ModMult(newValue, subp[i - 1], inverse);
    _ModMult(inverse, r[i]);
    Load256(r[i], newValue);
  }

  Load256(r[0], inverse);

}
// End...
