/*
 * This code modified by Alek-76
 *
 *
*/

// CUDA Kernel main function
// Compute SecpK1 keys
// We use affine coordinates for elliptic curve point (ie Z=1)

// Jump distance
__device__ __constant__ uint64_t dS[NB_JUMP][4];
// jump points
__device__ __constant__ uint64_t Spx[NB_JUMP][4];
__device__ __constant__ uint64_t Spy[NB_JUMP][4];

// ---------------------------------------------------------------------------------------

#define SetPoint(x,d,idx) {\
out[pos*ITEM_SIZE32 + 1] = ((uint32_t *)x)[0]; \
out[pos*ITEM_SIZE32 + 2] = ((uint32_t *)x)[1]; \
out[pos*ITEM_SIZE32 + 3] = ((uint32_t *)x)[2]; \
out[pos*ITEM_SIZE32 + 4] = ((uint32_t *)x)[3]; \
out[pos*ITEM_SIZE32 + 5] = ((uint32_t *)x)[4]; \
out[pos*ITEM_SIZE32 + 6] = ((uint32_t *)x)[5]; \
out[pos*ITEM_SIZE32 + 7] = ((uint32_t *)x)[6]; \
out[pos*ITEM_SIZE32 + 8] = ((uint32_t *)x)[7]; \
out[pos*ITEM_SIZE32 + 9] = ((uint32_t *)d)[0]; \
out[pos*ITEM_SIZE32 + 10] = ((uint32_t *)d)[1]; \
out[pos*ITEM_SIZE32 + 11] = ((uint32_t *)d)[2]; \
out[pos*ITEM_SIZE32 + 12] = ((uint32_t *)d)[3]; \
out[pos*ITEM_SIZE32 + 13] = ((uint32_t *)d)[4]; \
out[pos*ITEM_SIZE32 + 14] = ((uint32_t *)d)[5]; \
out[pos*ITEM_SIZE32 + 15] = ((uint32_t *)d)[6]; \
out[pos*ITEM_SIZE32 + 16] = ((uint32_t *)d)[7]; \
out[pos*ITEM_SIZE32 + 17] = ((uint32_t *)idx)[0]; \
out[pos*ITEM_SIZE32 + 18] = ((uint32_t *)idx)[1]; \
}

// -----------------------------------------------------------------------------------------

__device__ void ComputeKeys(uint64_t *Tkangaroos, uint64_t DPmod, uint32_t hop_modulo, uint32_t maxFound, uint32_t *out) {

	// The new code from 30.10.2023
	uint64_t dx[GPU_GRP_SIZE][4];
	
	uint64_t px[GPU_GRP_SIZE][4];// points
	uint64_t py[GPU_GRP_SIZE][4];
		
	uint64_t tk[GPU_GRP_SIZE][4];// keys
	uint32_t pw1 = 0;
		
	uint64_t dy[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t _s[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t _p2[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	uint64_t rx[4] = { 0ULL,0ULL,0ULL,0ULL };
	uint64_t ry[4] = { 0ULL,0ULL,0ULL,0ULL };
	
	
	// Load starting key
	__syncthreads();	
	LoadKangaroos(Tkangaroos, px, py, tk);
		
	// Sp-table Spx[256], Spy[256]
	// Distance dS[256]
	// NB_SPIN 256
	uint32_t j;
	for (j = 0; j < NB_SPIN; j++) {
	
		__syncthreads();
		
		// Get dx
		for(uint32_t g = 0; g < GPU_GRP_SIZE; g++) {
			#ifdef USE_ORIGINAL_JUMP
			pw1 = (uint32_t)(px[g][0]) & hop_modulo;// Get mask hop_modulo -= 1 in GPUEngine.cu
			#else
			pw1 = (uint32_t)(px[g][0]) & (NB_JUMP - 1);
			#endif
			ModSub256(dx[g], px[g], Spx[pw1]);
		}
		
		
		_ModInvGrouped(dx);
		
		
		for(uint32_t g = 0; g < GPU_GRP_SIZE; g++) { 
			
			__syncthreads();
			
			// Get jump size
			#ifdef USE_ORIGINAL_JUMP
			pw1 = (uint32_t)(px[g][0]) & hop_modulo;// Get mask hop_modulo -= 1 in GPUEngine.cu
			#else
			pw1 = (uint32_t)(px[g][0]) & (NB_JUMP - 1);
			#endif
			
			// Add Hops Distance
			// Modulo Order
			//ModAdd256Order(tk[g], dS[pw1]);
			Add256(tk[g], dS[pw1]);// faster if without Order
			
			// Affine points addition
			
			ModSub256(dy, py[g], Spy[pw1]);
			_ModMult(_s, dy, dx[g]);
			_ModSqr(_p2, _s);
			
			ModSub256(rx, _p2, Spx[pw1]);
			ModSub256(rx, px[g]);
			
			ModSub256(ry, px[g], rx);
			_ModMult(ry, _s);
			ModSub256(ry, py[g]);
			
			Load256(px[g], rx);
			Load256(py[g], ry);
			
			
			//if (px[g][0] % DPmod == 0) {// I use % mod !!
			if ((px[g][0] & DPmod) == 0) {// edit GPUEngine.cu !! DPmodule - 1; 
				uint32_t pos = atomicAdd(out,1);
				if(pos < maxFound) {
					uint64_t kIdx = (uint64_t)IDX + (uint64_t)g*(uint64_t)blockDim.x + (uint64_t)blockIdx.x*((uint64_t)blockDim.x*GPU_GRP_SIZE);
					
					SetPoint(px[g], tk[g], &kIdx);
				}
			}
			
		}
	
	}
	
	// Update starting points
	__syncthreads();
	StoreKangaroos(Tkangaroos, px, py, tk);

}
// Hi ;)
