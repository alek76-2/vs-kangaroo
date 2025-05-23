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

#ifndef VANITYH
#define VANITYH

#include <string>
#include <vector>
#include "SECP256k1.h"
#include "GPU/GPUEngine.h"
#ifdef WIN64
#include <Windows.h>
#endif

#ifdef WIN64
#define LOCK(ghMutex) WaitForSingleObject(ghMutex, INFINITE);
#define UNLOCK(ghMutex) ReleaseMutex(ghMutex);
#else
#define LOCK(ghMutex) pthread_mutex_lock(&ghMutex);
#define UNLOCK(ghMutex) pthread_mutex_unlock(&ghMutex);
#endif

class VanitySearch;

#define WILD 0  // Wild kangaroo
#define TAME 1  // Tame kangaroo

/////////////////////////////////////////////////

typedef struct {
  VanitySearch *obj;
  int	threadId;
  bool	isRunning;
  bool	hasStarted;
  bool	rekeyRequest;
  int	gridSizeX;
  int	gridSizeY;
  int	gpuId;

  Int	dK;
  Point Kp;
  bool	type; // false - Tame, true - Wild

} TH_PARAM;


typedef struct {
	Int bnL;
	Int bnU;
	Int bnW;
	int pow2L;
	int pow2U;
	int pow2W;
	Int bnM;
	Int bnWsqrt;
	int pow2Wsqrt;
} structW;

typedef struct {
	uint64_t n0;
	//uint64_t n1;
	//uint64_t n2;
	//uint64_t n3;
	Int distance;
} hashtb_entry;


/////////////////////////////////////////////////

class VanitySearch {

public:

  VanitySearch(Secp256K1 *secp, std::vector<Point> &targetPubKeys, Point targetPubKey, structW *stR, int nbThread, int KangPower, int DPm, bool stop, std::string outputFile, int flag_verbose, uint32_t maxFound, uint64_t rekey, bool flag_comparator, int divBit, bool useWorkFile, bool createWorkFile, bool useDrive);    

  void Search(bool useGpu, std::vector<int> gpuId, std::vector<int> gridSize);
  void FindKeyCPU(TH_PARAM *p);
  void getGPUStartingKeys(int thId, int groupSize, int nbThread, Point itargetPubKey, Int *keys, Point *p);
  bool File2save(Int px,  Int key, uint32_t stype, int nb_file);
  bool File2saveHT(Int *HTpx, Int *HTkey, uint32_t *HTtype, int nb_file);
  //bool Comparator();
  void ReWriteFiles();
  bool SolverChmod();
  void CreateJumpTable();
  void CreateJumpTable(uint32_t Jmax, int pow2w);
  bool TWSaveToDrive();
  bool TWUpload();
    
  void FindKeyGPU(TH_PARAM *p);
  void SolverGPU(TH_PARAM *p);

private:

  bool checkPrivKeyCPU(Int &checkPrvKey, Point &pSample);
  
  bool output(std::string msg);
  bool outputgpu(std::string msg);

  bool isAlive(TH_PARAM *p);
  bool hasStarted(TH_PARAM *p);
  void rekeyRequest(TH_PARAM *p);
  uint64_t getGPUCount();
  uint64_t getCPUCount();
  uint64_t getCountJ();
  
  uint64_t getJmaxofSp(Int& optimalmeanjumpsize, Int * dS);
    
  Secp256K1 *secp;
  
  void GenerateCodeDP(Secp256K1 *secp, int size);
    
  Int wkey;
  Int tkey;
  
  // Jump Table
  Int jumpDistance[NB_JUMP];
  Int jumpPointx[NB_JUMP];
  Int jumpPointy[NB_JUMP];
  
  Point	targetPubKey;
  Int resultPrvKey;

  uint64_t countj[256];

  int nbThread;
  int KangPower;
  int DPm;
  bool TWRevers;
  bool useDrive;
  bool flag_comparator;
  bool DivideKey_flag;
  volatile int div_bit_range;
  //
  bool useWorkFile;
  bool createWorkFile;
  bool save_work_fl;
  bool save_work_fl_pr;
  uint32_t save_work_timer;
  uint32_t save_work_timer_interval;
  uint32_t save_work_timer_start;
  
  int nbCPUThread;
  int nbGPUThread;
  uint64_t totalRW;
  
  uint64_t rekey;
  uint64_t lastRekey;
  
  std::string outputFile;  
  uint32_t maxFound;
    
  std::vector<Point> &targetPubKeys;
 
  int flag_verbose;
  bool flag_endOfSearch;
  bool flag_startGPU;
    
  structW *stR;
  Int bnL, bnU, bnW;
  int pow2L, pow2U, pow2W;
  Int bnM, bnWsqrt;
  int pow2Wsqrt;

  int xU, xV;
  uint64_t xUV;
  Int bxU, bxV, bxUV;

  uint64_t DPmodule;
  uint64_t JmaxofSp;
  uint32_t GPUJmaxofSp;
  Int sizeJmax;

  uint64_t maxDP, countDT, countDW, countColl;
  uint64_t HASH_SIZE;
  hashtb_entry *DPht;

  char			buff_s32[32+1] = {0};
  unsigned char	buff_u32[32+1] = {0};

#ifdef WIN64
  HANDLE ghMutex;
#else
  pthread_mutex_t  ghMutex;
#endif

};

#endif // VANITYH
