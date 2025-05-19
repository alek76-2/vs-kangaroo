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
 *
 * THIS CODE MODIFIED BY ALEK-76
*/

#include "Vanity.h"
#include "Base58.h"
#include "Bech32.h"
#include "hash/sha256.h"
#include "hash/sha512.h"
#include "IntGroup.h"
#include "Timer.h"
#include "hash/ripemd160.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#ifndef WIN64
#include <pthread.h>
#endif
#include <chrono>

#include <iostream>
#include <fstream>
//#include <string>

// add openssl
#include <openssl/bio.h>
#include <openssl/err.h>
#include <openssl/ec.h>
#include <openssl/rand.h>
#include <openssl/pem.h>
#include <openssl/ripemd.h>
#define ECCTYPE "secp256k1"
//

#include "Backup.h"

using namespace std;

Point	Sp[256];
Int		dS[256];

// ----------------------------------------------------------------------------


// max=2^48, with fractional part
char * prefSI_double(char *s, size_t s_size, double doNum) {

	size_t ind_si = 0;
	char prefSI_list[9] = { ' ', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y' };

	while ((uint64_t)(doNum/1000) > 0) {
		ind_si += 1;
		doNum /= 1000;
		if (ind_si > 100) {
			printf("\n[FATAL_ERROR] infinity loop in prefSI!\n");
			exit(EXIT_FAILURE);
		}
	}

	if (ind_si < sizeof(prefSI_list) / sizeof(prefSI_list[0])) {
		snprintf(&s[0], s_size, "%5.1lf", doNum);
		snprintf(&s[0+5], s_size-5, "%c", prefSI_list[ind_si]);
	}
	else {
		snprintf(&s[0], s_size, "infini");
	}

	return s;
}


// max=2^256, without fractional part
char * prefSI_Int(char *s, size_t s_size, Int bnNum) {

	size_t ind_si = 0;
	char prefSI_list[9] = { ' ', 'K', 'M', 'G', 'T', 'P', 'E', 'Z', 'Y' };

	Int bnZero; bnZero.SetInt32(0);
	Int bn1000; bn1000.SetInt32(1000);
	Int bnTmp;	bnTmp = bnNum;
	bnTmp.Div(&bn1000);
	while (bnTmp.IsGreater(&bnZero)) {
		ind_si += 1;
		bnTmp.Div(&bn1000);
		bnNum.Div(&bn1000);
		if (ind_si > 100) {
			printf("\n[FATAL_ERROR] infinity loop in prefSI!\n");
			exit(EXIT_FAILURE);
		}
	}

	if (ind_si < sizeof(prefSI_list) / sizeof(prefSI_list[0])) {
		snprintf(&s[0], s_size, "%3s.0", bnNum.GetBase10().c_str());
		snprintf(&s[0+5], s_size-5, "%c", prefSI_list[ind_si]);
	}
	else {
		snprintf(&s[0], s_size, "infini");
	}

	return s;
}


void passtime_tm(char *s, size_t s_size, const struct tm *tm) {

	size_t offset_start = 0;

	if (tm->tm_year - 70 > 0) {
		snprintf(&s[offset_start], s_size - offset_start, " %1iy", tm->tm_year - 70);
		offset_start += 3;
		if (((tm->tm_year - 70) / 10) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 100) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 1000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 10000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 100000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 1000000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 10000000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 100000000) != 0) offset_start += 1;
		if (((tm->tm_year - 70) / 1000000000) != 0) offset_start += 1;
	}
	if (tm->tm_mon > 0 || tm->tm_year - 70 > 0) {
		snprintf(&s[offset_start], s_size - offset_start, " %2im", tm->tm_mon);
		offset_start += 4;
	}
	if (tm->tm_mday - 1 > 0 || tm->tm_mon > 0 || tm->tm_year - 70 > 0) {
		snprintf(&s[offset_start], s_size - offset_start, " %2id", tm->tm_mday - 1);
		offset_start += 4;
	}

	snprintf(&s[offset_start], s_size - offset_start, " %02i:%02i:%02is", tm->tm_hour, tm->tm_min, tm->tm_sec);
}


void passtime(char *s, size_t s_size, Int& bnSec, double usec = 0.0, char v[] = "11111100") {

	size_t ofst = 0;

	Int bnTmp;
	Int bnZero; bnZero.SetInt32(0);
	char buff_s6[6+1] = {0};

	Int Y_tmp; Y_tmp.SetInt32(0);
	Int M_tmp; M_tmp.SetInt32(0);
	Int d_tmp; d_tmp.SetInt32(0);
	Int h_tmp; h_tmp.SetInt32(0);
	Int m_tmp; m_tmp.SetInt32(0);
	Int s_tmp; s_tmp.SetInt32(0);

	int ms_tmp = 0, us_tmp = 0;
	if (usec != 0.0) {
		ms_tmp = (int)((uint64_t)(usec * 1000) % 1000);
		us_tmp = (int)((uint64_t)(usec * 1000000) % 1000);
	}

	if (v[0] != 48) {	// year
		Y_tmp = bnSec; 
		bnTmp.SetInt32(60*60*24*30*12); Y_tmp.Div(&bnTmp);
		if (Y_tmp.IsGreater(&bnZero)) {
			prefSI_Int(buff_s6, sizeof(buff_s6), Y_tmp);
			snprintf(&s[ofst], s_size-ofst, " %6sy", buff_s6); ofst += 8;
		}
	}
	if (v[1] != 48) { //month
		M_tmp = bnSec; 
		bnTmp.SetInt32(60*60*24*30); M_tmp.Div(&bnTmp); 
		bnTmp.SetInt32(12); M_tmp.Mod(&bnTmp);
		if (M_tmp.IsGreater(&bnZero) || Y_tmp.IsGreater(&bnZero)) {
			snprintf(&s[ofst], s_size-ofst, " %2sm", M_tmp.GetBase10().c_str()); ofst += 4;
		}
	}
	if (v[2] != 48) { //day
		d_tmp = bnSec;
		bnTmp.SetInt32(60*60*24); d_tmp.Div(&bnTmp);
		bnTmp.SetInt32(30); d_tmp.Mod(&bnTmp);
		if (d_tmp.IsGreater(&bnZero) || M_tmp.IsGreater(&bnZero) || Y_tmp.IsGreater(&bnZero)) {
			snprintf(&s[ofst], s_size-ofst, " %02sd", d_tmp.GetBase10().c_str()); ofst += 4;
		}
	}
	if (v[3] != 48) { //hour
		h_tmp = bnSec;
		bnTmp.SetInt32(60*60); h_tmp.Div(&bnTmp);
		bnTmp.SetInt32(24); h_tmp.Mod(&bnTmp);
		if (1) {
			snprintf(&s[ofst], s_size-ofst, " %02s", h_tmp.GetBase10().c_str()); ofst += 3;
		}
	}
	if (v[4] != 48) { //min
		m_tmp = bnSec;
		bnTmp.SetInt32(60); m_tmp.Div(&bnTmp);
		bnTmp.SetInt32(60); m_tmp.Mod(&bnTmp);
		if (1) {
			snprintf(&s[ofst], s_size-ofst, ":%02s", m_tmp.GetBase10().c_str()); ofst += 3;
		}
	}
	if (v[5] != 48) { //sec
		s_tmp = bnSec;
		bnTmp.SetInt32(60); s_tmp.Mod(&bnTmp);
		if (1) {
			snprintf(&s[ofst], s_size-ofst, ":%02s", s_tmp.GetBase10().c_str()); ofst += 3;
		}
	}


	if (v[6] != 48) { //msec
		if ((v[6] == 49) 
			|| (Y_tmp.IsEqual(&bnZero) && M_tmp.IsEqual(&bnZero) && d_tmp.IsEqual(&bnZero) 
				&& h_tmp.IsEqual(&bnZero) && m_tmp.IsEqual(&bnZero) && s_tmp.IsEqual(&bnZero)
				)
			) {
			snprintf(&s[ofst], s_size-ofst, " %03dms", ms_tmp); ofst += 6;
		}
	}

	if (v[7] != 48) { //usec
		if ((v[7] == 49)
			|| (Y_tmp.IsEqual(&bnZero) && M_tmp.IsEqual(&bnZero) && d_tmp.IsEqual(&bnZero)
				&& h_tmp.IsEqual(&bnZero) && m_tmp.IsEqual(&bnZero) && s_tmp.IsEqual(&bnZero)
				&& ms_tmp==0)
			) {
			snprintf(&s[ofst], s_size-ofst, " %03dus", us_tmp); ofst += 6;
		}
	}

}



// ----------------------------------------------------------------------------

VanitySearch::VanitySearch(Secp256K1 *secp, vector<Point> &targetPubKeys, Point targetPubKey, structW *stR, int nbThread, int KangPower, int DPm, bool stop, std::string outputFile, int flag_verbose, uint32_t maxFound, uint64_t rekey, bool flag_comparator, int divBit, bool useWorkFile, bool createWorkFile, bool useDrive)   
  :targetPubKeys(targetPubKeys) {

  this->secp = secp;

  this->targetPubKeys = targetPubKeys;
  this->targetPubKey = targetPubKey;

  this->stR = stR;
  this->bnL = stR->bnL;
  this->bnU = stR->bnU;
  this->bnW = stR->bnW;
  this->pow2L = stR->pow2L;
  this->pow2U = stR->pow2U;
  this->pow2W = stR->pow2W;
  this->bnM = stR->bnM;
  this->bnWsqrt = stR->bnWsqrt;
  this->pow2Wsqrt = stR->pow2Wsqrt;
  
  this->nbThread = nbThread;
  this->KangPower = KangPower;
  this->DPm = DPm;
  this->flag_comparator = flag_comparator;
  
  this->nbGPUThread = 0;
  this->maxFound = maxFound;
  this->rekey = rekey;
    
  this->outputFile = outputFile;
  this->flag_verbose = flag_verbose;
  //
  this->useWorkFile = useWorkFile;
  this->createWorkFile = createWorkFile;
  this->useDrive = useDrive;
  
  this->div_bit_range = divBit;// Set Divider =  2^divBit
  
  // !!! SET DIVIDER !!!
  
  //div_bit_range = 11;// pow2W / 4;// = 0;// = 32;// for bit 130
  
  //div_bit_range = 32;//div_bit_range = (pow2W / 6) + 1;
  
  //div_bit_range = 11;// FIXED 2^11
  
  //div_bit_range = 10;// = 10;// For Bit 130
  
  //div_bit_range = 11;// for BIT 64
  
  //div_bit_range = 50;// = 20; for BIT 128
  
  //GenerateCodeDP(secp, 0xff);
  
  //
  
  // openssl
  //static EC_KEY *myecc = NULL;
  OpenSSL_add_all_algorithms();
  //myecc = EC_KEY_new_by_curve_name(OBJ_txt2nid(ECCTYPE));
  
  #ifdef WIN64
   // Seed random number generator with screen scrape and other hardware sources
   //RAND_screen();
   //printf("\nOpenSSL RAND_screen()\n");
   //printf("\nUsed OpenSSL v1.0.1a Random number generator \n");
   printf("[i] Used OpenSSL Random number generator \n");
  #else
   printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");
   printf("[i] Used OpenSSL Random number generator \n");
   printf("[i] ");
   //bool cmd_v = system("openssl version -a");
   bool cmd_v = system("openssl version -v");
   printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");   
  #endif
  
  // Seed random number generator with performance counter
  RandAddSeed();
  //
  
  lastRekey = 0;
  
  printf("[rangeW]	2^%u..2^%u ; W = U - L = 2^%u\n"
	  , pow2L, pow2U
	  , pow2W
  );

  /////////////////////////////////////////////////
  // hashtable for distinguished points

  // DPht,DTht,DWht - points, distinguished points of Tp/Wp
  // in hashtable, provides uniqueness distinguished points

  countDT = countDW = countColl = 0;
  //maxDP = 1 << 10; // 2^10=1024
  maxDP = 1 << 20; // 2^20=1048576
  printf("[DPsize]	%llu (hashtable size)\n", (unsigned long long)maxDP);

  HASH_SIZE = 2 * maxDP;

  DPht = (hashtb_entry *)calloc(HASH_SIZE, sizeof(hashtb_entry));

  if (NULL == DPht) {
	  printf("\n[FATAL ERROR] can't alloc mem %.2f %s \n", (float)(HASH_SIZE) * sizeof(hashtb_entry)/1024/1024/1024, "GiB");
	  exit(EXIT_FAILURE);
  }

  /////////////////////////////////////////////////
  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");

  printf("[pubkey#%d] loaded\n", pow2U);
  printf("[Xcoordinate] %s\n", targetPubKey.x.GetBase16().c_str());
  printf("[Ycoordinate] %s\n", targetPubKey.y.GetBase16().c_str());

  if (!secp->EC(targetPubKey)) {
	  printf("\n[FATAL_ERROR] invalid public key (Not lie on elliptic curve)\n");
	  exit(EXIT_FAILURE);
  }

  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");

  /////////////////////////////////////////////////
  // pre-compute set S(i) jumps of pow2 jumpsize

  Sp[0] = secp->G;
  dS[0].SetInt32(1);
  for (int i = 1; i < 256; ++i) {
	  dS[i].Add(&dS[i-1], &dS[i-1]);
	  Sp[i] = secp->DoubleAffine(Sp[i-1]);
  }
  printf("[+] Sp-table of pow2 points - ready \n");
  
}

// ----------------------------------------------------------------------------

bool VanitySearch::output(string msg) {
  
  FILE *f = stdout;
  f = fopen(outputFile.c_str(), "a");
    
  if (f == NULL) {
	  printf("[error] Cannot open file '%s' for writing! \n", outputFile.c_str());
	  f = stdout;
	  return false;
  } 
  else {
	  fprintf(f, "%s\n", msg.c_str());
	  fclose(f);
	  return true;
  }
}

// ----------------------------------------------------------------------------

bool VanitySearch::outputgpu(string msg) {
  
  FILE *f = stdout;
  f = fopen(outputFile.c_str(), "a");
    
  if (f == NULL) {
	  printf("[error] Cannot open file '%s' for writing! \n", outputFile.c_str());
	  f = stdout;
	  return false;
  } 
  else {
	  fprintf(f, "\nPriv: %s\n", msg.c_str());
	  fprintf(f, "\nTips: 1NULY7DhzuNvSDtPkFzNo6oRTZQWBqXNE9 \n");
	  fclose(f);
	  return true;
  }
}

// ----------------------------------------------------------------------------

bool VanitySearch::checkPrivKeyCPU(Int &checkPrvKey, Point &pSample) {

  Point checkPubKey = secp->ComputePubKey(&checkPrvKey);
	  
  if (!checkPubKey.equals(pSample)) {
	  if (flag_verbose > 1) {
		  printf("[pubkeyX#%d] %s \n", pow2U, checkPubKey.x.GetBase16().c_str());
		  printf("[originX#%d] %s \n", pow2U, pSample.x.GetBase16().c_str());

		  printf("[pubkeyY#%d] %s \n", pow2U, checkPubKey.y.GetBase16().c_str());
		  printf("[originY#%d] %s \n", pow2U, pSample.y.GetBase16().c_str());
	  }
	  return false;
  }
  return true;
}


// ----------------------------------------------------------------------------

#ifdef WIN64
DWORD WINAPI _FindKey(LPVOID lpParam) {
#else
void *_FindKey(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->FindKeyCPU(p);
  return 0;
}

#ifdef WIN64
DWORD WINAPI _FindKeyGPU(LPVOID lpParam) {
#else
void *_FindKeyGPU(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->FindKeyGPU(p);
  return 0;
}

#ifdef WIN64
DWORD WINAPI _SolverGPU(LPVOID lpParam) {
#else
void *_SolverGPU(void *lpParam) {
#endif
  TH_PARAM *p = (TH_PARAM *)lpParam;
  p->obj->SolverGPU(p);
  return 0;
}

// ----------------------------------------------------------------------------



void VanitySearch::FindKeyCPU(TH_PARAM *ph) {

  int thId = ph->threadId;

  if (flag_verbose > 0)
	  printf("[th][%s#%d] run.. \n", (ph->type ? "wild" : "tame"), (ph->type ? thId+1-xU : thId+1));


  countj[thId] = 0;
  ph->hasStarted = true;


  while (!flag_endOfSearch) {

	  countj[thId] += 1;

	  // check, is it distinguished point ?
	  if (!(ph->Kp.x.bits64[0] % DPmodule)) {

		  //printf("[Xn0] 0x%016llx \n", ph->Kp.x.bits64[0]);
		  //printf("[Xcoord] %064s \n", ph->Kp.x.GetBase16().c_str());

		  // send new distinguished point to parent
		  #ifdef WIN64
		  WaitForSingleObject(ghMutex, INFINITE);
		  #else
		  pthread_mutex_lock(&ghMutex);
		  #endif

		  uint64_t entry = ph->Kp.x.bits64[0] & (HASH_SIZE-1);
		  while (DPht[entry].n0 != 0) {

			  if (DPht[entry].n0 == ph->Kp.x.bits64[0]
				  //&& DPht[entry].n1 == ph->Kp.x.bits64[1]
				  //&& DPht[entry].n2 == ph->Kp.x.bits64[2]
				  //&& DPht[entry].n3 == ph->Kp.x.bits64[3]
				  && !flag_endOfSearch
				  ) {

					  //printf("[X] 0x%s\n", ph->Kp.x.GetBase16().c_str());

					  if (ph->dK.IsLower(&DPht[entry].distance))
						  resultPrvKey.Sub(&DPht[entry].distance, &ph->dK);
					  else if (ph->dK.IsGreater(&DPht[entry].distance))
						  resultPrvKey.Sub(&ph->dK, &DPht[entry].distance);
					  else {
						  printf("\n[FATAL_ERROR] dT == dW !!!\n");
						  //exit(EXIT_FAILURE);
					  }

					  if (checkPrivKeyCPU(resultPrvKey, targetPubKey)) {
						  flag_endOfSearch = true;
						  break;
					  }
					  else {
						  ++countColl;
						  printf("\n");
						  printf("[warn] hashtable collision(#%llu) found! \n", (unsigned long long)countColl);
						  //printf("[warn] hashtable collision(#%llu) found! 0x%016llX \n", countColl, ph->Kp.x.bits64[0]);
						  //printf("[warn] Xcoord=%064s \n", ph->Kp.x.GetBase16().c_str());
						  //printf("[warn] wrong prvkey 0x%s \n", resultPrvKey.GetBase16().c_str());
					  }

			  }
			  entry = (entry + (ph->Kp.x.bits64[1] | 1)) & (HASH_SIZE-1);
		  }
		  //if (flag_endOfSearch) break;
		  
		  if (ph->type) { ++countDW; } else { ++countDT; }
		  if (flag_verbose > 1) {
			  printf("\n[th][%s#%d][DP %dT+%dW=%llu+%llu=%llu] new distinguished point!\n"
				  , (ph->type ? "wild" : "tame"), (ph->type ? thId+1 - xU : thId+1)
				  , xU, xV, (unsigned long long)countDT, (unsigned long long)countDW, (unsigned long long)(countDT+countDW)
			  );
		  }
		  if (countDT+countDW >= maxDP) {
			  printf("\n[FATAL_ERROR] DP hashtable overflow! %dT+%dW=%llu+%llu=%llu (max=%llu)\n"
				  , xU, xV, (unsigned long long)countDT, (unsigned long long)countDW, (unsigned long long)(countDT+countDW), (unsigned long long)maxDP
			  );
			  exit(EXIT_FAILURE);
		  }

		  
		  DPht[entry].distance = ph->dK;
		  DPht[entry].n0 = ph->Kp.x.bits64[0];
		  //DPht[entry].n1 = ST.n[1];
		  //DPht[entry].n2 = ST.n[2];
		  //DPht[entry].n3 = ST.n[3];

		  #ifdef WIN64
		  ReleaseMutex(ghMutex);
		  #else
		  pthread_mutex_unlock(&ghMutex);
		  #endif
	  }
	  //if (flag_endOfSearch) break;

	  uint64_t pw = ph->Kp.x.bits64[0] % JmaxofSp;

	  //nowjumpsize = 1 << pw
	  Int nowjumpsize = dS[pw];

	  ph->dK.Add(&nowjumpsize);
	  
	  	  
	  // Jacobian points addition
	  //ph->Kp = secp->AddJacobian(ph->Kp, Sp[pw]); ph->Kp.Reduce(); 

	  // Affine points addition
	  ph->Kp = secp->AddAffine(ph->Kp, Sp[pw]);
  }

  ph->isRunning = false;

}

/*
// Default starting keys
// ----------------------------------------------------------------------------
void VanitySearch::getGPUStartingKeys(int thId, int groupSize, int nbThread, Point itargetPubKey, Int *keys, Point *p) {

	// Variables
	volatile int trbit = pow2W;// Tame Random Bit
	volatile int wrbit = pow2W;// Wild Random Bit
	// Starting Keys
	//Int TameStartingKey(&bnU);
	Int TameStartingKey(&bnM);
	//Int TameStartingKey(&bnL);
	//
	printf("\nGPU thId: %d ", thId);
	//volatile int bitlen = bnL.GetBitLength();
	//printf("GPU bnL: 2^%d \nGPU TameStartingKey: %s", bitlen, TameStartingKey.GetBase16().c_str());
	printf("GPU TameStartingKey: %s", TameStartingKey.GetBase16().c_str());
	// Make keys
	volatile int i;
	volatile int g;
	volatile int index = 0;
	//
	for (i = 0; i < nbThread; i++) {
	for (g = 0; g < (int)GPU_GRP_SIZE; g++) {
	// index
	index = i * (int)GPU_GRP_SIZE + g;
	// Get keys	
	if (rekey > 1) {
		// Get keys
		keys[index].SetInt32(0);// CLR !!!
		tkey.Rand(trbit);
		wkey.Rand(wrbit);
		//
		if (g % 2 == TAME) {
			keys[index].Add(&TameStartingKey);// Tame keys
			keys[index].Add(&tkey);
			// Check GPU Random Key
			if (g < 10 && i == 0) printf("\nGPU Tame Starting Key %d: %s ", g, keys[index].GetBase16().c_str());
			if (g == (int)GPU_GRP_SIZE-1 && i == nbThread-1) printf("\nGPU Tame Starting Key %d: %s Kangaroos: %d ", (i*g)+g, keys[index].GetBase16().c_str(), nbThread * (int)GPU_GRP_SIZE);
		} else {
			keys[index].Add(&wkey);// Wild keys
			// Check GPU Random Key
			if (g < 10 && i == 0) printf("\nGPU Wild Starting Key %d: %s ", g, keys[index].GetBase16().c_str());
			if (g == (int)GPU_GRP_SIZE-1 && i == nbThread-1) printf("\nGPU Wild Starting Key %d: %s Kangaroos: %d ", (i*g)+g, keys[index].GetBase16().c_str(), nbThread * (int)GPU_GRP_SIZE);
		}
	} else {
		// Get keys
		keys[index].SetInt32(0);// CLR !!!
		tkey.Rand(trbit);
		wkey.Rand(wrbit);
		//
		if (g % 2 == TAME) {
			keys[index].Add(&TameStartingKey);// Tame keys
			keys[index].Add(&tkey);
			// Check GPU Random Key
			if (g < 10 && i == 0) printf("\nGPU Tame Starting Key %d: %s ", g, keys[index].GetBase16().c_str());
			if (g == (int)GPU_GRP_SIZE-1 && i == nbThread-1) printf("\nGPU Tame Starting Key %d: %s Kangaroos: %d ", (i*g)+g, keys[index].GetBase16().c_str(), nbThread * (int)GPU_GRP_SIZE);
			
		} else {
			keys[index].Add(&wkey);// Wild keys
			// Check GPU Random Key
			if (g < 10 && i == 0) printf("\nGPU Wild Starting Key %d: %s ", g, keys[index].GetBase16().c_str());
			if (g == (int)GPU_GRP_SIZE-1 && i == nbThread-1) printf("\nGPU Wild Starting Key %d: %s Kangaroos: %d ", (i*g)+g, keys[index].GetBase16().c_str(), nbThread * (int)GPU_GRP_SIZE);
		}	  
	}// end if rekey
	
	// For check GPU code
	//keys[index].SetBase16("FF");
	//keys[index].SetInt32(10);
	//
	
	Int k;	
	k.SetInt32(0);// clr	
	k.Set(&keys[index]);
	// Point
    p[index] = secp->ComputePublicKey(&k);
	
	//if (0) {// Set 0 for check compute points in GPU
	if (g % 2 == WILD) {
	
		Point p1 = p[index];
		// WILD Add Points
		//p[index] = secp->AddAffine(itargetPubKey, p[index]);
		p[index] = secp->AddAffine(itargetPubKey, p1);
				
	}// end if WILD
	}// end for GPU_GRP_SIZE
	}// end for nbThread
	
	printf("\nNB Thread: %d , GPU_GRP_SIZE: %d ", nbThread, (int)GPU_GRP_SIZE);
	
	// end of function
}
*/

// ----------------------------------------------------------------------------

#define NB_DIVIDE_MAX 2048 // 2^11
//#define NB_DIVIDE_MAX 1024 // 2^10

void VanitySearch::getGPUStartingKeys(int thId, int groupSize, int nbThread, Point itargetPubKey, Int *keys, Point *p) {

	// Date modify 29.10.2023 22:31  Hi ;)
	
	// for check BIT 130 KEY
	// Compressed Address: 15jqHenqhNFmKavrENqnmTG7pbMAiauCT5
	// Secret hex: 0x278a8b6bbcb53e5a58b2e1adc53b9eb2c
	// pk: 024f1652a52418ebe61e5dadb090e70bdb9568436ac8dc13ed7e485db2c7fd6944
	// PX: 4f1652a52418ebe61e5dadb090e70bdb9568436ac8dc13ed7e485db2c7fd6944
	// PY: 9b08101f88f43b26a0648979a12ffacba9af0845303a42c931a905a48bdf7914
	//
	Int checkKey;
	checkKey.SetInt32(0);// CLR !!!
	//checkKey.SetBase16("278a8b6bbcb53e5a58b2e1adc53b9eb2c");// bit 130
	checkKey.Rand(pow2U);// Rand pow2U
	//checkKey.SetBase16("f7051f27b09112d4");// bit 64
	Int checkKeyDivideOut;
	checkKeyDivideOut.SetInt32(0);// CLR !!!
	volatile int OK_divider_cnt = 0;
	//
	
	// Constant seed for function RandFix()
	//rseed(0x600DCAFE);
	
	// Seed random number generator with performance counter
	//RandAddSeed();
	//
	  
	// Save Divide key in file
	std::string msg = "";
	FILE *f = stdout;
	if (DivideKey_flag) {
		// fopen
		f = fopen("DivideKeys.txt", "a+");
		if (f == NULL) {
			printf("\n[error] Cannot open file DivideKeys.txt for writing! \n");
			f = stdout;	  
		} else {
			printf("\n[i] Open file DivideKeys.txt for writing! ");
		}
	}
	// Variables
	Int *DivideKey = new Int[NB_DIVIDE_MAX];
	Point *DividePubKey = new Point[NB_DIVIDE_MAX];	
	//volatile int divkey_max;
	//divkey_max = (int)NB_DIVIDE_MAX;
	
	// temp point
	Point *T_Point = new Point[NB_DIVIDE_MAX];
	//Int *T_Key = new Int[NB_DIVIDE_MAX];
	// Order
	Point _G_Point = secp->G;
	Int _G_Key;
	_G_Key.SetInt32(1);
	// adding point
	Point *Step_Point = new Point[NB_DIVIDE_MAX];
	Int *Step_Key = new Int[NB_DIVIDE_MAX];
	//volatile uint32_t Step_Key_u32 = 1;
	//
	
	// Variable nbBit - Check Len Bits
	volatile int nbBit = 0;// = Key.GetBitLength();
		
	if(DivideKey_flag) {// if used Divide 
		// Print info
		printf("!!! Random Divide Key - Bit Range: 2^%d \n", div_bit_range); 
		fprintf(f, "!!! Random Divide Key - Bit Range: 2^%d \n", div_bit_range); 
		fprintf(f, "DividePubKeyPointX DivideKey AddKey AddTargetPubKeyPointX\n");
	}
	
	// Set Low Divide bit
	volatile int div_bit_low = div_bit_range - 1;
	if (div_bit_low < 0) div_bit_low = 0;
	Int Div_bnL;
	Div_bnL.SetInt32(1);
	Div_bnL.ShiftL((uint32_t)div_bit_low);
	// Set Up Divide bit
	Int Div_bnU;
	Div_bnU.SetInt32(1);
	Div_bnU.ShiftL((uint32_t)div_bit_range);  
	//printf("Divide Key bits: %d - %d Range Key %s:%s\n", div_bit_low, div_bit_range, Div_bnL.GetBase16().c_str(), Div_bnU.GetBase16().c_str());
	//	
	// Variables
	Int *d = new Int[NB_DIVIDE_MAX];// Divide Key  
	//int divkey = 2;
	volatile int cnt = 0;
	volatile int k = 0;
	//rseed(0x600DCAFE);// Fix RandFix()
		
	if (DivideKey_flag) {// if used Divide 
	for (k = 0; k < NB_DIVIDE_MAX; k++) {
		
		//DivStart:
		//divkey = k + 1;
		//DivideKey[k].SetInt32(divkey);
		//d[k].SetInt32(k + 2);// Set Divide Key 
		//d[k].SetInt32(k + 1);// Set Divide Key 
		
		// Random Divide key
		d[k].SetInt32(1);// CLR and Set 1
		
		//d[k].Rand(div_bit_range + 1);//d[k].Rand(div_bit_range);
		
		//d[k].Rand(div_bit_range);// !!!
		d[k].RandFix(div_bit_range);// FIXED RANDOM KEYS !!!
		
		if (d[k].IsZero()) {// if == 0
			d[k].SetInt32(1);// Set 1
		}
		
		// Check Range DIVIDER for TEST KEY
		checkKeyDivideOut.SetInt32(0);// CLR !!!
		checkKeyDivideOut.Set(&checkKey);
		Int DivModulo;
		DivModulo.SetInt32(1);// CLR and Set 1
		DivModulo.Set(&d[k]);
		Int InDiv(&DivModulo);// add !!!
		checkKeyDivideOut.Div(&InDiv, &DivModulo);
		
		Timer::SleepMillis(1);
		
		//
		//printf("\n[i] %d TRUE DIVIDER NB: %d TestKey: 0x%s Divider: 0x%s Result: 0x%s ", OK_divider_cnt, k, checkKey.GetBase16().c_str(), d[k].GetBase16().c_str(), checkKeyDivideOut.GetBase16().c_str()); 
		// OK
		if (DivModulo.IsZeroBits0()) {
			OK_divider_cnt++;			
			//printf("\n[i] %d TRUE DIVIDER NB: %d TestKey: 0x%s Divider: 0x%s Result: 0x%s ", OK_divider_cnt, k, checkKey.GetBase16().c_str(), d[k].GetBase16().c_str(), checkKeyDivideOut.GetBase16().c_str()); 
			printf("\n[i] %d TRUE DIVIDER NB: %d TestKey: 0x%s Divider: 0x%s Result: 0x%s ", OK_divider_cnt, k, checkKey.GetBase16().c_str(), InDiv.GetBase16().c_str(), checkKeyDivideOut.GetBase16().c_str()); 
			// OK
			//printf("\n[i] !!! DivModulo: 0x%s", DivModulo.GetBase16().c_str()); //debug
		}
		// end Check DIVIDER for TEST KEY
		
		// Fixed
		//rseed(0x600DCAFE);
		//d[k].RandFix(div_bit_range);// Fixed rseed() - fixed random divide keys !!!!!!!!!!!!!!!!
		
		// Set Low Divider 
		//if (k < 8) d[k].SetInt32((uint32_t)k);
		
		// !!! Set 0,1,2,3,4,5,6,...,(NB_DIVIDE_MAX-1)
		//d[k].SetInt32((uint32_t)k);
		
		if (k == 0) d[k].SetInt32(1);// Original Target PubKey
		
		// for check bit64
		//d[1].SetInt32(0x3AC4);// check bit64
		//d[2].SetInt32(0x3AC4);// check bit64
		//
		DivideKey[k].Set(&d[k]);// SET DivideKey !!!
		DivideKey[k].ModInvOrder();// !!! OK Inverse modulo _N the Divide key
		
		Timer::SleepMillis(1);
			
		//printf("\n nbDivideKey: %d Inverse DivideKey: %s ", divkey, DivideKey[k].GetBase16().c_str());
		cnt = k;
		if (cnt % 200 == 0) {//if (cnt % 500 == 0) {
			printf("\n[i] NB: %d DIVKey: %s Inverse DivideKey: %s     ", k, d[k].GetBase16().c_str(), DivideKey[k].GetBase16().c_str());
		}
		// Divide target PubKey
		//DividePubKey[k] = secp->MultKAffine(DivideKey[k], itargetPubKey);
		//
		// Add + Divide
		//Step_Key[k].Set(&d[k]);// Step Key
		// Adding keys
		Step_Key[k].SetInt32(0);// CLR !!!
		Step_Key[k].Rand(div_bit_range);// Step_Key - the Random Add Key to itargetPubKey 
		//
		Step_Point[k] = secp->ComputePublicKey(&Step_Key[k]);// Step Point
		if(k > 0) { 
			// add Point
			T_Point[k] = secp->AddAffine(itargetPubKey, Step_Point[k]);
			
		} else {
			// Original Point
			Step_Key[k].SetInt32(0);
			T_Point[k] = itargetPubKey;// Set Original Target PubKey 
		}		
		//
		// Divide Adding Points
		DividePubKey[k] = secp->MultKAffine(DivideKey[k], T_Point[k]);// divide Point
		
		// Divide Original Points
		//DividePubKey[k] = secp->MultKAffine(DivideKey[k], itargetPubKey);
		
		// Write msg in file DivideKeys.txt		
		msg = DividePubKey[k].x.GetBase16() + " " + d[k].GetBase16() + " " + Step_Key[k].GetBase16() + " " + T_Point[k].x.GetBase16();// Add + Divide
		
		fprintf(f, "%s\n", msg.c_str());		
		
	}// end for
	}// end if DivideKey_flag
	//
	// Close file
	if (DivideKey_flag) {
		fclose(f);
		printf("\n[i] Data Saved in file DivideKeys.txt ");
		#ifdef WIN64
		#else
		if (TWRevers) { //if (1) { //
			Timer::SleepMillis(100);
			// Copy file DivideKeys.txt in drive
			std::string comm_save = "cp ./DivideKeys.txt ../drive/'My Drive'/DivideKeys.txt";
			const char *csave = comm_save.c_str();
			bool saved = system(csave);
			if (!saved) printf("\n[i] Copy file DivideKeys.txt to Drive");// (saver) !!! err
		}	
		#endif
	}
	//
	// Variables Random Bit
	volatile int trbit = pow2W;// Tame Random Bit
	volatile int wrbit = pow2W;// Wild Random Bit
	volatile int offset_bit = div_bit_range;
	Int TameStartingKey(&bnM);
	// Random StartingKey range bnL - bnU
	//Int TameStartingKey;
	//Int tsk;
	//new_tsk:
	//tsk.Rand(pow2U);
	//if (tsk.IsLower(&bnL)) goto new_tsk;
	//TameStartingKey.Set(&tsk);
	//
	//
	if(DivideKey_flag) {// if used Divide 
		// Set Range Random Bit
		trbit = (trbit - div_bit_range) + 1;// +1 !!!
		wrbit = (wrbit - div_bit_range) + 1;
		//trbit = (trbit - div_bit_range);
		//wrbit = (wrbit - div_bit_range);
		// Set Ranges and space	
		// Offset Tame Starting Key
		TameStartingKey.ShiftR((uint32_t)offset_bit);// Tame Starting Key Shift -> R 
		//Offset space 
		//bnL.ShiftR((uint32_t)offset_bit);
		//bnU.ShiftR((uint32_t)offset_bit);
		//
		printf("\n[i] GPU TameStartingKey: %s div_bit_range: %d", TameStartingKey.GetBase16().c_str(), div_bit_range);
		
	} else {
		printf("\n[i] GPU TameStartingKey: %s", TameStartingKey.GetBase16().c_str());		
	}
	//  
	printf("\nGPU thId: %d ", thId);
	volatile int bitlen = bnL.GetBitLength();
	printf("GPU bnL: %d ", bitlen);
	if(DivideKey_flag) {
		printf("\nGPU Tame Points: [M-2^%d] + Rand((pow2W-2^%d)+1) ", offset_bit, div_bit_range);
		printf("\nGPU Wild Points: [Target] + Rand((pow2W-2^%d)+1) ", div_bit_range);
	} else {
		printf("\nGPU Tame Points: [M] + Rand(pow2W)");
		printf("\nGPU Wild Points: [Target] + Rand(pow2W)");
	}
	
	// Make keys
	volatile int i;
	volatile int g;
	volatile int index = 0;
	volatile int ind = 0;
	for (i = 0; i < nbThread; i++) {
	for (g = 0; g < (int)GPU_GRP_SIZE; g++) {
	// index
	index = i * (int)GPU_GRP_SIZE + g;
	// Get keys	
	if (rekey > 1) {
		// Get keys
		keys[index].SetInt32(0);// CLR !!!
		//tkey.Rand(trbit);// old
		//wkey.Rand(wrbit);//old
		// Check Length Random bits - tkey
		//tkeyRand1:
		tkey.Rand(trbit);
		//nbBit = tkey.GetBitLength();
		//if (nbBit < trbit - 1) goto tkeyRand1;
		// Check Length Random bits - wkey
		//wkeyRand1:
		wkey.Rand(wrbit);
		//nbBit = wkey.GetBitLength();
		//if (nbBit < wrbit - 1) goto wkeyRand1;
		// end Check Length Random bits
		//
		if (g % 2 == TAME) {
			keys[index].Add(&TameStartingKey);// Tame keys
			keys[index].Add(&tkey);
			// Check GPU Random Key
			if (g < 10 && i == 0) printf("\nGPU Tame Starting Key %d: %s ", g, keys[index].GetBase16().c_str());
			if (g == (int)GPU_GRP_SIZE-1 && i == nbThread-1) printf("\nGPU Tame Starting Key %d: %s Kangaroos: %d ", (i*g)+g, keys[index].GetBase16().c_str(), nbThread * (int)GPU_GRP_SIZE);
		} else {
			keys[index].Add(&wkey);// Wild keys
			// Check GPU Random Key
			if (g < 10 && i == 0) printf("\nGPU Wild Starting Key %d: %s ", g, keys[index].GetBase16().c_str());
			if (g == (int)GPU_GRP_SIZE-1 && i == nbThread-1) printf("\nGPU Wild Starting Key %d: %s Kangaroos: %d ", (i*g)+g, keys[index].GetBase16().c_str(), nbThread * (int)GPU_GRP_SIZE);
		}
	} else {
		// Get keys
		keys[index].SetInt32(0);// CLR !!!
		tkey.Rand(trbit);// old
		wkey.Rand(wrbit);// old
		//
		// Check Length Random bits - tkey / Check parity bits
		//tkeyRand:
		//tkey.Rand(trbit);
		//nbBit = tkey.GetBitLength();
		//if (nbBit < trbit - 1) goto tkeyRand;
		// Check parity bits
		//if ( (tkey.bits64[0] % 2) != 0 ) goto tkeyRand;
		// printf check parity
		//if (g < 10 && i == 0) printf("\nGPU Check Random Key %d of Parity bits Key: 0x%llX ", i, tkey.bits64[0]);// Check
		// Check Length Random bits - wkey
		//wkeyRand:
		//wkey.Rand(wrbit);
		//nbBit = wkey.GetBitLength();
		//if (nbBit < wrbit - 1) goto wkeyRand;
		// Check parity bits
		//if ( (wkey.bits64[0] % 2) != 0 ) goto wkeyRand;
		// end Check 
		//
		if (g % 2 == TAME) {
			keys[index].Add(&TameStartingKey);// Tame keys
			keys[index].Add(&tkey);
			// Check GPU Random Key
			if (g < 10 && i == 0) printf("\nGPU Tame Starting Key %d: %s ", g, keys[index].GetBase16().c_str());
			if (g == (int)GPU_GRP_SIZE-1 && i == nbThread-1) printf("\nGPU Tame Starting Key %d: %s Kangaroos: %d ", (i*g)+g, keys[index].GetBase16().c_str(), nbThread * (int)GPU_GRP_SIZE);
			
		} else {
			keys[index].Add(&wkey);// Wild keys
			// Check GPU Random Key
			if (g < 10 && i == 0) printf("\nGPU Wild Starting Key %d: %s ", g, keys[index].GetBase16().c_str());
			if (g == (int)GPU_GRP_SIZE-1 && i == nbThread-1) printf("\nGPU Wild Starting Key %d: %s Kangaroos: %d ", (i*g)+g, keys[index].GetBase16().c_str(), nbThread * (int)GPU_GRP_SIZE);
		}	  
	}// end if rekey
	
	// For check GPU code
	//keys[index].SetBase16("FF");
	//keys[index].SetInt32(10);
	//
	
	Int k;	
	k.SetInt32(0);// clr	
	k.Set(&keys[index]);
	// Point
    p[index] = secp->ComputePublicKey(&k);
	
	//volatile int ind = 0;
	
	if (g % 2 == WILD) { // Set 0 for check compute points in GPU
	
		Point p1 = p[index];// p1 test?
		
		if (!DivideKey_flag) {// Normal mode - No Divide
			// WILD Add Points - Normal mode
			//p[index] = secp->AddAffine(itargetPubKey, p[index]);
			p[index] = secp->AddAffine(itargetPubKey, p1);
			//
		} else {// Divide mode
			
			ind = (i * (int)GPU_GRP_SIZE + g) % (int)NB_DIVIDE_MAX;
			
			if (g < 1 && i < 20) {
				printf("\n[i] NB: %d DIVKey: %s  WILD Add DividePubKey.x: %s ", ind, d[ind].GetBase16().c_str(), DividePubKey[ind].x.GetBase16().c_str()); 
			} 
			
			// WILD Add Divide Points			
			//p[index] = secp->AddAffine(DividePubKey[ind], p[index]);
			p[index] = secp->AddAffine(DividePubKey[ind], p1);
		}
	}// end WILD
	}// end for GPU_GRP_SIZE
	}// end for nbThread
	
	// Print info
	//printf("\nDivide Key bits: %d - %d Range Key %s:%s ", div_bit_low, div_bit_range, Div_bnL.GetBase16().c_str(), Div_bnU.GetBase16().c_str());
	//printf("\nDivide Key bits: 1 - %d Range Divide Key: 1:%s ", div_bit_range, Div_bnU.GetBase16().c_str());
	printf("\nDivide Key bits: %d Range Divide Key: %s:%s ", div_bit_range, Div_bnL.GetBase16().c_str(), Div_bnU.GetBase16().c_str());
	//
	//printf("\n!!! Divide Key Bit Range: 2^%d ", div_bit_range);
	printf("\nNB Thread: %d , GPU_GRP_SIZE: %d ", nbThread, (int)GPU_GRP_SIZE);
	
	// clr memory
	delete [] DivideKey;
	delete [] DividePubKey;
	delete [] T_Point;
	//delete [] T_Key;
	delete [] Step_Point;
	delete [] Step_Key;
	delete [] d;// the Divide Key  
	//
	// end
}

// ----------------------------------------------------------------------------

void VanitySearch::CreateJumpTable() {
	
	printf("[i] Create Jump Table Size: %d \n", (int)NB_JUMP);
	
	// Create Jump Table
	Point *Sp = new Point[NB_JUMP];// !!! Set in GPUEngine.h #define NB_JUMP 128
	Int *dS = new Int[NB_JUMP];
	Sp[0] = secp->G;
	dS[0].SetInt32(1);
	jumpPointx[0].Set(&Sp[0].x);
	jumpPointy[0].Set(&Sp[0].y);
	jumpDistance[0].SetInt32(1);
	if (flag_verbose > 1) {
		printf("Jump: 0 Distance: %s \n", jumpDistance[0].GetBase16().c_str());
	}
	for (int i = 1; i < NB_JUMP; i++) {
		
		dS[i].Add(&dS[i-1], &dS[i-1]);
		Sp[i] = secp->DoubleAffine(Sp[i-1]);
		jumpDistance[i].Set(&dS[i]);
		jumpPointx[i].Set(&Sp[i].x);
		jumpPointy[i].Set(&Sp[i].y);
		
		if (flag_verbose > 1) {
			printf("Jump: %d Distance: %s \n", i, jumpDistance[i].GetBase16().c_str());
		}
	}
	// clr memory
	delete [] Sp;
	delete [] dS;
	//
}

// ----------------------------------------------------------------------------

void VanitySearch::CreateJumpTable(uint32_t Jmax, int pow2w) {
	
	//int jumpBit = pow2w / 2 + 1;
	int jumpBit_max = (int)Jmax + 1;// Do not change!
	// add - fixed max value
	//if ( jumpBit_max > NB_JUMP) jumpBit_max = (int)NB_JUMP; old 
	printf("Create Jump Table (size: %d) Max Jump: 2^%d \n", (int)NB_JUMP, jumpBit_max);
	//double maxAvg = pow(2.0,(double)jumpBit - 0.95);
	//double minAvg = pow(2.0,(double)jumpBit - 1.05);
	double distAvg;
	//printf("Jump Avg distance min: 2^%.2f\n",log2(minAvg));
	//printf("Jump Avg distance max: 2^%.2f\n",log2(maxAvg));
	
	// Kangaroo jumps
	// Constant seed for compatibilty of files tame-1.txt, tame-2.txt and wild-1.txt, wild-2.txt
	rseed(0x600DCAFE);
	
	Int totalDist;
	totalDist.SetInt32(0);
	// Original code
	//for (int i = 0; i < NB_JUMP; i++) {
	//	jumpDistance[i].Rand(jumpBit);
	//	totalDist.Add(&jumpDistance[i]);
	//	Point J = secp->ComputePublicKey(&jumpDistance[i]);
	//	jumpPointx[i].Set(&J.x);
	//	jumpPointy[i].Set(&J.y);		
	//}
	
	// Add small jumps
	const int small_jump = NB_JUMP / 2;// #define NB_JUMP 32
	Point *Sp = new Point[small_jump];
	Int *dS = new Int[small_jump];
	Sp[0] = secp->G;
	dS[0].SetInt32(1);	
	jumpPointx[0].Set(&Sp[0].x);
	jumpPointy[0].Set(&Sp[0].y);
	jumpDistance[0].SetInt32(1);
	if (flag_verbose > 1) {
		printf("Jump: 0 Distance: %s \n", jumpDistance[0].GetBase16().c_str());
	}
	for (int i = 1; i < NB_JUMP; i++) {
		
		if (i < small_jump) {
			// Set small jump
			dS[i].Add(&dS[i-1], &dS[i-1]);
			Sp[i] = secp->DoubleAffine(Sp[i-1]);
			jumpDistance[i].Set(&dS[i]);
			totalDist.Add(&jumpDistance[i]);
			jumpPointx[i].Set(&Sp[i].x);
			jumpPointy[i].Set(&Sp[i].y);		
		}		
		else {
			// Set original jumps
			jumpDistance[i].RandFix(jumpBit_max);// Original renamed Rand()
			//jumpDistance[i].Rand(jumpBit_max);// Rand OpenSLL
			totalDist.Add(&jumpDistance[i]);
			Point J = secp->ComputePublicKey(&jumpDistance[i]);
			jumpPointx[i].Set(&J.x);
			jumpPointy[i].Set(&J.y);
		}
		if (flag_verbose > 1) {
			printf("Jump: %d Distance: %s \n", i, jumpDistance[i].GetBase16().c_str());
		}
	}
	
	distAvg = totalDist.ToDouble() / (double)NB_JUMP;	
	
	printf("Jump Avg distance: 2^%.2f (+/- use option -p)\n", log2(distAvg));
	
	rseed((unsigned long)time(NULL));// Comment for check setting GPU_GRP_SIZE NB_JUMP
	
	// clr memory
	delete [] Sp;
	delete [] dS;
	//
	
}


// ----------------------------------------------------------------------------

bool VanitySearch::File2save(Int px,  Int key, uint32_t stype, int nb_file) {
	
	std::string n_file = std::to_string(nb_file);
	std::string fTame = "tame-" + n_file + ".txt";
	std::string fWild = "wild-" + n_file + ".txt";
	
	std::string str_px = px.GetBase16().c_str();
	std::string str_key = key.GetBase16().c_str();
	std::string getpx = "";
	std::string getkey = "";
	std::string p0 = "0";
	std::string k0 = "0";
	
	for (int i = (int)str_px.size(); i < 64; i++) {
		getpx.append(p0);
	}
	getpx.append(str_px);
	
	for (int i = (int)str_key.size(); i < 64; i++) {
		getkey.append(k0);
	}
	getkey.append(str_key);
	
	std::string str_write = getpx + " " + getkey;
	
	int lenstr = (int)str_write.length();
	
	//printf("\n lenstr: %d stype: %lu \n", lenstr, (unsigned long)stype);
	//printf("\n str_write: %s \n", str_write.c_str());
	
	// Write files 
	if (stype == 1 && lenstr == 129) {
		FILE *f1 = fopen(fTame.c_str(), "a");
		if (f1 == NULL) {
			printf("\n[error] Cannot open file %s for writing! %s \n", fTame.c_str(), strerror(errno));
			f1 = stdout;
			return false;
		} else {
			fprintf(f1, "%s\n", str_write.c_str());
			fclose(f1);
			return true;
		}		
	}
	if (stype == 0 && lenstr == 129) {
		FILE *f2 = fopen(fWild.c_str(), "a");
		if (f2 == NULL) {
			printf("\n[error] Cannot open file %s for writing! %s \n", fWild.c_str(), strerror(errno));
			f2 = stdout;
			return false;
		} else {
			fprintf(f2, "%s\n", str_write.c_str());
			fclose(f2);
			return true;
		}		
	}
	
	return true;
}

// ----------------------------------------------------------------------------
/*
bool VanitySearch::File2saveHT(Int *HTpx, Int *HTkey, uint32_t *HTtype, int nb_file) {
	
	std::string n_file = std::to_string(nb_file);
	std::string fTame = "tame-" + n_file + ".txt";
	std::string fWild = "wild-" + n_file + ".txt";
	std::string p0 = "0";
	std::string k0 = "0";
	
	FILE *f1 = fopen(fTame.c_str(), "a");
	if (f1 == NULL) {
		printf("\n[error] Cannot open file %s for writing! %s \n", fTame.c_str(), strerror(errno));
		f1 = stdout;
		return false;
	}
	
	FILE *f2 = fopen(fWild.c_str(), "a");
	if (f2 == NULL) {
		printf("\n[error] Cannot open file %s for writing! %s \n", fWild.c_str(), strerror(errno));
		f2 = stdout;
		return false;
	}	
	
	// Separate of index HT
	volatile int q;
	for (q = 0; q < HT_SIZE; q++) {
		std::string str_px = HTpx[q].GetBase16().c_str();
		std::string str_key = HTkey[q].GetBase16().c_str();
		uint32_t stype = HTtype[q];
		// check printf
		//printf("\n  tpx: %s stype: %lu ", str_px.c_str(), (unsigned long)stype);
		//printf("\n tkey: %s stype: %lu \n", str_key.c_str(), (unsigned long)stype);
		
		std::string getpx = "";
		std::string getkey = "";
		
		for (int i = (int)str_px.size(); i < 64; i++) {
			getpx.append(p0);
		}
		getpx.append(str_px);
		
		for (int i = (int)str_key.size(); i < 64; i++) {
			getkey.append(k0);
		}
		getkey.append(str_key);
		
		std::string str_write = getpx + " " + getkey;
		
		int lenstr = (int)str_write.length();
		
		// check printf
		//printf("\n lenstr: %d stype: %lu \n", lenstr, (unsigned long)stype);
		//printf("\n str_write: %s \n", str_write.c_str());
		//printf("\n str_px.size(): %d \n", (int)str_px.size());
		
		// Write in file tame
		if (stype == 1 && lenstr == 129) {
			fprintf(f1, "%s\n", str_write.c_str());
		}
		// Write in file wild
		if (stype == 0 && lenstr == 129) {
			fprintf(f2, "%s\n", str_write.c_str());
		}
	}
	
	fclose(f1);
	fclose(f2);
	
	return true;
}
*/

bool VanitySearch::File2saveHT(Int *HTpx, Int *HTkey, uint32_t *HTtype, int nb_file) {
	
	std::string n_file = std::to_string(nb_file);
	std::string fTame = "tame-" + n_file + ".txt";
	std::string fWild = "wild-" + n_file + ".txt";
	std::string p0 = "0";
	std::string k0 = "0";
	
	// Separate of index HT
	volatile int q = 0;

	// tame file
	FILE *f1 = fopen(fTame.c_str(), "a");
	if (f1 == NULL) {
		printf("\n[error] Cannot open file %s for writing! %s \n", fTame.c_str(), strerror(errno));
		f1 = stdout;
		return false;
	}
	
	for (q = 0; q < HT_SIZE; q++) {
		std::string str_px = HTpx[q].GetBase16().c_str();
		std::string str_key = HTkey[q].GetBase16().c_str();
		uint32_t stype = HTtype[q];
		// check printf
		//printf("\n  tpx: %s stype: %lu ", str_px.c_str(), (unsigned long)stype);
		//printf("\n tkey: %s stype: %lu \n", str_key.c_str(), (unsigned long)stype);
		
		std::string getpx = "";
		std::string getkey = "";
		
		for (int i = (int)str_px.size(); i < 64; i++) {
			getpx.append(p0);
		}
		getpx.append(str_px);
		
		for (int i = (int)str_key.size(); i < 64; i++) {
			getkey.append(k0);
		}
		getkey.append(str_key);
		
		std::string str_write = getpx + " " + getkey;
		
		int lenstr = (int)str_write.length();
		
		// check printf
		//printf("\n lenstr: %d stype: %lu \n", lenstr, (unsigned long)stype);
		//printf("\n str_write: %s \n", str_write.c_str());
		//printf("\n str_px.size(): %d \n", (int)str_px.size());
		
		// Write in file tame
		if (stype == 1 && lenstr == 129) {
			fprintf(f1, "%s\n", str_write.c_str());
		}
	}
	
	fclose(f1);
	
	// wild file
	FILE *f2 = fopen(fWild.c_str(), "a");
	if (f2 == NULL) {
		printf("\n[error] Cannot open file %s for writing! %s \n", fWild.c_str(), strerror(errno));
		f2 = stdout;
		return false;
	}	
	
	for (q = 0; q < HT_SIZE; q++) {
		std::string str_px = HTpx[q].GetBase16().c_str();
		std::string str_key = HTkey[q].GetBase16().c_str();
		uint32_t stype = HTtype[q];
		// check printf
		//printf("\n  tpx: %s stype: %lu ", str_px.c_str(), (unsigned long)stype);
		//printf("\n tkey: %s stype: %lu \n", str_key.c_str(), (unsigned long)stype);
		
		std::string getpx = "";
		std::string getkey = "";
		
		for (int i = (int)str_px.size(); i < 64; i++) {
			getpx.append(p0);
		}
		getpx.append(str_px);
		
		for (int i = (int)str_key.size(); i < 64; i++) {
			getkey.append(k0);
		}
		getkey.append(str_key);
		
		std::string str_write = getpx + " " + getkey;
		
		int lenstr = (int)str_write.length();
		
		// check printf
		//printf("\n lenstr: %d stype: %lu \n", lenstr, (unsigned long)stype);
		//printf("\n str_write: %s \n", str_write.c_str());
		//printf("\n str_px.size(): %d \n", (int)str_px.size());
		
		// Write in file wild
		if (stype == 0 && lenstr == 129) {
			fprintf(f2, "%s\n", str_write.c_str());
		}
	}
	
	fclose(f2);
	
	return true;
}


// ----------------------------------------------------------------------------
/*
bool VanitySearch::Comparator() {
	
	// Used chrono for get compare time 
	auto begin = std::chrono::steady_clock::now();
	
	string WfileName = "wild.txt";
	string TfileName = "tame.txt";
	vector<string> Wpoint;
	vector<string> Wkey;

	vector<string> Tpoint;
	vector<string> Tkey;

	// Wild
	// Get file size wild
	FILE *fw = fopen(WfileName.c_str(), "rb");
	if (fw == NULL) {
		printf("Error: Cannot open %s %s\n", WfileName.c_str(), strerror(errno));
	}
	fseek(fw, 0L, SEEK_END);
	size_t wsz = ftell(fw); // Get bytes
	size_t nbWild = wsz / 129; // Get lines
	fclose(fw);
	// For check
	//printf("File wild.txt: %llu bytes %llu lines \n", (uint64_t)wsz, (uint64_t)nbWild);

	// Parse File Wild
	int nbWLine = 0;
	string Wline = "";
	ifstream inFileW(WfileName);
	Wpoint.reserve(nbWild);
	Wkey.reserve(nbWild);
	while (getline(inFileW, Wline)) {

		// Remove ending \r\n
		int l = (int)Wline.length() - 1;
		while (l >= 0 && isspace(Wline.at(l))) {
			Wline.pop_back();
			l--;
		}

		if (Wline.length() == 129) {
			Wpoint.push_back(Wline.substr(0, 64));
			Wkey.push_back(Wline.substr(65, 129));
			nbWLine++;
			// For check
			//printf(" %s    %d \n", Wpoint[0].c_str(), nbWLine);
			//printf(" %s    %d \n", Wkey[0].c_str(), nbWLine);						
		}
	}

	// Tame
	// Get file size tame
	FILE *ft = fopen(TfileName.c_str(), "rb");
	if (ft == NULL) {
		printf("Error: Cannot open %s %s\n", TfileName.c_str(), strerror(errno));
	}
	fseek(ft, 0L, SEEK_END);
	size_t tsz = ftell(ft); // Get bytes
	size_t nbTame = tsz / 129; // Get lines
	fclose(ft);
	// For check
	//printf("File tame.txt: %llu bytes %llu lines \n", (uint64_t)tsz, (uint64_t)nbTame);

	// Parse File Tame
	int nbTLine = 0;
	string Tline = "";
	ifstream inFileT(TfileName);
	Tpoint.reserve(nbTame);
	Tkey.reserve(nbTame);
	while (getline(inFileT, Tline)) {

		// Remove ending \r\n
		int l = (int)Tline.length() - 1;
		while (l >= 0 && isspace(Tline.at(l))) {
			Tline.pop_back();
			l--;
		}

		if (Tline.length() == 129) {
			Tpoint.push_back(Tline.substr(0, 64));
			Tkey.push_back(Tline.substr(65, 129));
			nbTLine++;
			// For check
			//printf(" %s    %d \n", Tpoint[0].c_str(), nbTLine);
			//printf(" %s    %d \n", Tkey[0].c_str(), nbTLine);
		}
	}

	// Compare lines
	int result = 0;
	string WDistance = "";
	string TDistance = "";
	for (int wi = 0; wi < nbWLine; wi++) {

		for (int ti = 0; ti < nbTLine; ti++) {
			
			if (strcmp(Wpoint[wi].c_str(), Tpoint[ti].c_str()) == 0) {
				result++;
				if (result > 0) { 
				printf("\n%d Compared lines Tame %d = Wild %d ", result, ti+1, wi+1);
				printf("\nTame Distance: 0x%s ", Tkey[ti].c_str());
				printf("\nWild Distance: 0x%s ", Wkey[wi].c_str());				
				}
				WDistance = Wkey[wi].c_str();
				TDistance = Tkey[ti].c_str();								
			}
		}

	}
	
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> elapsed_ms = end - begin;
	if (flag_verbose > 0) { 
		printf("\n[i] Comparator time: %.*f msec %s %llu bytes %s %llu bytes \n", 3, elapsed_ms, WfileName.c_str(), (uint64_t)wsz, TfileName.c_str(), (uint64_t)tsz); 
	}
	
	if (result > 0) {
		
		// Get SOLVED
		Int WDist;
		Int TDist;
		Int Priv;
		char *wd = new char [WDistance.length()+1];
		char *td = new char [TDistance.length()+1];
		strcpy(wd, WDistance.c_str());
		strcpy(td, TDistance.c_str());
		WDist.SetBase16(wd);
		TDist.SetBase16(td);
		Priv.SetInt32(0);
		
		// clr memory
		free(wd);
		free(td);
		//
		
		if (TDist.IsLower(&WDist))
			Priv.Sub(&WDist, &TDist);
		else if (TDist.IsGreater(&WDist))
			Priv.Sub(&TDist, &WDist);
		else {
			printf("\n[FATAL_ERROR] Wild Distance == Tame Distance !!!\n");			
		}
		
		printf("\nSOLVED: 0x%s \n", Priv.GetBase16().c_str());
		printf("Tame Distance: 0x%s \n", TDist.GetBase16().c_str());
		printf("Wild Distance: 0x%s \n", WDist.GetBase16().c_str());
		printf("Tips: 1NULY7DhzuNvSDtPkFzNo6oRTZQWBqXNE9  ;-) \n");
		printf("\n[i] Comparator time: %.*f msec %s %llu bytes %s %llu bytes \n", 3, elapsed_ms, WfileName.c_str(), (uint64_t)wsz, TfileName.c_str(), (uint64_t)tsz); 
		
		// SAVE SOLVED
		bool saved = outputgpu(Priv.GetBase16().c_str());
		if (saved) {
			printf("[i] Success saved to file %s\n", outputFile.c_str());
		}		
		return true;
	}
	return false;
}
*/

// ----------------------------------------------------------------------------

bool VanitySearch::TWSaveToDrive() {

	#ifdef WIN64
	#else
	// Copy tame and wild files in My drive
	for (int i = 0; i < NB_WORK; i++) {
		std::string si = std::to_string(i);
		std::string fTame = "tame-" + si + ".txt";
		std::string fWild = "wild-" + si + ".txt";
		std::string save_tame = "cp ./" + fTame + " ../drive/'My Drive'/" + fTame;
		std::string save_wild = "cp ./" + fWild + " ../drive/'My Drive'/" + fWild;
		const char* tsave = save_tame.c_str();
		const char* wsave = save_wild.c_str();
		bool tsaved = system(tsave);
		Timer::SleepMillis(1000);
		bool wsaved = system(wsave);
		Timer::SleepMillis(1000);
		if (!tsaved && !wsaved && i == (NB_WORK - 1)) {
			return true;
		}
	}	
	#endif 
	return false;

}

// ----------------------------------------------------------------------------

bool VanitySearch::TWUpload() {

	#ifdef WIN64
	#else
	// Upload tame and wild files
	for (int i = 0; i < NB_WORK; i++) {
		std::string ui = std::to_string(i);
		std::string fTame = "tame-" + ui + ".txt";
		std::string fWild = "wild-" + ui + ".txt";
		std::string upload_tame = "cp ../drive/'My Drive'/" + fTame + " ./" + fTame;
		std::string upload_wild = "cp ../drive/'My Drive'/" + fWild + " ./" + fWild;
		const char* tupload = upload_tame.c_str();
		const char* wupload = upload_wild.c_str();
		bool tup = system(tupload);
		Timer::SleepMillis(1000);
		bool wup = system(wupload);
		Timer::SleepMillis(1000);
		if (!tup && !wup && i == (NB_WORK - 1)) {
			return true;
		}
	}	
	#endif 
	return false;

}

// ----------------------------------------------------------------------------

void VanitySearch::FindKeyGPU(TH_PARAM *ph) {

  bool ok = true;
  bool get_ok = true;

#ifdef WITHGPU

  // GPU Thread
  
  TWRevers = false;// Aggregation of distinguished points, if true - Set int pow2dp = 26;// Fixed in GPUEngine.cu
  //TWRevers = true;
  TWRevers = useDrive;
  
  // Divide flag
  DivideKey_flag = true;
  DivideKey_flag = false;// Disablr Divide PubKey !!!
  
  if (div_bit_range > 0) DivideKey_flag = true;
  
  if (TWRevers) { 
	//ReWriteFiles();
	printf("[i] Uploading tame and wild files, wait...\n");//printf("[i] Uploading work files, wait...\n");
	bool F2Upload = TWUpload();
	if (F2Upload) {
		printf("[i] Upload tame and wild files\n");
	}
  }
  else {
	if (createWorkFile) ReWriteFiles();// add if !!!
	if (!useWorkFile) ReWriteFiles();
  }
    
  // Solver chmod
  #ifdef WIN64
  #else
  bool setChmod = SolverChmod();
  #endif  
  
  // Select comparator type
  //flag_comparator = true;
  if (flag_comparator) {
	printf("[i] Used Comparator in Python\n");
  } else {
	//printf("[i] Used Comparator in C++\n");
	printf("[i] -nocmp OFF Comparator in Python\n");	  
  }
  
  
  // mean jumpsize
  // by Pollard ".. The best choice of m (mean jump size) is w^(1/2)/2 .."
  Int GPUmidJsize;
  GPUmidJsize.Set(&bnWsqrt);
  GPUmidJsize.ShiftR(1);// Wsqrt / 2
  //GPUJmaxofSp = (uint32_t)getJmaxofSp(GPUmidJsize, dS);// For Table G,2G,4G,8G,16G,...,(2^NB_JUMP)G
  
  printf("===========================================================================\n");
  
  //GPUJmaxofSp = (uint32_t)pow2Wsqrt / 2;// Lower !!
  // !!!
  GPUJmaxofSp = (uint32_t)pow2Wsqrt + 1;// +1 ?
  
  //DPm = 18;//20;// = 20;// !!!!!!!!!!!!!!!! Fixed DP Modulo
  
  //GPUJmaxofSp = (uint32_t)pow2W - DPm;// Fixed Jump pow2W - DPm !!!
  
  int Set_div_bit_range = div_bit_range;
  
  if(DivideKey_flag) {
	//GPUJmaxofSp = (uint32_t)pow2Wsqrt;
	GPUJmaxofSp = (uint32_t)pow2Wsqrt + 1;// +1 ?
	//GPUJmaxofSp = (uint32_t)pow2W - DPm;
	
  }
  //
  printf("by Pollard (.. The best choice of m (mean jump size: 2^%lu) is w^(1/2)/2 ..) \n", (unsigned long)GPUJmaxofSp);
    
  GPUJmaxofSp += (uint32_t)KangPower;// + Kangaroo Power  
  
  if (GPUJmaxofSp < (uint32_t)(pow2Wsqrt / 2)) GPUJmaxofSp = (uint32_t)pow2Wsqrt / 2;// Lower !!
  
  #ifdef USE_ORIGINAL_JUMP
  
  printf("[i] Used %d Jump Table: G,2G,4G,8G,16G,...,(2^NB_JUMP)G\n", (int)NB_JUMP);
  
  CreateJumpTable();// G,2G,4G,8G,16G,...,(2^NB_JUMP)G  
  
  #else
  
  printf("[i] Used Mixed %d Jump Table: G,2G,4G,8G,16G,...,Rand(2^pow2Wsqrt+1)\n", (int)NB_JUMP);
  
  CreateJumpTable(GPUJmaxofSp, pow2W);
  
  #endif
  
  printf("===========================================================================\n");
  
  // Global init
  int thId = ph->threadId;
  GPUEngine g(ph->gridSizeX, ph->gridSizeY, ph->gpuId, maxFound, (rekey!=0), pow2W, totalRW, GPUJmaxofSp, KangPower, DPm);
  // Create Kangaroos
  int nbThread = g.GetNbThread();
  const int nbKang = nbThread * (int)GPU_GRP_SIZE;
  Point *p = new Point[nbKang];// Tame point
  Point *wp = new Point[nbKang];// Wild point
  Int *keys = new Int[nbKang];// Tame start keys
  Int *wkeys = new Int[nbKang];// Wild start keys
  vector<ITEM> found;
  // Work Kangaroos
  Point *kangPoints = new Point[nbKang];// Kangaroo Point
  Int *kangDistance = new Int[nbKang];// Kangaroo Distance Keys
  
  // Table for agregate DP
  Int *HTpx = new Int[HT_SIZE];
  Int *HTkey = new Int[HT_SIZE];
  uint32_t *HTtype = new uint32_t[HT_SIZE];
  volatile int ht_ind = 0;// index

  printf("GPU %s (%.1f MB used; GRP size: %d; nbSpin: %d) \n", g.deviceName.c_str(), g.GetMemory() / 1048576.0, (int)GPU_GRP_SIZE, (int)NB_SPIN); 
    
  countj[thId] = 0;
  uint32_t n_count = 0;
  uint32_t file_cnt;
  bool f2save = false;
  int nb_work = 0;
  save_work_fl = false;// Save Work - Start Keys to file Work_Kangaroos_id.txt
  
  if (useWorkFile || createWorkFile) save_work_fl = true;// add if !!!
  
  save_work_fl_pr = true;// Protection flag
  
  save_work_timer_start = (uint32_t)time(NULL);
  
  save_work_timer_interval = 60 * 10;// 5-30 min
  
  LOCK(ghMutex);
  
  // Get points and keys
  
  //getGPUStartingKeys(thId, g.GetGroupSize(), nbThread, targetPubKey, keys, p); old str
  
  if (useWorkFile) {
	  	bool getKeys = false;
		std::string file_name = "";
		
		// Skip Load Work - Generate New Starting Keys
		if (createWorkFile) goto getNewKeys;
		
		#ifdef WIN64 
		getKeys = LoadWorkKangaroosFromFile(thId, nbThread, p, keys);
		if (getKeys) { printf("[i] GPU Use Start Keys from work file: %s \n", file_name.c_str()); }
		#else 
		if (TWRevers) {// Get file from Drive
			// Upload file
			std::string fileId = std::to_string(thId);
			file_name = "Work_Kangaroos_" + fileId + ".txt";
			std::string scmd = "cp ../drive/'My Drive'/" + file_name + " ./" + file_name;
			const char* ccmd = scmd.c_str();
			bool up = system(ccmd);
			Timer::SleepMillis(1000);
			if (!up) {// ok upload
				printf("[i] Upload work file: %s \n", file_name.c_str());
				getKeys = LoadWorkKangaroosFromFile(thId, nbThread, p, keys);
			} else {
				printf("[i] ERROR !!! NO Upload work file: %s \n", file_name.c_str());
				printf("[i] ERROR !!! NO Upload work file: %s \n", file_name.c_str());
			}
		} else {// // Use local Drive
			getKeys = LoadWorkKangaroosFromFile(thId, nbThread, p, keys);
		}
		// info
		if (getKeys) { printf("[i] GPU Use Start Keys from work file: %s \n", file_name.c_str()); }
		#endif 
  } else {// NOT useWorkFile and Default Create Starting Keys
	  // goto
	  getNewKeys:
	  // Get Starting Keys
	  getGPUStartingKeys(thId, g.GetGroupSize(), nbThread, targetPubKey, keys, p);
  }
  
  //UNLOCK(ghMutex); g.SetKangaroos()
  
  ok = g.SetKangaroos(p, keys, jumpDistance, jumpPointx, jumpPointy);
  
  UNLOCK(ghMutex);
  
  ph->rekeyRequest = false;  
  ph->hasStarted = true;
  
  flag_startGPU = true;  
  
  
  while (ok && !flag_endOfSearch) {

    if (ph->rekeyRequest && (!useWorkFile) && (!createWorkFile)) {// if (ph->rekeyRequest) {
      LOCK(ghMutex);
	  getGPUStartingKeys(thId, g.GetGroupSize(), nbThread, targetPubKey, keys, p);
	  ok = g.SetKangaroos(p, keys, jumpDistance, jumpPointx, jumpPointy);
	  UNLOCK(ghMutex);
      ph->rekeyRequest = false;
	  n_count = 0;
    }
	
	// Call kernel
    ok = g.Launch(found);
	
	volatile int nb_found = (int)found.size();
	
	// printf("\n nbFound %d \n", (int)found.size());
	
	if (nb_found > 0) {// if GPU found DP
	
	LOCK(ghMutex);
		
	for (volatile int i = 0; i < nb_found && (!flag_endOfSearch); i++) {
		
		ITEM it = found[i];
		
		Int Tpx(&it.tpx);
		Int Tkey(&it.tkey);
		// check bits
		//if (Tpx.bits64[4] > 0) printf("\nERROR: Tpx.bits64[4] > 0 \n");
		//if (Tkey.bits64[4] > 0) printf("\nERROR: Tkey.bits64[4] > 0 \n");
		
		uint32_t kType = (uint32_t)(found[i].kIdx % 2);
		
		if (1) {
			
			//printf("\n it.kIdx: %llu \n", it.kIdx);			
			//printf("\n     Tpx: %s ", Tpx.GetBase16().c_str());
			//printf("\n    Tkey: %s ", Tkey.GetBase16().c_str());
			//printf("\n    type: %lu \n", (unsigned long)kType);// type 0 - Wild, 1 - Tame
			
			// Check Output Data GPU and Compute Tame keys
			
			if ((n_count & 1023) == 0) { // add if !!!
			if (kType == 1 && GPU_OUTPUT_CHECK == 1) {
				
				Int chk(&Tkey);
				Point chp = secp->ComputePubKey(&chk);
				if (strcmp(Tpx.GetBase16().c_str(), chp.x.GetBase16().c_str()) != 0) {
					printf("\n[error] Check Output Data GPU and Compute keys, kIdx: %llu", it.kIdx);
					printf("\n Check  key: %s ", Tkey.GetBase16().c_str());
					printf("\n    Point x: %s ", chp.x.GetBase16().c_str());
					printf("\n DP Point x: %s \n", Tpx.GetBase16().c_str());
					printf("[i] Set GPU_GRP_SIZE 64 or 32 \n");
					// !!!
					save_work_fl_pr = false;// Protection - disable saving
					printf("\n!!! Protection - DISABLED SAVE WORK KANGAROOS !!!\n\n");
				}
			}
			}// end if
			
			// Copy DP data in *ht
			HTpx[ht_ind].Set(&it.tpx);
			HTkey[ht_ind].Set(&it.tkey);
			HTtype[ht_ind] = kType;
			ht_ind++;
			// Save DP
			if (ht_ind == HT_SIZE) {
				
				ht_ind = 0;// Index = 0 - (HT_SIZE-1)
				
				file_cnt = n_count + (uint32_t)nb_found;// file_cnt = n_count + (uint32_t)found.size();
				
				nb_work = file_cnt % NB_WORK;
				
				//f2save = File2save(&Tpx, &Tkey, kType, nb_work); old
				
				f2save = File2saveHT(HTpx, HTkey, HTtype, nb_work);
				
				if (!f2save) {
					printf("\n[error] Not File2save type: %lu nb: %d\n", (unsigned long)kType, nb_work);
				}			
				f2save = false;// add				
			}			
		}
	}
	
	UNLOCK(ghMutex);
	
	}// end if GPU found DP
	
	// Save work - GPU Get Kangaroos of timers
	if (save_work_fl && save_work_fl_pr) {
		
		LOCK(ghMutex);
		
		// Function GET Kangaroo from GPUEngine
		get_ok = g.GetKangaroos(kangPoints, kangDistance);
		
		// info
		if (!get_ok) {		
			printf("\nERROR: GPUEngine::GetKangaroos() \n");
			printf("kangPoints[1].x: %s \n", kangPoints[1].x.GetBase16().c_str());// Tame Point 1 not even 
			printf("kangDistance[1]: %s \n", kangDistance[1].GetBase16().c_str());// Tame Distance 1 not even 
		}
		
		bool saveKang = SaveWorkKangaroosToFile(thId, nbThread, kangPoints, kangDistance);
		
		if (saveKang) {
			printf("[i] Save Work file OK\n");
		}
		
		save_work_fl = false;
		
		UNLOCK(ghMutex);
		
		// Copy to Drive
		#ifdef WIN64
		#else
		// Copy file to Drive
		if (TWRevers) {// if (1) {
			// Load file
			std::string load_fileId = std::to_string(thId);
			std::string load_file_name = "Work_Kangaroos_" + load_fileId + ".txt";
			std::string load_scmd = "cp ./" + load_file_name + " ../drive/'My Drive'/" + load_file_name;
			const char* load_ccmd = load_scmd.c_str();
			bool ld = system(load_ccmd);
			Timer::SleepMillis(100);
			if (!ld) printf("\n[i] Load Work file to Drive\n");
		}
		// end Copy file to Drive		
		#endif
		
	}// end Save work
	
	// Check time interval for save work
	if ((n_count & 1023) == 0) {
		if (useWorkFile || createWorkFile) {
			save_work_timer = (uint32_t)time(NULL);
			uint32_t time_diff = save_work_timer - save_work_timer_interval;
			if (time_diff > save_work_timer_start) {
				save_work_timer_start = save_work_timer;
				// Set flag to Save Work file
				save_work_fl = true;
			}
		}
	}
	// end Check time interval
	
	//printf("\n n_count: %lu ", (unsigned long)n_count);
	
	if (ok) {
      ++ n_count;
	  countj[thId] += (uint64_t)NB_SPIN * nbKang;
	  //countj[thId] += (uint64_t)NB_SPIN * nbKang * 2;// if use GPUCompute_small.h
    }

  }
    
  delete [] keys;
  delete [] p;
  delete [] wkeys;
  delete [] wp;
  // clr memory
  delete [] kangPoints;
  delete [] kangDistance;
  delete [] HTpx;
  delete [] HTkey;
  delete [] HTtype;
  //

#else
  ph->hasStarted = true;
  printf("GPU code not compiled, use -DWITHGPU when compiling.\n");
#endif

  ph->isRunning = false;
  
}

// ----------------------------------------------------------------------------

void VanitySearch::SolverGPU(TH_PARAM *ph) {
	
	// Wait Started GPU threads
	while (!flag_startGPU) {
		Timer::SleepMillis(500);
	}
	
	int slp = pow2Wsqrt * 1000;
	//slp *= 5;
	//slp = 1000 * 30;
	//slp = 1000 * 15;
	//slp = 1000 * 60;
	//slp = 1000 * 60 * 5;
	slp = 1000 * 60 * 1;// 1 min 
	
	//printf("\n[+] Runing Comparator every: %d sec\n", slp / 1000);

	if (flag_comparator) {
		printf("\n[+] Runing Comparator every: %d sec\n", slp / 1000);
	} else {
		printf("\n[+] OFF Comparator\n");
		return;
	}
	
	if (useWorkFile || createWorkFile) printf("[i] Save Work_Kangaroos_id.txt file every: %lu sec\n", (unsigned long)save_work_timer_interval);// add if !!!
	
	
	while (!flag_endOfSearch) {
		
		// Python Comparator
		// Get compare time
		auto begin = std::chrono::steady_clock::now();
		
		for (int ti = 0; ti < NB_WORK; ti++) {
			
			for (int wi = 0; wi < NB_WORK; wi++) {
			
				std::string nb_tame = std::to_string(ti);
				std::string nb_wild = std::to_string(wi);
				std::string fileTame = " tame-" + nb_tame + ".txt";
				std::string fileWild = " wild-" + nb_wild + ".txt";
				#ifdef WIN64
				// win solver
				std::string comm_cmp_win = "solver-all.py" + fileTame + fileWild;
				const char* scmpw = comm_cmp_win.c_str();
				bool solver_win = system(scmpw);
				if (!solver_win) {
					Timer::SleepMillis(500);
					flag_endOfSearch = true;
					//ReWriteFiles();
					printf("\n[i] No Cleaning wild and tame files \n");
					slp = 500;// ReWriteFiles
				}
				#else
				// solver
				//std::string comm_cmp = "./solver-all.py" + fileTame + fileWild;
				std::string comm_cmp = "python3 ./solver-all.py" + fileTame + fileWild;
				const char* scmp = comm_cmp.c_str();		
				bool solver = system(scmp);		
				if (!solver) {
					// Copy result in drive
					std::string comm_save = "cp ./Result.txt ../drive/'My Drive'/Result.txt";
					const char* csave = comm_save.c_str();
					bool saved = system(csave);
					Timer::SleepMillis(500);
					// Copy file DivideKeys.txt in drive
					comm_save = "cp ./DivideKeys.txt ../drive/'My Drive'/DivideKeys_copy.txt";
					csave = comm_save.c_str();
					saved = system(csave);
					Timer::SleepMillis(500);
					//
					flag_endOfSearch = true;
					//ReWriteFiles();
					printf("\n[i] No Cleaning wild and tame files \n");
					slp = 500;// ReWriteFiles
				}		
				#endif
			}
		}
		auto end = std::chrono::steady_clock::now();
		//auto elapsed_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin);
		std::chrono::duration<double, std::milli> elapsed_ms = end - begin;
		//if (flag_verbose > 0) { printf("\nPython Comparator time: %.*f msec \n", 3, elapsed_ms); }
		printf("\nPython Comparator time: %.*f msec \n", 3, elapsed_ms.count());
		
		Timer::SleepMillis(500);
		if (TWRevers) {
			bool TWSave = TWSaveToDrive();
			if (TWSave) {
				printf("\n[i] Copy tame and wild files in My Drive \n");
			}
		}
		
		Timer::SleepMillis(slp);
		
	}	
}

// ----------------------------------------------------------------------------

bool VanitySearch::isAlive(TH_PARAM *p) {

  bool isAlive = true;
  int total = nbCPUThread + nbGPUThread;
  for(int i = 0 ; i < total ; ++i)
    isAlive = isAlive && p[i].isRunning;

  return isAlive;

}

// ----------------------------------------------------------------------------

bool VanitySearch::hasStarted(TH_PARAM *p) {

  bool hasStarted = true;
  int total = nbCPUThread + nbGPUThread;
  for (int i = 0; i < total; ++i)
    hasStarted = hasStarted && p[i].hasStarted;

  return hasStarted;

}

// ----------------------------------------------------------------------------

void VanitySearch::rekeyRequest(TH_PARAM *p) {

  bool hasStarted = true;
  int total = nbCPUThread + nbGPUThread;
  for (int i = 0; i < total; i++)
  p[i].rekeyRequest = true;

}

// ----------------------------------------------------------------------------

uint64_t VanitySearch::getGPUCount() {

  uint64_t count = 0;
  for(int i=0;i<nbGPUThread;i++)
    count += countj[0x80L+i];
  return count;

}


// ----------------------------------------------------------------------------

uint64_t VanitySearch::getCountJ() {

  uint64_t count = 0;
  for(int i = 0 ; i < nbCPUThread ; ++i)
    count += countj[i];
  return count;

}

// ----------------------------------------------------------------------------

uint64_t VanitySearch::getJmaxofSp(Int& optimalmeanjumpsize, Int * dS) {

	if (flag_verbose > 0) {
		printf("[optimal_mean_jumpsize] %s \n", optimalmeanjumpsize.GetBase16().c_str());
	}
		
	Int sumjumpsize; 
	sumjumpsize.SetInt32(0);

	Int now_meanjumpsize, next_meanjumpsize;
	Int Isub1, Isub2;

	Int Ii;
	for (int i = 1; i < 256; ++i) {

		Ii.SetInt32(i);

		//sumjumpsize  = (2**i)-1
		//sumjumpsize += 2**(i-1)
		//sumjumpsize += dS[i-1]
		sumjumpsize.Add(&dS[i-1]);

		//now_meanjumpsize = int(round(1.0*(sumjumpsize) / (i)));
		now_meanjumpsize = sumjumpsize; now_meanjumpsize.Div(&Ii);

		//next_meanjumpsize = int(round(1.0*(sumjumpsize + 2**i) / (i+1)));
		//next_meanjumpsize = int(round(1.0*(sumjumpsize + dS[i]) / (i+1)));
		next_meanjumpsize = sumjumpsize; next_meanjumpsize.Add(&dS[i]); 
		Ii.SetInt32(i+1); next_meanjumpsize.Div(&Ii); Ii.SetInt32(i);

		//if  optimalmeanjumpsize - now_meanjumpsize <= next_meanjumpsize - optimalmeanjumpsize : 
		Isub1.Sub(&optimalmeanjumpsize, &now_meanjumpsize);
		Isub2.Sub(&next_meanjumpsize, &optimalmeanjumpsize);

		if (flag_verbose > 1)
			printf("[meanjumpsize#%d] %s(now) <= %s(optimal) <= %s(next)\n", i
				, now_meanjumpsize.GetBase16().c_str()
				, optimalmeanjumpsize.GetBase16().c_str()
				, next_meanjumpsize.GetBase16().c_str()
			);

		//if (Isub1.IsLowerOrEqual(&Isub2)) invalid compare for signed int!!! only unsigned int correct compared.
		if (
			   ((Isub1.IsNegative() || Isub1.IsZero()) && Isub2.IsPositive())
			|| ( Isub1.IsNegative() && (Isub2.IsZero() || Isub2.IsPositive()))
			|| ((Isub1.IsPositive() || Isub1.IsZero()) && (Isub2.IsPositive() || Isub2.IsZero()) && Isub1.IsLowerOrEqual(&Isub2))
			|| ((Isub1.IsNegative() || Isub1.IsZero()) && (Isub2.IsNegative() || Isub2.IsZero()) && Isub1.IsLowerOrEqual(&Isub2))
		) {

			if (flag_verbose > 0)
				printf("[meanjumpsize#%d] %s(now) <= %s(optimal) <= %s(next)\n", i
					, now_meanjumpsize.GetBase16().c_str()
					, optimalmeanjumpsize.GetBase16().c_str()
					, next_meanjumpsize.GetBase16().c_str()
				);

			// location in keyspace on the strip
			if (flag_verbose > 0) {

				//if (optimalmeanjumpsize - now_meanjumpsize) >= 0:
				if (Isub1.IsZero() || Isub1.IsPositive()) {
					//len100perc = 60
					Int len100perc; len100perc.SetInt32(60);
					//size1perc = (next_meanjumpsize-now_meanjumpsize)//len100perc
					Int size1perc; size1perc.Sub(&next_meanjumpsize, &now_meanjumpsize);
					size1perc.Div(&len100perc);
					printf("[i] Sp[%d]|", i);
						//, '-'*(abs(optimalmeanjumpsize - now_meanjumpsize)//size1perc)
					Isub1.Abs(); Isub1.Div(&size1perc);
					for (uint32_t j = 0 ; j < Isub1.GetInt32() ; ++j) printf("-");
					printf("J");
						//, '-'*(abs(next_meanjumpsize - optimalmeanjumpsize)//size1perc)
					Isub2.Abs(); Isub2.Div(&size1perc);
					for (uint32_t j = 0 ; j < Isub2.GetInt32() ; ++j) printf("-");
					printf("|Sp[%d]\n", i+1);
					//if (1.0 * abs(optimalmeanjumpsize - now_meanjumpsize) / abs(next_meanjumpsize - optimalmeanjumpsize) >= 0.25) {
					Isub1.Sub(&optimalmeanjumpsize, &now_meanjumpsize); Isub1.Abs();
					Isub2.Sub(&next_meanjumpsize, &optimalmeanjumpsize); Isub2.Abs();
					if ((float) Isub1.GetInt32() / Isub2.GetInt32() >= 0.25) {
						printf("[i] this Sp set has low efficiency (over -25%%) for this mean jumpsize\n");
					}
				}
				else {
					//now_meanjumpsize = int(round(1.0*(sumjumpsize - dS[i-1]) / (i-1)))
					//next_meanjumpsize = int(round(1.0*(sumjumpsize) / (i)))
					Ii.SetInt32(i-1);
					now_meanjumpsize = sumjumpsize; now_meanjumpsize.Sub(&dS[i-1]); now_meanjumpsize.Div(&Ii);
					Ii.SetInt32(i);
					next_meanjumpsize = sumjumpsize; next_meanjumpsize.Div(&Ii);

					//if  optimalmeanjumpsize - now_meanjumpsize <= next_meanjumpsize - optimalmeanjumpsize : 
					Isub1.Sub(&optimalmeanjumpsize, &now_meanjumpsize);
					Isub2.Sub(&next_meanjumpsize, &optimalmeanjumpsize);

					//len100perc = 60
					Int len100perc; len100perc.SetInt32(60);
					//size1perc = (next_meanjumpsize - now_meanjumpsize)//len100perc
					Int size1perc; size1perc.Sub(&next_meanjumpsize, &now_meanjumpsize);
					size1perc.Div(&len100perc);
					printf("[i] Sp[%d]|", i-1);
					//, '-'*(abs(optimalmeanjumpsize - now_meanjumpsize)//size1perc)
					Isub1.Abs(); Isub1.Div(&size1perc);
					for (uint32_t j = 0; j < Isub1.GetInt32() ; ++j) printf("-");
					printf("J");
					//, '-'*(abs(next_meanjumpsize - optimalmeanjumpsize)//size1perc)
					Isub2.Abs(); Isub2.Div(&size1perc);
					for (uint32_t j = 0; j < Isub2.GetInt32() ; ++j) printf("-");
					printf("|Sp[%d]\n", i);
					//if (1.0 * abs(next_meanjumpsize - optimalmeanjumpsize) / abs(optimalmeanjumpsize - now_meanjumpsize) >= 0.25) {
					Isub1.Sub(&next_meanjumpsize, &optimalmeanjumpsize); Isub1.Abs();
					Isub2.Sub(&optimalmeanjumpsize, &now_meanjumpsize); Isub2.Abs();
					if ((float) Isub1.GetInt32() / Isub2.GetInt32() >= 0.25) {
						printf("[i] this Sp set has low efficiency (over -25%%) for this mean jumpsize\n");
					}
				}
			}

			if (flag_verbose > 0)
				printf("[JmaxofSp] Sp[%d]=%s nearer to optimal mean jumpsize of Sp set\n", i
					, now_meanjumpsize.GetBase16().c_str()
				);
						
			return i;
		}
	}
	return 0;
}


// ----------------------------------------------------------------------------

void VanitySearch::Search(bool useGpu, std::vector<int> gpuId, std::vector<int> gridSize) {

  flag_endOfSearch = false;

  nbCPUThread = nbThread;
  nbGPUThread = (useGpu ? (int)gpuId.size() : 0);
  totalRW = 0;
  
  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");

  /////////////////////////////////////////////////
  // profiles load
  xU = 0;    
  xV = 0;
  // number kangaroos of herd T/W
  if (nbCPUThread == 1 || nbCPUThread == 2) {
	  xU = xV = 1;
	  xUV = 1;
	  bxU.SetInt32(xU); bxV.SetInt32(xV); bxUV.Mult(&bxU, &bxV);
  } 
  else if (nbCPUThread >= 4) {
	  // odd int
	  xU = (nbCPUThread/2)-1;
	  xV = (nbCPUThread/2)+1;
	  xUV = (uint64_t)xU * (uint64_t)xV;
	  bxU.SetInt32(xU); bxV.SetInt32(xV); bxUV.Mult(&bxU, &bxV);

	  //printf("[+] NO Recalc Sp-table of multiply UV\n");
	  if (1) {
		  for (int k = 0; k < 256; ++k) {
			  dS[k].Mult(&bxUV);
			  Sp[k] = secp->MultKAffine(bxUV, Sp[k]);
		  }
		  if (flag_verbose > 0)
			  printf("[+] recalc Sp-table of multiply UV\n");
	  }

  }

  printf("[UV] U*V=%d*%d=%llu (0x%02llx)\n", xU, xV, (unsigned long long)xUV, (unsigned long long)xUV);


  /////////////////////////////////////////////////
  //

  Int midJsize;
  midJsize.SetInt32(0);
  //printf("[i] 0x%s\n", midJsize.GetBase16().c_str());

  // mean jumpsize
  if (xU==1 && xV==1) {
	  // by Pollard ".. The best choice of m (mean jump size) is w^(1/2)/2 .."
	  //midJsize = (Wsqrt//2)+1
	  //midJsize = int(round(1.0*Wsqrt / 2))
	  midJsize = bnWsqrt; midJsize.ShiftR(1);
  } 
  else {
	  // expected of 2w^(1/2)/cores jumps
	  //midJsize = int(round(1.0*((1.0*W/(xU*xV))**0.5)/2))
	  //midJsize = int(round(1.0*(xU+xV)*Wsqrt/4));
	  midJsize = bnWsqrt; midJsize.Mult((uint64_t)(xU+xV)); midJsize.ShiftR(2);
	  //midJsize = int(round(1.0*Wsqrt/2));

  }

  JmaxofSp = (uint64_t)getJmaxofSp(midJsize, dS);

  //sizeJmax = 2**JmaxofSp
  sizeJmax = dS[JmaxofSp];

  /////////////////////////////////////////////////
  // discriminator of distinguished points

  int pow2dp = (pow2W/2)-2;
  if (pow2dp > 24) {//if (pow2dp > 63) {
	  //printf("\n[FATAL_ERROR] overflow DPmodule! (uint64_t)\n");
	  //exit(EXIT_FAILURE);
	  printf("[i] Old DPmodule: 2^%d \n", pow2dp);
	  pow2dp = 24;//pow2dp = 63;
	  printf("[i] New DPmodule: 2^%d \n", pow2dp);
  }
  DPmodule = (uint64_t)1<<pow2dp;
  if (flag_verbose > 0)
	  printf("[DPmodule] 2^%d = %llu (0x%llX) \n", pow2dp, (unsigned long long)DPmodule, (unsigned long long)DPmodule);


  /////////////////////////////////////////////////
  // create T/W herd

  //TH_PARAM *params = (TH_PARAM *)calloc((nbThread), sizeof(TH_PARAM));
  TH_PARAM *params = (TH_PARAM *)malloc((nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));
  memset(params, 0, (nbCPUThread + nbGPUThread) * sizeof(TH_PARAM));

  // dK - int, sum distance traveled
  // Kp - point, sum distance traveled
  // add !!!
  Int KtmpR;
  KtmpR.Rand(pow2W - 1);

  // Tame herd, generate start points
  for (int k = 0; k < xU; ++k) {

	  params[k].type = false; // false - Tame, true - Wild
  
	  //dT.append(M + k*xV)  
	  params[k].dK = bnM;
	  Int Ktmp; Ktmp.SetInt32(k); Ktmp.Mult(&bxV);
	  params[k].dK.Add(&Ktmp);
	  
	  // add !!!
	  params[k].dK.Add(&KtmpR);
	  

	  if (flag_verbose > 1)	printf(" dT[%d] 0x%s \n", k, params[k].dK.GetBase16().c_str());

	  //Tp.append(mul_ka(dT[k]))
	  //params[k].Kp = secp->MultKAffine(params[k].dK, secp->G);
	  params[k].Kp = secp->ComputePubKey(&params[k].dK);
	  
  }


  // Wild herd, generate start points
  for (int k = xU; k < (xU+xV) ; ++k) {

	  params[k].type = true; // false - Tame, true - Wild
	  //dW.append(1 + xU*k)
	  
	  params[k].dK.SetInt32(1);
	  Int Ktmp; Ktmp.SetInt32(k-xU); Ktmp.Mult(&bxU);
	  params[k].dK.Add(&Ktmp);
	  
	  // add !!!
	  //params[k].dK.Add(&KtmpR);
	  
	  
	  if (flag_verbose > 1)	printf(" dW[%d] 0x%s \n", k, params[k].dK.GetBase16().c_str());

	  //Wp.append(add_a(W0p, mul_ka(dW[k])))
	  //params[k].Kp = secp->AddAffine(targetPubKey, secp->MultKAffine(params[k].dK, secp->G));
	  // ;)
	  Point pdK = secp->ComputePubKey(&params[k].dK);
	  params[k].Kp = secp->AddAffine(targetPubKey, pdK);
	  //params[k].Kp = secp->AddAffine(targetPubKey, secp->ComputePubKey(&params[k].dK));
	  	  
  }

  printf("[+] %dT+%dW kangaroos - ready\n", xU, xV);


  /////////////////////////////////////////////////
  // Launch threads

  if (1) {//if (useGpu) {

	  printf("[CPU] threads: %d \n", nbThread);

	  // Tame herd, start
	  for (int k = 0; k < xU; ++k) {

		  params[k].obj = this;
		  params[k].threadId = k;
		  params[k].isRunning = true;

		  #ifdef WIN64
		  DWORD thread_id;
		  CreateThread(NULL, 0, _FindKey, (void*)(params+k), 0, &thread_id);
		  ghMutex = CreateMutex(NULL, FALSE, NULL);
		  #else
		  pthread_t thread_id;
		  pthread_create(&thread_id, NULL, &_FindKey, (void*)(params+k));
		  ghMutex = PTHREAD_MUTEX_INITIALIZER;
		  #endif
	  }

	  // Wild herd, start
	  for (int k = xU; k < (xU+xV); ++k) {

		  params[k].obj = this;
		  params[k].threadId = k;
		  params[k].isRunning = true;

		  #ifdef WIN64
		  DWORD thread_id;
		  CreateThread(NULL, 0, _FindKey, (void*)(params+k), 0, &thread_id);
		  ghMutex = CreateMutex(NULL, FALSE, NULL);
		  #else
		  pthread_t thread_id;
		  pthread_create(&thread_id, NULL, &_FindKey, (void*)(params+k));
		  ghMutex = PTHREAD_MUTEX_INITIALIZER;
		  #endif
	  }

  } 
  if (useGpu) {//if (1) {
  
	printf("[GPU] threads: %d Hang on to your hats... ;-)\n", nbGPUThread);
	
	// Launch GPU threads
  for (int i = 0; i < nbGPUThread; i++) {
    params[nbCPUThread+i].obj = this;
    params[nbCPUThread+i].threadId = 0x80L+i;
	params[nbCPUThread+i].isRunning = true;
    params[nbCPUThread+i].gpuId = gpuId[i];
	int x = gridSize[2 * i];
	int y = gridSize[2 * i + 1];
	if(!GPUEngine::GetGridSize(gpuId[i],&x,&y)) {
		return;
	}
	else {
		params[nbCPUThread+i].gridSizeX = x;
		params[nbCPUThread+i].gridSizeY = y;
		printf("[GPU] gridSizeX: %d gridSizeY: % d \n", x, y);// check
	}
	totalRW += GPU_GRP_SIZE * x*y;
#ifdef WIN64
    DWORD thread_id;
    CreateThread(NULL, 0, _FindKeyGPU, (void*)(params+(nbCPUThread+i)), 0, &thread_id);
	ghMutex = CreateMutex(NULL, FALSE, NULL);
#else
    pthread_t thread_id;
    pthread_create(&thread_id, NULL, &_FindKeyGPU, (void*)(params+(nbCPUThread+i)));
	ghMutex = PTHREAD_MUTEX_INITIALIZER;
#endif
  }
  // Solver
  #ifdef WIN64
	DWORD thread_id;
	CreateThread(NULL, 0, _SolverGPU, (void*)(params+(nbCPUThread)), 0, &thread_id);
	ghMutex = CreateMutex(NULL, FALSE, NULL);
  #else
	pthread_t thread_id;
	pthread_create(&thread_id, NULL, &_SolverGPU, (void*)(params+(nbCPUThread)));
	ghMutex = PTHREAD_MUTEX_INITIALIZER;
  #endif  
  //   
  }

  // Wait that all threads have started
  while (!hasStarted(params)) {
	Timer::SleepMillis(500);
  }

  /////////////////////////////////////////////////
  // indicator, progress, time, info...

  #ifndef WIN64
  setvbuf(stdout, NULL, _IONBF, 0);
  #endif
  
  uint64_t gpuCount = 0;
  uint64_t lastGPUCount = 0;

  //time vars
  double t0, t1, timestart;
  t0 = t1 = timestart = Timer::get_tick();
  time_t timenow, timepass;
  char timebuff[255 + 1] = { 0 };

  uint64_t countj_all=0, lastCount=0;
  memset(countj, 0, sizeof(countj));

  // Key rate smoothing filter
  #define FILTER_SIZE 8
  double lastkeyRate[FILTER_SIZE];
  double lastGpukeyRate[FILTER_SIZE];
  uint32_t filterPos = 0;
  
  double keyRate = 0.0;
  double gpuKeyRate = 0.0;
  
  memset(lastkeyRate, 0, sizeof(lastkeyRate));
  memset(lastGpukeyRate,0,sizeof(lastkeyRate));
  
  // Wait that all threads have started
  while (!hasStarted(params)) {
    Timer::SleepMillis(500);
  }
  
  int n_rot = 1;
  Int bnTmp, bnTmp2;

  // wait solv..
  while (isAlive(params)) {

	  int delay = 2000;
	  while (isAlive(params) && delay>0) {
		  Timer::SleepMillis(500);
		  delay -= 500;
	  }

	  t1 = Timer::get_tick();
	  //countj_all = getCountJ();
	  uint64_t gpuCount = getGPUCount();
	  countj_all = getCountJ() + gpuCount;
	  
	  keyRate = (double)(countj_all - lastCount) / (t1 - t0);
	  gpuKeyRate = (double)(gpuCount - lastGPUCount) / (t1 - t0);
	  lastkeyRate[filterPos%FILTER_SIZE] = keyRate;
	  lastGpukeyRate[filterPos%FILTER_SIZE] = gpuKeyRate;
	  filterPos++;

	  // KeyRate smoothing
	  double avgKeyRate = 0.0;
	  double avgGpuKeyRate = 0.0;
	  uint32_t nbSample;
	  for (nbSample = 0; (nbSample < FILTER_SIZE) && (nbSample < filterPos); nbSample++) {
		  avgKeyRate += lastkeyRate[nbSample];
		  avgGpuKeyRate += lastGpukeyRate[nbSample];
	  }
	  avgKeyRate /= (double)(nbSample);
	  avgGpuKeyRate /= (double)(nbSample);

	  if (isAlive(params)) {
			
		  printf("\r");

		  if     (n_rot == 1) { printf("[\\]"); n_rot = 2; }
		  else if(n_rot == 2) { printf("[|]"); n_rot = 3; }
		  else if(n_rot == 3) { printf("[/]"); n_rot = 0; }
		  else                { printf("[-]"); n_rot = 1; }

		  timepass = (time_t)(t1-timestart);
		  //passtime(timebuff, sizeof(timebuff), gmtime(&timepass));
		  bnTmp.SetInt32((uint32_t)timepass);
		  passtime(timebuff, sizeof(timebuff), bnTmp, 0.0, "11111100");
		  printf("[%s ;", timebuff); // timelost

		  //printf(" %6s j/s;", prefSI_double(buff_s32, sizeof(buff_s32), (double)avgKeyRate)); // speed
		  printf(" %6s j/s; [GPU %.2f Mj/s]", prefSI_double(buff_s32, sizeof(buff_s32), (double)avgKeyRate), avgGpuKeyRate / 1000000.0); // speed
		  //bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)avgKeyRate);
		  //printf(" %6s j/s;", prefSI_Int(buff_s32, sizeof(buff_s32), bnTmp) ); // speed

		  //printf(" %6sj", prefSI_double(buff_s32, sizeof(buff_s32), (double)countj_all)); // count jumps
		  bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)countj_all);
		  printf(" %6sj", prefSI_Int(buff_s32, sizeof(buff_s32), bnTmp)); // count jumps

		  bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)countj_all); bnTmp.Mult((uint64_t)100); 
		  //bnTmp.ShiftR(pow2Wsqrt+1);
		  //printf(" %3s.0%%;", bnTmp.GetBase10().c_str()); // percent progress, expected 2w^(1/2) jumps
		  bnTmp.ShiftR(pow2Wsqrt-2); double percprog = (double)bnTmp.GetInt32(); percprog /= 2*2*2;
		  printf(" %5.1f%%;", percprog); // percent progress, expected 2w^(1/2) jumps

		  printf(" dp/kgr=%.1lf;", (double)(countDT+countDW)/2 ); // distinguished points

		  bnTmp.SetInt32(1); bnTmp.ShiftL(1 + pow2Wsqrt); bnTmp.Sub((uint64_t)countj_all);
		  if (bnTmp.IsPositive()) {
			  bnTmp2.SetInt32(0); bnTmp2.SetQWord(0, (uint64_t)avgKeyRate);
			  bnTmp.Div(&bnTmp2);
			  passtime(timebuff, sizeof(timebuff), bnTmp, 0.0, "11111100");
			  printf("%s ]", timebuff); // timeleft
		  }
		  else {
			  bnTmp.SetInt32(0);
			  passtime(timebuff, sizeof(timebuff), bnTmp, 0.0, "11111100");
			  printf("%s ]", timebuff); // timeleft
		  }

		  printf("  ");
		  
		  /*
		  // PubKeys list End of search
		  int time_out = 180;// seconds
		  int max_perc = 300;// max percent
		  if (time_out > 0) {
			if ((int)timepass >= time_out && (int)percprog > max_perc) {
				printf("\n[i] End of search! Maximum percent: %d and time: %d seconds \n", max_perc, time_out);
				flag_endOfSearch = true;
				t0 = t1 = timestart = Timer::get_tick();
				timepass = time(NULL);
				countj_all = 0;
				lastCount = 0;
				gpuCount = 0;
				lastGPUCount = 0;
			} 
		  }
		  */
	  }
	  
	  if (rekey > 0) {
		if ((gpuCount - lastRekey) > (1000000 * rekey)) {
			// Rekey request
			rekeyRequest(params);
			lastRekey = gpuCount;
		}
	  }
	  
	  lastCount = countj_all;
	  lastGPUCount = gpuCount;
	  t0 = t1;
  }
  printf("\n");

  t1 = Timer::get_tick();
  timepass = (time_t)(t1 - timestart);
  if (timepass == 0) timepass = 1;

  /////////////////////////////////////////////////
  // print prvkey

  if (nbCPUThread > 0){
  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");

  printf("[prvkey#%d] 0x%s \n", pow2U, resultPrvKey.GetBase16().c_str());

  if (checkPrivKeyCPU(resultPrvKey, targetPubKey)) {
	  if (outputFile.length() > 0 
		  && output(resultPrvKey.GetBase16().c_str()
			  + string(":") + string("04") + targetPubKey.x.GetBase16().c_str() + targetPubKey.y.GetBase16().c_str()
			 )
		)  
		printf("[i] success saved pair prvkey:pubkey to file '%s'\n", outputFile.c_str());
  }
  else {
	  printf("[pubkey-check] failed!\n");
  }
  }

  /////////////////////////////////////////////////
  // final stat

  printf("[i]");

  t1 = Timer::get_tick();
  //countj_all = getCountJ();
  countj_all = getCountJ() + getGPUCount();

  ////timepass = (time_t)(t1 - timestart);
  printf(" %6s j/s;", prefSI_double(buff_s32, sizeof(buff_s32), (double)(countj_all / timepass))); // speed
  //bnTmp2.SetInt32((uint32_t)timepass); 
  //bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)countj_all); bnTmp.Div(&bnTmp2);
  //printf(" %6s j/s;", prefSI_Int(buff_s32, sizeof(buff_s32), bnTmp)); // speed

  //printf(" %6sj", prefSI_double(buff_s32, sizeof(buff_s32), (double)countj_all)); // count jumps
  bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)countj_all);
  printf(" %6sj", prefSI_Int(buff_s32, sizeof(buff_s32), bnTmp)); // count jumps

  bnTmp.SetInt32(1); bnTmp.ShiftL(1 + pow2Wsqrt);
  printf(" of %6sj", prefSI_Int(buff_s32, sizeof(buff_s32), bnTmp)); // expected 2w^(1/2) jumps

  bnTmp.SetInt32(0); bnTmp.SetQWord(0, (uint64_t)countj_all); bnTmp.Mult((uint64_t)100);
  //bnTmp.ShiftR(pow2Wsqrt+1);
  //printf(" %3s.0%%;", bnTmp.GetBase10().c_str()); // percent progress, expected 2w^(1/2) jumps
  bnTmp.ShiftR(pow2Wsqrt - 2); double percprog = (double)bnTmp.GetInt32(); percprog /= 2 * 2 * 2;
  printf(" %5.1f%%;", (double)percprog); // percent progress, expected 2w^(1/2) jumps

  printf(" DP %dT+%dW=%lu+%lu=%lu; dp/kgr=%.1lf;"
	  , xU, xV, (unsigned long)countDT, (unsigned long)countDW, (unsigned long)(countDT+countDW), (double)(countDT+countDW)/2
  ); // distinguished points

  printf("\n");

  /////////////////////////////////////////////////

  //timepass = (time_t)(t1 - timestart);
  bnTmp.SetInt32((uint32_t)timepass);
  passtime(timebuff, sizeof(timebuff), bnTmp, 0.0, "11111100");
  printf("[runtime] %s \n", timebuff);

  /////////////////////////////////////////////////

  free(params);
  free(DPht);

  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");
  timenow = time(NULL);
  strftime(timebuff, sizeof(timebuff), "%d %b %Y %H:%M:%S", gmtime(&timenow));
  printf("[DATE(utc)] %s\n", timebuff);
  printf("[~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~]\n");
  printf("[x] EXIT\n");exit(EXIT_SUCCESS);
}

// ----------------------------------------------------------------------------

void VanitySearch::ReWriteFiles(){

	bool ReWrite;
	std::string chkFile = "tame-0.txt";
	
	FILE *f = stdout;
	f = fopen(chkFile.c_str(), "r");
	
	if (f == NULL) {
		f = stdout;
		ReWrite = true;
	}
	else {
		fclose(f);
		ReWrite = false;
	}
	
	//Timer::SleepMillis(500);
	
	if (ReWrite) {
		
		printf("[i] Created tame and wild .txt files for save DP-Points!\n");
		
		for (int i = 0; i < NB_WORK; i++) {
			std::string ri = std::to_string(i);
			std::string fTame = "tame-" + ri + ".txt";
			std::string fWild = "wild-" + ri + ".txt";
			FILE *f1 = fopen(fTame.c_str(), "w");
			fclose(f1);
			FILE *f2 = fopen(fWild.c_str(), "w");
			fclose(f2);
		}
	}
	
	//Timer::SleepMillis(500);
}

// ----------------------------------------------------------------------------

bool VanitySearch::SolverChmod() {

	string comm_chmod = "chmod 755 ./solver-all.py";
	const char* schmod = comm_chmod.c_str();
	bool chm = system(schmod);
	
	if (chm) {
		return true;
	}
	else {
		return false;
	}

}

// Generator Divide Target Points
void VanitySearch::GenerateCodeDP(Secp256K1 *secp, int size) {
	
	// Compute Generator Divide Target Points
	size = 256;
	
	Point T_Point;
	
	T_Point = targetPubKey;// Target PubKey
	
	Int *Dk = new Int[size];
	Point *Dp = new Point[size];
	
	Dk[0].SetInt32(1);
	Dp[0] = T_Point;
	
	Int DivideKey;
	DivideKey.SetInt32(1);
	Int _One;
	_One.SetInt32(1);
	
	for (int i = 1; i < size; i++) {
		printf("\nGenerate Divide Target Points GPUDividePoints.h size i: %d", i);
		//
		DivideKey.Add(&DivideKey, &_One);// + 1
		Dk[i].Set(&DivideKey);
		printf("\n   DivKey: %s", Dk[i].GetBase16().c_str());
		Dk[i].ModInvOrder();// !!! OK Inverse modulo _N the Divide key
		Dp[i] = secp->MultKAffine(Dk[i], T_Point);// divide Point
		//
		printf("\nInvDivKey: %s", Dk[i].GetBase16().c_str());
	}
	
	// Write file
	FILE *f = fopen("GPU/GPUDividePoints.h", "wb");
	fprintf(f, "// File generated by GenerateCodeDP()\n");
	fprintf(f, "\n\n");
	fprintf(f, "// SecpK1 Generator Divide Target Points P/1, P/2, P/3, P/4, P/5,..., P/%d. \n", size);
	fprintf(f, "__device__ __constant__ uint64_t Dpx[][4] = {\n");
	for (int i = 0; i < size; i++) {
		fprintf(f, "    %s,\n", Dp[i].x.GetC64Str(4).c_str());
	}
	fprintf(f, "};\n");
	
	fprintf(f, "__device__ __constant__ uint64_t Dpy[][4] = {\n");
	for (int i = 0; i < size; i++) {
		fprintf(f, "    %s,\n", Dp[i].y.GetC64Str(4).c_str());
	}
	fprintf(f, "};\n\n");
	
	fprintf(f, "__device__ __constant__ uint64_t Dk[][4] = {\n");
	for (int i = 0; i < size; i++) {
		fprintf(f, "    %s,\n", Dk[i].GetC64Str(4).c_str());
	}
	fprintf(f, "};\n\n");
	
	fclose(f);
	delete[] Dk;
	delete[] Dp;

}


// Hi ;)
