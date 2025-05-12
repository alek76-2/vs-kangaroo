/*
 * This free code from alek76
 * Backup Kangaroos 
 *
*/

#include "Backup.h"
#include "GPU/GPUEngine.h"
#include "IntGroup.h"
#include "Timer.h"
#include <string.h>
#include <math.h>
#include <algorithm>
#include <chrono>

#include <iostream>
#include <fstream>

using namespace std;


// ========================================================================================================
// Save work Kangaroos to file
// code from alek76
// ========================================================================================================
bool SaveWorkKangaroosToFile(int thId, int nbThread, Point *kangPoints, Int *kangDistance) { 
	// names
	std::string fileId = std::to_string(thId);
	//std::string file_name_tame;
	//std::string file_name_wild;
	//file_name_tame = "Work_Tame_Kangaroos_" + fileId + ".txt";
	//file_name_wild = "Work_Wild_Kangaroos_" + fileId + ".txt";
	std::string file_name;
	file_name = "Work_Kangaroos_" + fileId + ".txt";
	// msg
	//std::string tame_msg = "";
	//std::string wild_msg = "";
	std::string msg = "";
	std::string str_px = ""; 
	std::string str_py = ""; 
	std::string str_key = ""; 
	std::string getpx = "";
	std::string getpy = "";
	std::string getkey = "";
	std::string s0 = "0";
	// variables
	volatile int nbKang = nbThread * GPU_GRP_SIZE;
	volatile int kang_cnt = 0;
	
	/*
	// Open 2 files
	FILE *ft = stdout;
	ft = fopen(file_name_tame.c_str(), "wb");
	if (ft == NULL) {
		printf("\n[error] Cannot open file '%s' for writing! \n", file_name_tame.c_str()); 
		ft = stdout;
		return false;
	}
	Timer::SleepMillis(500);
	FILE *fw = stdout;
	fw = fopen(file_name_wild.c_str(), "wb");
	if (fw == NULL) {
		printf("\n[error] Cannot open file '%s' for writing! \n", file_name_wild.c_str()); 
		fw = stdout;
		return false;
	}
	*/
	// Open 1 files
	FILE *f = stdout;
	f = fopen(file_name.c_str(), "wb");// wb 
	if (f == NULL) {
		printf("\n[ERROR] SaveWorkKangaroosToFile() Cannot open file '%s' for writing! \n", file_name.c_str()); 
		f = stdout;
		return false;
	}
	//
	// Separation
	// #define WILD 0  // Wild kangaroo
	// #define TAME 1  // Tame kangaroo
	//
	volatile int i;
	volatile int g;
	//
	for (i = 0; i < nbThread; i++) {
		for (g = 0; g < (int)GPU_GRP_SIZE; g++) {
			// Get index
			int ind = (i * (int)GPU_GRP_SIZE + g);
			// get data
			str_px = kangPoints[ind].x.GetBase16().c_str(); 
			str_py = kangPoints[ind].y.GetBase16().c_str(); 
			str_key = kangDistance[ind].GetBase16().c_str(); 
			// normalize length 64
			getpx = "";// Point X
			for (int i = (int)str_px.size(); i < 64; i++) {
				getpx.append(s0);
			}
			getpx.append(str_px);
			//
			getpy = "";// Point Y
			for (int i = (int)str_py.size(); i < 64; i++) {
				getpy.append(s0);
			}
			getpy.append(str_py);
			//
			getkey = "";// Distance
			for (int i = (int)str_key.size(); i < 64; i++) {
				getkey.append(s0);
			}
			getkey.append(str_key);
			// end normalize length 64
			
			// string msg
			//tame_msg = getpx + " " + getpy + " " + getkey; 
			//wild_msg = getpx + " " + getpy + " " + getkey; 
			msg = getpx + " " + getpy + " " + getkey; 
			
			/*
			// Tipe Kangaroo is even and not even
			int kTipe = g % 2;
			if (kTipe == (int)TAME) {
				fprintf(ft, "%s\n", tame_msg.c_str());
			}
			if (kTipe == (int)WILD) {
				fprintf(fw, "%s\n", wild_msg.c_str());
			}
			*/
			// Write 1 work file
			fprintf(f, "%s\n", msg.c_str());
			//printf("kTipe: %d \n", kTipe);
			//printf("index: %d \n", ind);
			kang_cnt++;
			// Check debug printf
			if (i == 0 && g < 2) {// g == 1 TAME 
				//printf("\n[i] Check:\n");
				//printf("[i]    Px: %s\n", kangPoints[ind].x.GetBase16().c_str());
				//printf("[i]    Py: %s\n", kangPoints[ind].y.GetBase16().c_str());
				//printf("[i]  Dist: %s\n", kangDistance[ind].GetBase16().c_str());
				if(g == 0) printf("\n[i] WILD Check Distance: %s\n", kangDistance[ind].GetBase16().c_str());
				if(g == 1) printf("[i] TAME Check Distance: %s\n", kangDistance[ind].GetBase16().c_str());
			}
		}
	}// end Separation
	
	// close files
	fclose(f);
	//fclose(ft);
	//fclose(fw);
	
	if (kang_cnt == nbKang) { 
		printf("\n[i] Write work Kangaroos in file: %s \n", file_name.c_str()); 
		return true;
	}
	return false;
}

// ========================================================================================================
// Load work Kangaroos from files 
// code from alek76 date 16.10.2023 
// code tested OK 
// ========================================================================================================
bool LoadWorkKangaroosFromFile(int thId, int nbThread, Point *kangPoints, Int *kangDistance) { 
	
	// Variables
	std::string fileId = std::to_string(thId);
	std::string file_name;
	file_name = "Work_Kangaroos_" + fileId + ".txt";
	std::string str_px = "";
	std::string str_py = "";
	std::string str_key = "";
	
	// Used chrono for get load time 
	auto begin = std::chrono::steady_clock::now();
	
	// Get file size wild
	FILE *f = fopen(file_name.c_str(), "rb");
	if (f == NULL) {
		printf("[ERROR] LoadWorkKangaroosFromFile() Cannot fopen() %s %s\n", file_name.c_str(), strerror(errno));
		f = stdout;
		return false;
	}
	fseek(f, 0L, SEEK_END);
	size_t sz = ftell(f); // Get bytes
	size_t nbl = sz / 194; // Get lines
	fclose(f);
	// info
	printf("[i] File %s: %lu bytes %lu lines \n", file_name.c_str(), (uint64_t)sz, (uint64_t)nbl);
	
	// Parse File 
	string sline = "";
	ifstream inFile(file_name);
	volatile int i;
	volatile int g;
	
	if (inFile.is_open()) { 
		for (i = 0; i < nbThread; i++) {
			for (g = 0; g < (int)GPU_GRP_SIZE; g++) {
				// Get index
				volatile int ind = (i * (int)GPU_GRP_SIZE + g);
				// get line
				sline = "";
				getline(inFile, sline, '\n');
				
				// Remove ending \r\n
				int l = (int)sline.length() - 1;
				while (l >= 0 && isspace(sline.at(l))) {
					sline.pop_back();
					l--;
				}
				
				if (sline.length() == 194) {
					// copy
					str_px = sline.substr(0, 64); 
					str_py = sline.substr(65, 129);
					str_key = sline.substr(130, 194);
					// check
					//if (i < 10 && g < 10) printf("g: %d sizeof line: %d line: %s\n", g, (int)sizeof(sline), sline.c_str()); 
					//
					// const char* to char*
					char *px =  new char [64];// no +1 for the \0 
					char *py =  new char [64];
					char *key = new char [64];
					// copy
					memcpy(px, str_px.c_str(), 64);
					memcpy(py, str_py.c_str(), 64);
					memcpy(key, str_key.c_str(), 64);
					// set
					// Point X
					kangPoints[ind].x.SetBase16(px);
					// Point Y
					kangPoints[ind].y.SetBase16(py);
					// Key kangDistance
					kangDistance[ind].SetBase16(key);
					
					// clr 
					free(px);
					free(py);
					free(key);					
				}
				// check
				//if (i < 10 && g < 10) {
					//printf("g: %d kangPoints[%d].x: %s \n", g, ind, kangPoints[ind].x.GetBase16().c_str());
					//printf("g: %d kangPoints[%d].y: %s \n", g, ind, kangPoints[ind].y.GetBase16().c_str());
					//printf("g: %d kangDistance[%d]: %s \n", g, ind, kangDistance[ind].GetBase16().c_str());
				//}				
			}
		}
	} else {
		printf("[ERROR] LoadWorkKangaroosFromFile() Cannot open %s %s\n", file_name.c_str(), strerror(errno));
		return false;
	}// end if
	
	// close file
	inFile.close();	
	
	auto end = std::chrono::steady_clock::now();
	std::chrono::duration<double, std::milli> elapsed_ms = end - begin;
	
	if (1) { //if (flag_verbose > 0) { 
		printf("\n[i] Get Keys time: %.3f msec From file: %s Size: %lu bytes \n", elapsed_ms, file_name.c_str(), (uint64_t)sz); 
	}	
	
	return true;
	
}

// ========================================================================================================

// Hi ;)
