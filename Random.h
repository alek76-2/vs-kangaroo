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

#ifndef RANDOM_H
#define RANDOM_H

#ifdef WIN64
	#include <windows.h>
	#include <time.h>
#else
	//#include <stdio.h>
	//#include <stdint.h>
	//#include <string.h>
	#include <sys/time.h>
	//#include <unistd.h>
	//#include <stdexcept>
#endif

#if defined(_MSC_VER) || defined(__BORLANDC__)
typedef __int64  int64;
typedef unsigned __int64  uint64;
#else
typedef long long  int64;
typedef unsigned long long  uint64;
#endif

double rnd();
unsigned long rndl();
unsigned long rndl_fixed();
void rseed(unsigned long seed);

int64 PerformanceCounter();
void RandAddSeed();

#endif
