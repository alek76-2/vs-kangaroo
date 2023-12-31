#ifndef BACKUPH
#define BACKUPH

#include "Point.h"


bool SaveWorkKangaroosToFile(int thId, int nbThread, Point *KangPoints, Int *KangDistance);

bool LoadWorkKangaroosFromFile(int thId, int nbThread, Point *kangPoints, Int *kangDistance);


#endif // BACKUPH