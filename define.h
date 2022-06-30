#ifndef _CUDA_GAME_ON_PLANE_
#define _CUDA_GAME_ON_PLANE_

typedef struct p
{
	float r;
	float q;
}point;

typedef struct m
{
	int rDim;
	int qDim;
	float* Value;
	bool* InGameSet;
} gameMatrix;

#define MAX(a,b) ((a > b) ? a : b)
#define MIN(a,b) ((a < b) ? a : b)


const int BLOCK_SIZE = 16;

const int ITERATION_NUM_CPU = 2;

#endif//#define _CUDA_GAME_ON_PLANE_