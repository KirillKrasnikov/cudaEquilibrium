#include <stdio.h>
#include <stdlib.h>
#include <float.h>
#include "define.h"

//--------------------nahozhdenie--A1-ravnovesiya-------------------------------
//Nahodim MaxInRaw
extern "C" __global__ void findMaxInRawReduce_kernel(int rDim, bool* inputInGS, bool* outputInGS,
										  float* inputData, float* outputData)
{
	__shared__ float data [BLOCK_SIZE];
	__shared__ bool inGameSet [BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = threadIdx.x;
	
	//Adres, kopiruemogo dannim processom elementa
	int ix = bx * blockDim.x + tid;
	int iy = by * rDim;
	
	inGameSet[tid] = inputInGS[ix + iy];
	if(inGameSet[tid])
		data[tid] = inputData[ix + iy];     // load into shared memeory
	else
		data[tid] = FLT_MIN;  //Esli ne v GameSet, iniciliziruem MIN_FLT,
							  //chtobi on v l'ubom sluchae ne stal Max

	__syncthreads ();

	for ( int s = blockDim.x / 2; s > 0; s >>= 1 )
    {
		if(tid < s)
		{
			data[tid] = MAX (data[tid], data[tid + s]);
			inGameSet[tid] = inGameSet[tid] || inGameSet[tid + s];
		}

		__syncthreads ();
	}

	if ( tid == 0 )                 // write result of block reduction
	{
		outputData[by * rDim + bx] = data [0];
		outputInGS[by * rDim + bx] = inGameSet[0];
	}
}

extern "C" __global__ void findMaxInRawFinish_kernel(int rDim, bool* inputInGS, bool* outputInGS,
										  float* inputData, float* outputData)
{
	__shared__ float data [BLOCK_SIZE];
	__shared__ bool inGameSet [BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = threadIdx.x;
	
	//Adres, kopiruemogo dannim processom elementa
	int ix = bx * blockDim.x + tid;
	int iy = by * rDim;
	
	inGameSet[tid] = inputInGS[ix + iy];
	if(inGameSet[tid])
		data[tid] = inputData[ix + iy];     // load into shared memeory
	else
		data[tid] = FLT_MIN;  //Esli ne v GameSet, iniciliziruem MIN_FLT,
							  //chtobi on v l'ubom sluchae ne stal Max

	__syncthreads ();

	if ( tid == 0 )                 
	{
		for ( int s = 1 ; s < blockDim.x; s++ )
	    {
				data[tid ] = MAX (data[tid ], data[tid + s]);
				inGameSet[tid] = inGameSet[tid] || inGameSet[tid + s];
	
		}

		outputData[by * rDim + bx] = data [0];
		outputInGS[by * rDim + bx] = inGameSet[0];
	}
}

//Tochnost' kak u CPU, vtoroy po skorosti sredi GPU
extern "C" __global__ void find_A1_EquilibriumVer0_kernel(float* Value, bool* InGameSet, int rDim, int qDim,
										  float* MaxInRaw, bool* A)
{
	int k = 0;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int ty = threadIdx.y;

	//Adress vichisl'aemogo dannim processom elementa
	int ix = bx;
	int iy = by * blockDim.y * rDim + ty * rDim;

	if(InGameSet[ix + iy])
	{
		A[ix + iy] = true;
		while(k < qDim && A[ix + iy])
		{
			if(InGameSet[k * rDim + ix] && (Value[k * rDim + ix] < Value[ix + iy]))
				if(Value[ix + iy] > MaxInRaw[k * rDim]) //Esli perviy mozhet beznakazanno uluchshit' svoy rezul'tat
				{
					A[ix + iy] = false; //tochka ne yavl'aetsa ravnovesnoy
				}
			k++;
		}
	}
	else
		A[ix + iy] = false;
}

//Samiy bistriy algoritm, no priemlemaya tochnost' pri dim >= 416*416
extern "C" __global__ void find_A1_EquilibriumVer1_kernel(float* Value, bool* InGameSet, int rDim, int qDim,
										  float* MaxInRaw, bool* A)
{
	__shared__ float Value_SH [BLOCK_SIZE];
	__shared__ float MaxInRaw_SH [BLOCK_SIZE];
	__shared__ bool InGameSet_SH [BLOCK_SIZE];

	int k = 0;
	int m = 0;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int ty = threadIdx.y;

	//Adress vichisl'aemogo dannim processom elementa
	int ix = bx;
	int iy = by * blockDim.y * rDim + ty * rDim;

	bool inGameSet_IxIy = InGameSet[ix + iy];
	float Value_IxIy = Value[ix + iy];
	bool A_IxIy;

	if(inGameSet_IxIy)
	{
		A_IxIy = true;
		while(k < gridDim.y && A_IxIy)
		{
			Value_SH[ty] = Value[k * blockDim.y * rDim + ty * rDim + ix];
			MaxInRaw_SH[ty] = MaxInRaw[k * blockDim.y * rDim + ty * rDim];
			InGameSet_SH[ty] = InGameSet[k * blockDim.y * rDim + ty * rDim + ix];

//			__syncthreads ();

			while(m < blockDim.y && A_IxIy)
			{
				if(InGameSet_SH[m] && (Value_SH[m] < Value_IxIy))
					if(Value_IxIy > MaxInRaw_SH[m]) //Esli perviy mozhet beznakazanno uluchshit' svoy rezul'tat
					{
						A_IxIy = false; //tochka ne yavl'aetsa ravnovesnoy
					}
				m++;
			}
			m = 0;
			k++;

//			__syncthreads (); //znachenie _SH massivov ne dolzhno men'atsa poka vse ne zakonchat
		}
	}
	else
		A_IxIy = false;

	A[ix + iy] = A_IxIy;
}


//Tochnost' kak u CPU, no Samiy madlenniy sredi GPU-algoritmov 
extern "C" __global__ void find_A1_EquilibriumVer2_kernel(float* Value, bool* InGameSet, int rDim, int qDim,
										  float* MaxInRaw, bool* A)
{
	__shared__ float Value_SH [BLOCK_SIZE];
	__shared__ float MaxInRaw_SH [BLOCK_SIZE];
	__shared__ bool InGameSet_SH [BLOCK_SIZE];
	__shared__ int finishedThreads;

	bool isFinished = false;

	int k = 0;
	int m = 0;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int ty = threadIdx.y;

	//Adress vichisl'aemogo dannim processom elementa
	int ix = bx;
	int iy = by * blockDim.y * rDim + ty * rDim;

	bool inGameSet_IxIy = InGameSet[ix + iy];
	float Value_IxIy = Value[ix + iy];
	bool A_IxIy = true;

	if(ty == 0)
		finishedThreads = 0;

	if(!inGameSet_IxIy)
	{
		A_IxIy = false;
		finishedThreads++;  //emu-debug gl'uchit s etim!!!
		isFinished = true;
	}

	__syncthreads (); //chtobi cikl ne vipoln'als'a esli vse processi vne game set
	if(ty == 0)
		printf("fin %d ", finishedThreads);

	while(k < gridDim.y && finishedThreads < blockDim.y)
	{
		Value_SH[ty] = Value[k * blockDim.y * rDim + ty * rDim + ix];
		MaxInRaw_SH[ty] = MaxInRaw[k * blockDim.y * rDim + ty * rDim];
		InGameSet_SH[ty] = InGameSet[k * blockDim.y * rDim + ty * rDim + ix];

		__syncthreads ();

		if(!isFinished)
		{
			while(m < blockDim.y && A_IxIy)
			{
				if(InGameSet_SH[m] && (Value_SH[m] < Value_IxIy))
					if(Value_IxIy > MaxInRaw_SH[m]) //Esli perviy mozhet beznakazanno uluchshit' svoy rezul'tat
					{
						A_IxIy = false; //tochka ne yavl'aetsa ravnovesnoy
						finishedThreads++;
						isFinished = true;
					}
				m++;
			}
			m = 0;
		}
		k++;

		__syncthreads (); //znachenie _SH massivov ne dolzhno men'atsa poka vse ne zakonchat
	}

	A[ix + iy] = A_IxIy;
}


//---------------------------A-ravnovesie dl'a 2go igroka---------------------------------------------
//Nahodim MinInCol
extern "C" __global__ void findMinInColReduce_kernel(int rDim, bool* inputInGS, bool* outputInGS,
										  float* inputData, float* outputData)
{
	__shared__ float data [BLOCK_SIZE];
	__shared__ bool inGameSet [BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = threadIdx.x;
	
	//Adres, kopiruemogo dannim processom elementa
	int ix = bx * blockDim.x + tid;
	int iy = by * rDim;
	
	inGameSet[tid] = inputInGS[ix + iy];
	if(inGameSet[tid])
		data[tid] = inputData[ix + iy];     // load into shared memeory
	else
		data[tid] = FLT_MIN;  //Esli ne v GameSet, iniciliziruem MIN_FLT,
							  //chtobi on v l'ubom sluchae ne stal Max

	__syncthreads ();

	for ( int s = blockDim.x / 2; s > 0; s >>= 1 )
    {
		if(tid < s)
		{
			data[tid] = MIN(data[tid], data[tid + s]);
			inGameSet[tid] = inGameSet[tid] || inGameSet[tid + s];
		}

		__syncthreads ();
	}

	if ( tid == 0 )                 // write result of block reduction
	{
		outputData[by * rDim + bx] = data [0];
		outputInGS[by * rDim + bx] = inGameSet[0];
	}
}

extern "C" __global__ void findMinInColFinish_kernel(int rDim, bool* inputInGS, bool* outputInGS,
										  float* inputData, float* outputData)
{
	__shared__ float data [BLOCK_SIZE];
	__shared__ bool inGameSet [BLOCK_SIZE];

	int bx = blockIdx.x;
	int by = blockIdx.y;
	int tid = threadIdx.x;
	
	//Adres, kopiruemogo dannim processom elementa
	int ix = bx * blockDim.x + tid;
	int iy = by * rDim;
	
	inGameSet[tid] = inputInGS[ix + iy];
	if(inGameSet[tid])
		data[tid] = inputData[ix + iy];     // load into shared memeory
	else
		data[tid] = FLT_MIN;  //Esli ne v GameSet, iniciliziruem MIN_FLT,
							  //chtobi on v l'ubom sluchae ne stal Max

	__syncthreads ();

	if ( tid == 0 )                 
	{
		for ( int s = 1 ; s < blockDim.x; s++ )
	    {
				data[tid ] = MIN (data[tid ], data[tid + s]);
				inGameSet[tid] = inGameSet[tid] || inGameSet[tid + s];
	
			__syncthreads ();
		}

		outputData[by * rDim + bx] = data [0];
		outputInGS[by * rDim + bx] = inGameSet[0];
	}
}

extern "C" __global__ void find_A2_Equilibrium_kernel(float* Value, bool* InGameSet, int rDim, int qDim,
										  float* MaxInRaw, bool* A)
{
	int k = 0;
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int ty = threadIdx.y;

	//Adress vichisl'aemogo dannim processom elementa
	int ix = bx;
	int iy = by * blockDim.y * rDim + ty * rDim;

	if(InGameSet[ix + iy])
	{
		A[ix + iy] = true;
		while(k < qDim && A[ix + iy])
		{
			if(InGameSet[k * rDim + ix] && (Value[k * rDim + ix] < Value[ix + iy]))
				if(Value[ix + iy] > MaxInRaw[k * rDim]) //Esli perviy mozhet beznakazanno uluchshit' svoy rezul'tat
				{
					A[ix + iy] = false; //tochka ne yavl'aetsa ravnovesnoy
				}
			k++;
		}
	}
	else
		A[ix + iy] = false;
}

//-------------------------B1-ravnovesie-------------------------------------------
extern "C" __global__ void find_B1_EquilibriumVer0_kernel(float* Value, bool* A, int rDim, int qDim,
										  float* MaxInRawOnA, bool* B)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int ty = threadIdx.y;

	//Adress vichisl'aemogo dannim processom elementa
	int ix = bx;
	int iy = by * blockDim.y * rDim + ty * rDim;

	//Esli danniy element yavl'aetsa luchshim dl'a vtorogo(r) sredi
	//vseh elementov dannoy stroki, vhod'ashih v Aq pervogo
	//To on prinadlezhit Bq
	if(A[ix + iy] && (Value[ix + iy] == MaxInRawOnA[iy]))
	{
		B[ix + iy] = true;
	}
	else
		B[ix + iy] = false;
}

//------------------------C1-ravnovesie---------------------------------------
extern "C" __global__ void find_C1_EquilibriumVer0_kernel(float* Value, bool* A, int rDim, int qDim,
										  float* MaxInRawOnG, bool* C)
{
	int bx = blockIdx.x;
	int by = blockIdx.y;
	int ty = threadIdx.y;

	//Adress vichisl'aemogo dannim processom elementa
	int ix = bx;
	int iy = by * blockDim.y * rDim + ty * rDim;

	//Esli danniy element yavl'aetsa luchshim dl'a vtorogo(r) sredi
	//vseh elementov dannoy stroki, vhod'ashih v G,
	//to on prinadlezhit Cq
	if(A[ix + iy] && (Value[ix + iy] == MaxInRawOnG[iy]))
	{
		C[ix + iy] = true;
	}
	else
		C[ix + iy] = false;
}
