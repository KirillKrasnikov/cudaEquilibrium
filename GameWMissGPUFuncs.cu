#include <stdio.h>
#include <stdlib.h>
#include "define.h"

//--------------------------------------Global variables-----------------------------------
bool firstCall = true;

float* ValueDev = NULL;
bool* InGameSetDev = NULL;

bool* AqForGPU;
bool* BqForGPU;
bool* CqForGPU;

//--------------------------------External--functions----------------------------------------
extern "C" __global__ void findMaxInRawReduce_kernel(int rDim, bool* inputInGS, bool* outputInGS,
													 float* inputData, float* outputData);
extern "C" __global__ void findMaxInRawFinish_kernel(int rDim, bool* inputInGS, bool* outputInGS,
										  float* inputData, float* outputData);
extern "C" __global__ void find_A1_EquilibriumVer0_kernel(float* Value, bool* InGameSet, int rDim, int qDim,
										  float* MaxInRaw, bool* A);
extern "C" __global__ void find_A1_EquilibriumVer1_kernel(float* Value, bool* InGameSet, int rDim, int qDim,
										  float* MaxInRaw, bool* A);
extern "C" __global__ void find_A1_EquilibriumVer2_kernel(float* Value, bool* InGameSet, int rDim, int qDim,
										  float* MaxInRaw, bool* A);
extern "C" __global__ void find_B1_EquilibriumVer0_kernel(float* Value, bool* A, int rDim, int qDim,
										  float* MaxInRawOnA, bool* B);
extern "C" __global__ void find_C1_EquilibriumVer0_kernel(float* Value, bool* A, int rDim, int qDim,
										  float* MaxInRawOnG, bool* C);
//-------------------------------------------------------------------------------------------


extern "C" void freeGameMatrixGPU(gameMatrix GMDev)
{
	cudaFree(GMDev.Value);
	cudaFree(GMDev.InGameSet);
}

extern "C" bool* find_A_EquilibriumGPU(gameMatrix GM, float* ValueDev, bool* InGameSetDev)
{
	int elem_num = GM.rDim;
	int i = 0;
	int q = 0;

	bool* A;
	bool* ADev;

	float* ValDev[2] = {NULL, NULL};
	bool* InGSDev[2] = {NULL, NULL};

	float* MaxInRawDev;
	float* MaxInRawHost;
	

	A = (bool*)malloc(GM.qDim * GM.rDim * sizeof(bool));
	cudaMalloc((void**)&ADev, GM.qDim * GM.rDim * sizeof(bool));

	//-----------Nahodim--MaxInRaw------------------------------------
	cudaMalloc((void**)&(ValDev[0]), GM.qDim * GM.rDim * sizeof(float));
	cudaMalloc((void**)&(ValDev[1]), GM.qDim * GM.rDim * sizeof(float));
	cudaMalloc((void**)&(InGSDev[0]), GM.qDim * GM.rDim * sizeof(bool));
	cudaMalloc((void**)&(InGSDev[1]), GM.qDim * GM.rDim * sizeof(bool));

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate ( &start);
    cudaEventCreate ( &stop );

    cudaEventRecord ( start, 0 );
	cudaMemcpy(ValDev[0], GM.Value, GM.qDim * GM.rDim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(InGSDev[0], GM.InGameSet, GM.qDim * GM.rDim * sizeof(bool), cudaMemcpyHostToDevice);
	
	for ( i = 0; elem_num >= BLOCK_SIZE; elem_num /= (BLOCK_SIZE), i ^= 1 )
	{
                              // set kernel launch configuration
		dim3 dimBlock ( BLOCK_SIZE, 1, 1 );
		dim3 dimGrid  ( elem_num / (dimBlock.x), GM.qDim, 1 );

		findMaxInRawReduce_kernel<<<dimGrid, dimBlock>>>(GM.rDim, InGSDev[i], InGSDev[i^1], ValDev[i], ValDev[i^1]);
/*
		printf("GPU-mtrix:\n");
		for(int i = 0; i < GM.qDim; i++)
		{
			for(int j = 0; j < GM.rDim; j++)
				if(InGSDev[i^1][i * GM.rDim + j])
					printf("%.1f ", ValDev[i^1][i * GM.rDim + j]);
				else
					printf("*.* ");
			printf("\n");
		}
*/


	}
	//Esli posle redukcii kol-vo elemrntov, sredi kotorih nushno
	//naiti max bol'she 1 (no men'she BLOCK_SIZE) to provodiem
	//eshe odnu iteraciyu
	if(elem_num > 1)
	{
		dim3 dimBlock2 ( elem_num, 1, 1 );
		dim3 dimGrid2  ( 1, GM.qDim, 1 );

		findMaxInRawFinish_kernel<<<dimGrid2, dimBlock2>>>(GM.rDim, InGSDev[i], InGSDev[i^1], ValDev[i], ValDev[i^1]);

		MaxInRawDev = ValDev[i^1];
	}
	else
		MaxInRawDev = ValDev[i];

	//Naxodim Ravnovesie
	dim3 dimBlock3 ( 1, BLOCK_SIZE, 1 );
	dim3 dimGrid3  ( GM.rDim, GM.qDim / dimBlock3.y,  1 );

	find_A1_EquilibriumVer0_kernel<<<dimGrid3, dimBlock3>>>(ValueDev, InGameSetDev, GM.rDim, GM.qDim, MaxInRawDev, ADev);

	cudaMemcpy(A, ADev, GM.qDim * GM.rDim * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaEventRecord ( stop, 0 );

	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &gpuTime, start, stop);

    cudaEventDestroy ( start );
    cudaEventDestroy ( stop );

	MaxInRawHost = (float*)malloc(GM.qDim * GM.rDim * sizeof(float));
	cudaMemcpy(MaxInRawHost, MaxInRawDev, GM.qDim * GM.rDim * sizeof(float), cudaMemcpyDeviceToHost);

/*	printf("CUDA Max in Raws: \n");
	for(q = 0; q < GM.qDim; q++)
		printf("Max[%d] = %.2f\n", q, MaxInRawHost[q * GM.rDim]);
*/

	printf("Time spent exequting A-eq on GPU: %fms\n", gpuTime);
//	printf("Game Matrix dimension: %d * %d\n", GM.qDim, GM.rDim);

/*
	printf("CUDA A-eq:\n");
	for(int i = 0; i < GM.qDim; i++)
	{
		for(int j = 0; j < GM.rDim; j++)
			printf("%d ", A[i * GM.rDim + j]);
		printf("\n");
	}
*/
	free(MaxInRawHost);

	cudaFree(ValDev[0]);
	cudaFree(ValDev[1]);
	cudaFree(InGSDev[0]);
	cudaFree(InGSDev[1]);
	cudaFree(ADev);

	return A;
}

//--------------Nahodim-B1-ravnovesie-------------------------------------
extern "C" bool* find_B_EquilibriumGPU(gameMatrix GM, float* ValueDev, bool* A)
{
	int elem_num = GM.rDim;
	int i = 0;
	int q = 0;

	bool* B;
	bool* BDev;

	float* ValTempDev[2] = {NULL, NULL};
	bool* ATempDev[2] = {NULL, NULL};
	bool* ADev;

	float* MaxInRawOnA_Dev;
	float* MaxInRawOnA_Host;
	

	B = (bool*)malloc(GM.qDim * GM.rDim * sizeof(bool));
	cudaMalloc((void**)&BDev, GM.qDim * GM.rDim * sizeof(bool));

	cudaMalloc((void**)&ADev, GM.qDim * GM.rDim * sizeof(bool));
	cudaMemcpy(ADev, A, GM.qDim * GM.rDim * sizeof(bool), cudaMemcpyHostToDevice);

	//-----------Nahodim--MaxInRaw------------------------------------
	cudaMalloc((void**)&(ValTempDev[0]), GM.qDim * GM.rDim * sizeof(float));
	cudaMalloc((void**)&(ValTempDev[1]), GM.qDim * GM.rDim * sizeof(float));
	cudaMalloc((void**)&(ATempDev[0]), GM.qDim * GM.rDim * sizeof(bool));
	cudaMalloc((void**)&(ATempDev[1]), GM.qDim * GM.rDim * sizeof(bool));

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate ( &start);
    cudaEventCreate ( &stop );

    cudaEventRecord ( start, 0 );
	cudaMemcpy(ValTempDev[0], GM.Value, GM.qDim * GM.rDim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(ATempDev[0], A, GM.qDim * GM.rDim * sizeof(bool), cudaMemcpyHostToDevice);
	
	for ( i = 0; elem_num >= BLOCK_SIZE; elem_num /= (BLOCK_SIZE), i ^= 1 )
	{
                              // set kernel launch configuration
		dim3 dimBlock ( BLOCK_SIZE, 1, 1 );
		dim3 dimGrid  ( elem_num / (dimBlock.x), GM.qDim, 1 );

		findMaxInRawReduce_kernel<<<dimGrid, dimBlock>>>(GM.rDim, ATempDev[i], ATempDev[i^1], ValTempDev[i], ValTempDev[i^1]);
/*
		printf("GPU-mtrix:\n");
		for(int i = 0; i < GM.qDim; i++)
		{
			for(int j = 0; j < GM.rDim; j++)
				if(InGSDev[i^1][i * GM.rDim + j])
					printf("%.1f ", ValTempDev[i^1][i * GM.rDim + j]);
				else
					printf("*.* ");
			printf("\n");
		}
*/


	}
	//Esli posle redukcii kol-vo elemrntov, sredi kotorih nushno
	//naiti max bol'she 1 (no men'she BLOCK_SIZE) to provodiem
	//eshe odnu iteraciyu
	if(elem_num > 1)
	{
		dim3 dimBlock2 ( elem_num, 1, 1 );
		dim3 dimGrid2  ( 1, GM.qDim, 1 );

		findMaxInRawFinish_kernel<<<dimGrid2, dimBlock2>>>(GM.rDim, ATempDev[i], ATempDev[i^1], ValTempDev[i], ValTempDev[i^1]);

		MaxInRawOnA_Dev = ValTempDev[i^1];
	}
	else
		MaxInRawOnA_Dev = ValTempDev[i];

	//Naxodim Ravnovesie
	dim3 dimBlock3 ( 1, BLOCK_SIZE, 1 );
	dim3 dimGrid3  ( GM.rDim, GM.qDim / dimBlock3.y,  1 );

	find_B1_EquilibriumVer0_kernel<<<dimGrid3, dimBlock3>>>(ValueDev, ADev, GM.rDim, GM.qDim, MaxInRawOnA_Dev, BDev);

	cudaMemcpy(B, BDev, GM.qDim * GM.rDim * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaEventRecord ( stop, 0 );

	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &gpuTime, start, stop);

    cudaEventDestroy ( start );
    cudaEventDestroy ( stop );

	printf("Time spent exequting B-eq on GPU: %fms\n", gpuTime);

/*	MaxInRawHost = (float*)malloc(GM.qDim * GM.rDim * sizeof(float));
	cudaMemcpy(MaxInRawHost, MaxInRawDev, GM.qDim * GM.rDim * sizeof(float), cudaMemcpyDeviceToHost);

	printf("CUDA Max in Raws: \n");
	for(q = 0; q < GM.qDim; q++)
		printf("Max[%d] = %.2f\n", q, MaxInRawHost[q * GM.rDim]);

	free(MaxInRawHost);
*/

/*
	printf("CUDA B-eq:\n");
	for(int i = 0; i < GM.qDim; i++)
	{
		for(int j = 0; j < GM.rDim; j++)
			printf("%d ", B[i * GM.rDim + j]);
		printf("\n");
	}
*/

	cudaFree(ValTempDev[0]);
	cudaFree(ValTempDev[1]);
	cudaFree(ATempDev[0]);
	cudaFree(ATempDev[1]);
	cudaFree(ADev);
	cudaFree(BDev);

	return B;
}

//----------------------------------Nahodim-C1-ravnovesie-----------------------------------------
extern "C" bool* find_C_EquilibriumGPU(gameMatrix GM, float* ValueDev, bool* InGameSetDev, bool* A)
{
	int elem_num = GM.rDim;
	int i = 0;
	int q = 0;

	bool* C;
	bool* CDev;

	float* ValTempDev[2] = {NULL, NULL};
	bool* InGSTempDev[2] = {NULL, NULL};
	bool* ADev;

	float* MaxInRawOnG_Dev;
	float* MaxInRawOnG_Host;
	

	C = (bool*)malloc(GM.qDim * GM.rDim * sizeof(bool));
	cudaMalloc((void**)&CDev, GM.qDim * GM.rDim * sizeof(bool));

	cudaMalloc((void**)&ADev, GM.qDim * GM.rDim * sizeof(bool));
	cudaMemcpy(ADev, A, GM.qDim * GM.rDim * sizeof(bool), cudaMemcpyHostToDevice);

	//-----------Nahodim--MaxInRaw------------------------------------
	cudaMalloc((void**)&(ValTempDev[0]), GM.qDim * GM.rDim * sizeof(float));
	cudaMalloc((void**)&(ValTempDev[1]), GM.qDim * GM.rDim * sizeof(float));
	cudaMalloc((void**)&(InGSTempDev[0]), GM.qDim * GM.rDim * sizeof(bool));
	cudaMalloc((void**)&(InGSTempDev[1]), GM.qDim * GM.rDim * sizeof(bool));

    cudaEvent_t start, stop;
    float gpuTime = 0.0f;

    cudaEventCreate ( &start);
    cudaEventCreate ( &stop );

    cudaEventRecord ( start, 0 );
	cudaMemcpy(ValTempDev[0], GM.Value, GM.qDim * GM.rDim * sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(InGSTempDev[0], GM.InGameSet, GM.qDim * GM.rDim * sizeof(bool), cudaMemcpyHostToDevice);
	
	for ( i = 0; elem_num >= BLOCK_SIZE; elem_num /= (BLOCK_SIZE), i ^= 1 )
	{
                              // set kernel launch configuration
		dim3 dimBlock ( BLOCK_SIZE, 1, 1 );
		dim3 dimGrid  ( elem_num / (dimBlock.x), GM.qDim, 1 );

		findMaxInRawReduce_kernel<<<dimGrid, dimBlock>>>(GM.rDim, InGSTempDev[i], InGSTempDev[i^1], ValTempDev[i], ValTempDev[i^1]);
/*
		printf("GPU-mtrix:\n");
		for(int i = 0; i < GM.qDim; i++)
		{
			for(int j = 0; j < GM.rDim; j++)
				if(InGSDev[i^1][i * GM.rDim + j])
					printf("%.1f ", ValTempDev[i^1][i * GM.rDim + j]);
				else
					printf("*.* ");
			printf("\n");
		}
*/


	}
	//Esli posle redukcii kol-vo elemrntov, sredi kotorih nushno
	//naiti max bol'she 1 (no men'she BLOCK_SIZE) to provodiem
	//eshe odnu iteraciyu
	if(elem_num > 1)
	{
		dim3 dimBlock2 ( elem_num, 1, 1 );
		dim3 dimGrid2  ( 1, GM.qDim, 1 );

		findMaxInRawFinish_kernel<<<dimGrid2, dimBlock2>>>(GM.rDim, InGSTempDev[i], InGSTempDev[i^1], ValTempDev[i], ValTempDev[i^1]);

		MaxInRawOnG_Dev = ValTempDev[i^1];
	}
	else
		MaxInRawOnG_Dev = ValTempDev[i];

	//Naxodim Ravnovesie
	dim3 dimBlock3 ( 1, BLOCK_SIZE, 1 );
	dim3 dimGrid3  ( GM.rDim, GM.qDim / dimBlock3.y,  1 );

	find_C1_EquilibriumVer0_kernel<<<dimGrid3, dimBlock3>>>(ValueDev, ADev, GM.rDim, GM.qDim, MaxInRawOnG_Dev, CDev);

	cudaMemcpy(C, CDev, GM.qDim * GM.rDim * sizeof(bool), cudaMemcpyDeviceToHost);
    cudaEventRecord ( stop, 0 );

	cudaEventSynchronize ( stop );
	cudaEventElapsedTime ( &gpuTime, start, stop);

    cudaEventDestroy ( start );
    cudaEventDestroy ( stop );

	printf("Time spent exequting C-eq on GPU: %fms\n", gpuTime);

/*	MaxInRawHost = (float*)malloc(GM.qDim * GM.rDim * sizeof(float));
	cudaMemcpy(MaxInRawHost, MaxInRawDev, GM.qDim * GM.rDim * sizeof(float), cudaMemcpyDeviceToHost);

	printf("CUDA Max in Raws: \n");
	for(q = 0; q < GM.qDim; q++)
		printf("Max[%d] = %.2f\n", q, MaxInRawHost[q * GM.rDim]);

	free(MaxInRawHost);
*/

/*
	printf("CUDA B-eq:\n");
	for(int i = 0; i < GM.qDim; i++)
	{
		for(int j = 0; j < GM.rDim; j++)
			printf("%d ", B[i * GM.rDim + j]);
		printf("\n");
	}
*/

	cudaFree(ValTempDev[0]);
	cudaFree(ValTempDev[1]);
	cudaFree(InGSTempDev[0]);
	cudaFree(InGSTempDev[1]);
	cudaFree(ADev);
	cudaFree(CDev);

	return C;
}
//-------------------------------------------------------------------------------------------------

extern "C" bool* findEquilibriumGPU(char eqName, int player, gameMatrix GM, bool needToRefreshData)
{
	bool* Eq;

	if(needToRefreshData)
	{
		cudaMalloc((void**)(&ValueDev), GM.qDim * GM.rDim * sizeof(float));
		cudaMalloc((void**)(&InGameSetDev), GM.qDim * GM.rDim * sizeof(bool));

		cudaMemcpy(ValueDev, GM.Value, GM.qDim * GM.rDim * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(InGameSetDev, GM.InGameSet, GM.qDim * GM.rDim * sizeof(bool), cudaMemcpyHostToDevice);
	}

	switch(eqName)
	{
	case 'A':
		AqForGPU = find_A_EquilibriumGPU(GM, ValueDev, InGameSetDev);
		return AqForGPU;
		break;

	case 'B':
		BqForGPU = find_B_EquilibriumGPU(GM, ValueDev, AqForGPU);
		return BqForGPU;
		break;

	case 'C':
		CqForGPU = find_C_EquilibriumGPU(GM, ValueDev, InGameSetDev, AqForGPU);
		return CqForGPU;
		break;
	}

}

extern "C" void freeDataGPU()
{
	cudaFree(ValueDev);
	cudaFree(InGameSetDev);
}

