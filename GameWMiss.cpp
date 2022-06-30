#include"define.h"
#include<stdlib.h>
#include<stdio.h>
#include <ctime>

float J(float r, float q)
{
	return (q - r) * (q - r);
//	return (q / r);
}

bool pointInGameSet(float r, float q)
{
	return ((q <= -r + 1) && (q >= r - 1) && (q >= -1) && (r >= -1) && (q <= 1));
/*	return ((q <= 1) && (r <= 1) && (q >= -r + 1.5) || (q >= -1) && (r <= 1) && (q <= r - 1.5) ||
			(q >= -1) && (r >= -1) && (q <= -r - 1.5) || (q <= 1) && (r >= -1) && (q >= r + 1.5)); 
*/
}

//------------------------------Vvod Dannih-----------------------------------------------
void inputData(float** Bound)
{
	(*Bound) = (float*)malloc(4 * sizeof(float));

	printf("\nEnter maximum r-coordinate: ");
	scanf("%f", (*Bound));
	printf("Enter minimum r-coordinate: ");
	scanf("%f", (*Bound + 1));
	if((*Bound)[0] <= (*Bound)[1])
	{
		int c;
		printf("Error: maximum coordinate <= minimum coordinate\n Enter any key to exit: ");
		scanf("%d", &c);
		exit(1);
	}
	printf("Enter maximum q-coordinate: ");
	scanf("%f", (*Bound + 2));
	printf("Enter minimum q-coordinate: ");
	scanf("%f", (*Bound + 3));
	if((*Bound)[2] <= (*Bound)[3])
	{
		int c;
		printf("Error: maximum coordinate <= minimum coordinate\n Enter any key to exit: ");
		scanf("%d", &c);
		exit(1);
	}
}

extern "C" bool* allocateBoolMatrixQR(int qDim, int rDim)
{
	bool* A;

	A = (bool*)malloc(rDim * qDim * sizeof(bool));
	if(A == NULL)
		printf("Pointer Error in bool-matrix allocation\n");

	return A;
}

//--------------------------------------Postroenie ravnovesiy------------------------------

//Horosho bi peredavat' s'uda ukazatel' na funkciyu J
gameMatrix findGameMatrix(float* Bound, float eps)
{
	gameMatrix GM;

	int temp;
	int delta;

	temp = (int)((Bound[0] - Bound[1]) / eps + 1);
	delta = BLOCK_SIZE - (temp % BLOCK_SIZE);
	GM.rDim = temp + delta; //razmernost' po r kratna BLOCK_SIZE
	Bound[0] += delta * eps; //Uvelichivaem MaxR v sootvetstvii s dobavlennimi elementami matrici

	temp = (int)((Bound[2] - Bound[3]) / eps + 1);
	delta = BLOCK_SIZE - (temp % BLOCK_SIZE);
	GM.qDim = temp + delta; //razmernost' po q kratna BLOCK_SIZE
	Bound[3] -= delta * eps; //Umen'shaem MinQ v sootvetstvii s dobavlennimi elementami matrici
	if(GM.rDim < 0 || GM.qDim < 0)
	{
		int c;
		printf("Error: maximum coordinate > minimum coordinate\n Enter any key to exit: ");
		scanf("%d", &c);
		exit(1);
	}

	printf("eps = %f\n", eps);
	printf("rDim: %d, qDim: %d\n", GM.rDim, GM.qDim);
	
	float MinR = Bound[1];
	float MaxQ = Bound[2];

	GM.Value = (float*)malloc(GM.rDim * GM.qDim * sizeof(float));
	GM.InGameSet = (bool*)malloc(GM.rDim * GM.qDim * sizeof(bool));

	for(int i = 0; i < GM.qDim; i++)
		for(int j = 0; j < GM.rDim; j++)
			if(pointInGameSet(MinR + j * eps, MaxQ - i * eps))
			{
				GM.Value[i * GM.rDim + j] = J(MinR + j * eps, MaxQ - i * eps);
				GM.InGameSet[i * GM.rDim + j] = true;
			}
			else
			{
				GM.Value[i * GM.rDim + j] = 0; //zanul'aem elementi, ne vxod'ashie v game set
				GM.InGameSet[i * GM.rDim + j] = false;
			}

	return GM;
}

bool* find_A_EquilibriumWMiss(int player, gameMatrix GM)
{
	bool* A;
	float* MinSet;
	float* MaxSet;

	A = allocateBoolMatrixQR(GM.qDim, GM.rDim);
	
	switch(player)
	{
		case 1: //Perviy(q) hodit po stolbtsam i minimiziryet
			MaxSet = (float*)malloc(GM.qDim * sizeof(float)); 
			if(MaxSet == NULL)
				printf("Pointer error in A-eq\n");
			for(int i = 0; i < GM.qDim; i++)//nahodim max v strokah
			{
				float max = 0;
				int j = 0;
				while(j < GM.rDim && !(GM.InGameSet[i * GM.rDim + j]))
					j++;
				if(j < GM.rDim)//v stroke nashels'a hot'a bi odin element iz game set
				{
					max = GM.Value[i * GM.rDim + j];
					while(j < GM.rDim)
					{
						if(GM.InGameSet[i * GM.rDim + j] && (GM.Value[i * GM.rDim + j] > max))
							max = GM.Value[i * GM.rDim + j];
						j++;
					}
				}
				MaxSet[i] = max;
			}

			for(int i = 0; i < GM.qDim; i++)
				for(int j = 0; j < GM.rDim; j++)
					if(GM.InGameSet[i * GM.rDim + j])
					{
						A[i * GM.rDim + j] = true;
						int k = 0;
						while(k < GM.qDim && A[i * GM.rDim + j])
						{
							if(GM.InGameSet[k * GM.rDim + j] && (GM.Value[k * GM.rDim + j] < GM.Value[i * GM.rDim + j]))
								if(GM.Value[i * GM.rDim + j] > MaxSet[k]) //Esli perviy mozhet beznakazanno uluchshit' svoy rezul'tat
									A[i * GM.rDim + j] = false; //tochka ne yavl'aetsa ravnovesnoy
							k++;
						}
					}
					else
						A[i * GM.rDim + j] = false;
			free(MaxSet);
			break;

		case 2: //Vtoroy(r) hodit po strokam i maximiziruet
			MinSet = (float*)malloc(GM.rDim * sizeof(float));
			if(MinSet == NULL)
				printf("Pointer error in A-eq\n");
			for(int j = 0; j < GM.rDim; j++)//nahodim min v stolbcah
			{
				float min = 0;
				int i = 0;
				while(i < GM.qDim && !(GM.InGameSet[i * GM.rDim + j]))
					i++;
				if(i < GM.qDim)//v stolbce nashels'a hot'a bi odin element iz game set
				{
					min = GM.Value[i * GM.rDim + j];
					while(i < GM.qDim)
					{
						if(GM.InGameSet[i * GM.rDim + j] && (GM.Value[i * GM.rDim + j] < min))
							min = GM.Value[i * GM.rDim + j];
						i++;
					}
				}
				MinSet[j] = min;
			}

			for(int i = 0; i < GM.qDim; i++)
				for(int j = 0; j < GM.rDim; j++)
					if(GM.InGameSet[i * GM.rDim + j])
					{
						A[i * GM.rDim + j] = true;
						int k = 0;
						while(k < GM.rDim && A[i * GM.rDim + j])
						{
							if(GM.InGameSet[i * GM.rDim + k] && (GM.Value[i * GM.rDim + k] > GM.Value[i * GM.rDim + j]))
								if(GM.Value[i * GM.rDim + j] < MinSet[k]) //Esli vtoroy mozhet beznakazanno uluchshit' svoy rezul'tat
									A[i * GM.rDim + j] = false; //tochka ne yavl'aetsa ravnovesnoy
							k++;
						}
					}
					else
						A[i * GM.rDim + j] = false;
			free(MinSet);
			break;
	}

	return A;
}

bool* find_B_EquilibriumWMiss(int player, bool* A, gameMatrix GM)
{
	bool* B;

	B = allocateBoolMatrixQR(GM.qDim, GM.rDim);
	
	switch(player)
	{
		case 1: //Perviy(q) predlagaet na svoih Aq vtoromu vibrat' luchshie dl'z seb'a - max v strokah
			for(int i = 0; i < GM.qDim; i++)
			{
				//V kazhdoy stroke nahodim max sredi vhod'ashih v A[i * GM.rDim + j]
				float max = 0;
				int j = 0;
				while(j < GM.rDim && !(A[i * GM.rDim + j]))
					j++;
				if(j < GM.rDim)//v stroke nashels'a hot'a bi odin element iz A
				{
					max = GM.Value[i * GM.rDim + j];
					while(j < GM.rDim)
					{
						if(A[i * GM.rDim + j] && (GM.Value[i * GM.rDim + j] > max))
							max = GM.Value[i * GM.rDim + j];
						j++;
					}
				}
				//Vse max v strokah sredi Aq vhod'at v Bq
				for(int j = 0; j < GM.rDim; j++)
				{
					B[i * GM.rDim + j] = false;
					if(A[i * GM.rDim + j] && GM.Value[i * GM.rDim + j] == max)
						B[i * GM.rDim + j] = true;
				}
			}
			break;
		case 2: //Vtoroy(r) predlagaet na svoem Ar pervomu(q) vibrat' nailuchshie dl/a seb'a 
			for(int j = 0; j < GM.rDim; j++)
			{
				float min = 0;
				int i = 0;
				while(i < GM.qDim && !(A[i * GM.rDim + j]))
					i++;
				if(i < GM.qDim)//v stolbce nashels'a hot'a bi odin element iz A
				{
					min = GM.Value[i * GM.rDim + j];
					while(i < GM.qDim)
					{
						if(A[i * GM.rDim + j] && (GM.Value[i * GM.rDim + j] < min))
							min = GM.Value[i * GM.rDim + j];
						i++;
					}
				}
				//Naimen'shie v stolbcah na Ar vhod'at v Br
				for(int i = 0; i < GM.qDim; i++)
				{
					B[i * GM.rDim + j] = false;
					if(A[i * GM.rDim + j] && GM.Value[i * GM.rDim + j] == min)
						B[i * GM.rDim + j] = true;
				}
			}
			break;
	}
	return B;
}

//Horosho bi sohran'at' max i min v strokah< podschitannie pri nahozhdenii A
//i ispol'zovat' zdes', ne schitaya zanovo

bool* find_C_EquilibriumWMiss(int player, bool* A, gameMatrix GM)
{
	bool* C;
	float* MaxInRaw;
	float* MinInCol;

	C = allocateBoolMatrixQR(GM.qDim, GM.rDim);
	
	switch(player)
	{
		case 1: //Ishem luchshie dl'a vtorogo - max v strokah na game set
			MaxInRaw = (float*)malloc(GM.qDim * sizeof(float)); 
			if(MaxInRaw == NULL)
				printf("Pointer error in C-eq\n");
			for(int i = 0; i < GM.qDim; i++)//nahodim max v strokah
			{
				float max = 0;
				int j = 0;
				while(j < GM.rDim && !(GM.InGameSet[i * GM.rDim + j]))
					j++;
				if(j < GM.rDim)//v stroke nashels'a hot'a bi odin element iz game set
				{
					max = GM.Value[i * GM.rDim + j];
					while(j < GM.rDim)
					{
						if(GM.InGameSet[i * GM.rDim + j] && (GM.Value[i * GM.rDim + j] > max))
							max = GM.Value[i * GM.rDim + j];
						j++;
					}
				}
				MaxInRaw[i] = max;
			}
			for(int i = 0; i < GM.qDim; i++)
			{
				//Vse max v strokah sredi Aq vhod'at v Bq
				for(int j = 0; j < GM.rDim; j++)
				{
					C[i * GM.rDim + j] = false;
					if(A[i * GM.rDim + j] && GM.Value[i * GM.rDim + j] == MaxInRaw[i])
						C[i * GM.rDim + j] = true;
				}
			}
			free(MaxInRaw);
			break;

		case 2: //Vtoroy(r) predlagaet na svoem Ar pervomu(q) vibrat' nailuchshie dl/a seb'a 
			MinInCol = (float*)malloc(GM.rDim * sizeof(float));
			if(MinInCol == NULL)
				printf("Pointer error in C-eq\n");
			for(int j = 0; j < GM.rDim; j++)//nahodim min v stolbcah
			{
				float min = 0;
				int i = 0;
				while(i < GM.qDim && !(GM.InGameSet[i * GM.rDim + j]))
					i++;
				if(i < GM.qDim)//v stolbce nashels'a hot'a bi odin element iz game set
				{
					min = GM.Value[i * GM.rDim + j];
					while(i < GM.qDim)
					{
						if(GM.InGameSet[i * GM.rDim + j] && (GM.Value[i * GM.rDim + j] < min))
							min = GM.Value[i * GM.rDim + j];
						i++;
					}
				}
				MinInCol[j] = min;
			}
			for(int j = 0; j < GM.rDim; j++)
			{
				//Naimen'shie v stolbcah na Ar vhod'at v Br
				for(int i = 0; i < GM.qDim; i++)
				{
					C[i * GM.rDim + j] = false;
					if(A[i * GM.rDim + j] && GM.Value[i * GM.rDim + j] == MinInCol[j])
						C[i * GM.rDim + j] = true;
				}
			}
			free(MinInCol);
			break;
	}
	return C;
}

extern void findEquilibriumWMissPTest(char eqName, gameMatrix GM, bool* A, int iteration_num)
{
	bool** Eq;

	clock_t t0, t1;
	float cpuTime;

	Eq = (bool**)malloc(iteration_num * sizeof(bool*));

	switch(eqName)
	{
	case 'A':
		t0 = clock();
		for(int i = 0; i < iteration_num; i++)
		{
			Eq[i] = find_A_EquilibriumWMiss(1, GM);
		}
		t1 = clock();
		cpuTime = (float)(t1 - t0)/(iteration_num);
		break;

	case 'B':
		t0 = clock();
		for(int i = 0; i < iteration_num; i++)
		{
			Eq[i] = find_B_EquilibriumWMiss(1, A, GM);
		}
		t1 = clock();
		cpuTime = (float)(t1 - t0)/(iteration_num);
		break;
	}

    printf("Time spent executing %c-eq by the CPU: %f \n", eqName, cpuTime);

	for(int i = 0; i < iteration_num; i++)
	{
		free(Eq[i]);
	}


}

//---------------------------------------Pechat'-----------------------------------
void printGameMatrix(gameMatrix GM)
{
	printf("\n");
	for(int i = 0; i < GM.qDim; i++)
	{
		for(int j = 0; j < GM.rDim; j++)
			if(GM.InGameSet[i * GM.rDim + j])
				printf("%.2f ", GM.Value[i * GM.rDim + j]);
			else
				printf("*.** ");

		printf("\n");
	}
}

void printBoolMatrixQR(int qDim, int rDim, bool*A)
{
	printf("\n");
	for(int i = 0; i < qDim; i++)
	{
		for(int j = 0; j < rDim; j++)
			printf("%d ", A[i * rDim + j]);
		printf("\n");
	}
}

//----------------------------------Osvobozhdenie pam'ati-------------------------
void freeGameMatrix(gameMatrix GM)
{
	free(GM.Value);
	free(GM.InGameSet);
}