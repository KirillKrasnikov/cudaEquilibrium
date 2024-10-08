#include<windows.h> 
#include<stdlib.h>
#include<stdio.h>
#include<ctime>
#include<gl/glut.h>
#include<gl/glu.h>
#include"define.h"


//-------------------------Global variables-------------------------------------
float eps = 0.1; //Znachenie po umolchaniyu
float* Bound;

gameMatrix GM;
gameMatrix GMDev;

bool* Aq;
bool* Ar;
bool* Bq;
bool* Br;
bool* Cq;
bool* Cr;

bool* AqGPU;
bool* ArGPU;
bool* BqGPU;
bool* BrGPU;
bool* CqGPU;
bool* CrGPU;

int render_type = 0;

//---------------------------------------------------------------------------------------
extern void inputData(float** Bound);
extern gameMatrix findGameMatrix(float* Bound, float eps);
extern bool* find_A_EquilibriumWMiss(int player, gameMatrix GM);
extern bool* find_B_EquilibriumWMiss(int player, bool* A, gameMatrix GM);
extern bool* find_C_EquilibriumWMiss(int player, bool* A, gameMatrix GM);
extern void printGameMatrix(gameMatrix GM);
extern void printBoolMatrixQR(int qDim, int rDim, bool*A);
extern void freeBoolMatrixMemoryQR(int qDim, int rDim, bool* A);
extern void freeGameMatrix(gameMatrix GM);
extern void findEquilibriumWMissPTest(char eqName, gameMatrix GM, bool* A, int iteration_num);
extern float J(float r, float q);

extern "C" bool* findEquilibriumGPU(char eqName, int player, gameMatrix GM, bool needToRefreshData);
extern "C" void freeDataGPU();
//--------------------------------Вывод текста-------------------------------------------------------


//--------------------------------------------------------------------------------------------------
void freeData()
{
	free(Aq);
	free(Ar);
	free(Bq);
	free(Br);
	free(Cq);
	free(Cr);
	freeGameMatrix(GM);
	//GPU:
	freeDataGPU();
	free(AqGPU);
	free(BqGPU);
	free(CqGPU);
}


//-------------------------------------OpenGL-----------------------------------------------------

void SetupRC()
{
	glClearColor(1.0, 1.0, 1.0, 1.0);
//  glEnable(GL_DEPTH_TEST);
//	glEnable(GL_LIGHTING); 

	glEnable(GL_ALPHA_TEST);
	glEnable(GL_BLEND);
	glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

//	BuildFont();										// Build The Font
}

void RenderGameSet()
{
	glColor4f(0.0, 0.0, 1.0, 0.2);
	for(int i = 0; i < GM.qDim; i++)
		for(int j = 0; j < GM.rDim; j++)
			if(GM.InGameSet[i * GM.rDim + j])
			{
				glRectf(Bound[1] + j * eps - (eps/2), Bound[2] - i * eps + (eps/2), Bound[1] + j * eps + (eps/2), Bound[2] - i * eps - (eps/2));
			}
}

void RenderEquilibrium(bool* A, float r, float g, float b, float alpha)
{
	glPushMatrix();
	glColor4f(r, g, b, alpha);
	for(int i = 0; i < GM.qDim; i++)
		for(int j = 0; j < GM.rDim; j++)
			if(A[i * GM.rDim + j])
			{
				glRectf(Bound[1] + j * eps - (eps/2), Bound[2] - i * eps + (eps/2), Bound[1] + j * eps + (eps/2), Bound[2] - i * eps - (eps/2));
			}
}

/*
void RenderFunction(void (*J)(float r, float q))
{
	glEnable(GL_DEPTH_TEST);
	
	glDisable(GL_DEPTH_TEST);
}
*/

void RenderString(float x, float y, void *font, char *string)
{  
	char *c;

	glRasterPos2f(x, y);
	for (c = string; *c != '\0'; c++)
		glutBitmapCharacter(font, *c);
}

void RenderItem(float x, float y, float z,  int r_type, char* text)
{
	glBegin(GL_QUADS);
		glVertex3f(x, y, 0.0);
		glVertex3f(x + 0.2, y, 0.0);
		glVertex3f(x + 0.2, y + 0.08, 0.0);
		glVertex3f(x, y + 0.08, 0.0);
	glEnd();
	if(render_type == r_type)
		glColor4f(0.0, 0.0, 0.0, 1.0);
	else
		glColor4f(0.6, 0.6, 0.6, 1.0);
	RenderString(x + 0.3, y + 0.01, GLUT_BITMAP_9_BY_15, text);
}

void RenderInterface()
{
	int type = (int)J(1,1);

	glPushMatrix();
	glTranslatef(0.1, 0.0, 0.0);

	glColor4f(0.0, 0.0, 0.0, 1.0);
	switch(type)
	{
	case 0:
		RenderString(1.5, 1.2, GLUT_BITMAP_9_BY_15, "Function: J = (q - r)^2");
		break;
	case 1:
		RenderString(1.5, 1.2, GLUT_BITMAP_9_BY_15, "Function: J =  q / r");
		break;
	}
	RenderString(1.5, 0.90, GLUT_BITMAP_9_BY_15, "Control:");

	glPushMatrix();
		glTranslatef(1.5, 0.74, 0.0);
		glColor4f(0.0, 0.0, 1.0, 0.2);
		RenderItem(0.0, 0.0, 0.0,  0, "Game set - 0");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(0.0, 1.0, 0.0, 0.3);
		RenderItem(0.0, 0.0, 0.0,  1, "Aq - 1");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(1.0, 0.0, 0.0, 0.3);
		RenderItem(0.0, 0.0, 0.0,  2, "Ar - 2");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(0.5, 0.5, 0.2, 0.5);
		RenderItem(0.0, 0.0, 0.0,  3, "Aq & Ar - 3");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(0.0, 1.0, 0.0, 0.5);
		RenderItem(0.0, 0.0, 0.0,  4, "Bq - 4");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(1.0, 0.0, 0.0, 0.5);
		RenderItem(0.0, 0.0, 0.0,  5, "Br - 5");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(0.5, 0.5, 0.2, 0.5);
		RenderItem(0.0, 0.0, 0.0,  6, "Bq & Br - 6");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(0.0, 1.0, 0.0, 0.7);
		RenderItem(0.0, 0.0, 0.0,  7, "Cq - 7");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(1.0, 0.0, 0.0, 0.7);
		RenderItem(0.0, 0.0, 0.0,  8, "Cr - 8");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(0.5, 0.5, 0.2, 0.7);
		RenderItem(0.0, 0.0, 0.0,  9, "Cq & Cr - 9");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(0.0, 1.0, 0.0, 0.3);
		RenderItem(0.0, 0.0, 0.0,  11, "CUDA Aq - q");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(0.0, 1.0, 0.0, 0.5);
		RenderItem(0.0, 0.0, 0.0,  14, "CUDA Bq - r");
		glTranslatef(0.0, -0.15, 0.0);
		glColor4f(0.0, 1.0, 0.0, 0.7);
		RenderItem(0.0, 0.0, 0.0,  17, "CUDA Cq - u");

		glPopMatrix();

	glPopMatrix();
}

void RenderScene()
{
	glClear(GL_COLOR_BUFFER_BIT);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
	glTranslatef(-0.5, 0.0, -6.0);


	glColor3f(0.0, 0.0, 0.0);
	glBegin(GL_LINES);
		//Os' Q
		glVertex3f(0.0, -1.5, 0.0);
		glVertex3f(0.0, 1.5, 0.0);
		glVertex3f(0.0, 1.5, 0.0);
		glVertex3f(0.04, 1.45, 0.0);
		glVertex3f(0.0, 1.5, 0.0);
		glVertex3f(-0.04, 1.45, 0.0);
		glVertex3f(-0.03, 1.0, 0.0);
		glVertex3f(0.03, 1.0, 0.0);
		glVertex3f(-0.03, -1.0, 0.0);
		glVertex3f(0.03, -1.0, 0.0);
		//Os' R
		glVertex3f(-1.5, 0.0, 0.0);
		glVertex3f(1.5, 0.0, 0.0);
		glVertex3f(1.5, 0.0, 0.0);
		glVertex3f(1.45, 0.04, 0.0);
		glVertex3f(1.5, 0.0, 0.0);
		glVertex3f(1.45, -0.04, 0.0);
		glVertex3f(1.0, -0.03, 0.0);
		glVertex3f(1.0, 0.03, 0.0);
		glVertex3f(-1.0, -0.03, 0.0);
		glVertex3f(-1.0, 0.03, 0.0);
	glEnd();

	RenderString(0.05, 1.42, GLUT_BITMAP_9_BY_15, "q");
	RenderString(0.05, 0.97, GLUT_BITMAP_9_BY_15, "1");
	RenderString(1.43, -0.13, GLUT_BITMAP_9_BY_15, "r");
	RenderString(0.97, -0.14, GLUT_BITMAP_9_BY_15, "1");


	RenderGameSet();

	RenderInterface();

	switch(render_type)
	{
		case 0:
			break;
		case 1:
			RenderEquilibrium(Aq, 0.0, 1.0, 0.0, 0.4);
			break;
		case 2:
			RenderEquilibrium(Ar, 1.0, 0.0, 0.0, 0.4);
			break;
		case 3:
			RenderEquilibrium(Ar, 1.0, 0.0, 0.0, 0.4);
			RenderEquilibrium(Aq, 0.0, 1.0, 0.0, 0.4);
			break;
		case 4:
			RenderEquilibrium(Bq, 0.0, 1.0, 0.0, 0.5);
			break;
		case 5:
			RenderEquilibrium(Br, 1.0, 0.0, 0.0, 0.5);
			break;
		case 6:
			RenderEquilibrium(Br, 1.0, 0.0, 0.0, 0.5);
			RenderEquilibrium(Bq, 0.0, 1.0, 0.0, 0.5);
			break;
		case 7:
			RenderEquilibrium(Cq, 0.0, 1.0, 0.0, 0.7);
			break;
		case 8:
			RenderEquilibrium(Cr, 1.0, 0.0, 0.0, 0.7);
			break;
		case 9:
			RenderEquilibrium(Cr, 1.0, 0.0, 0.0, 0.7);
			RenderEquilibrium(Cq, 0.0, 1.0, 0.0, 0.7);
			break;
		case 11:
			RenderEquilibrium(AqGPU, 0.0, 1.0, 0.0, 0.4);
			break;
		case 14:
			RenderEquilibrium(BqGPU, 0.0, 1.0, 0.0, 0.5);
			break;
		case 17:
			RenderEquilibrium(CqGPU, 0.0, 1.0, 0.0, 0.7);
			break;
	}


	glutSwapBuffers();
}

void KeyFunc(unsigned char key,int x,int y)
{
	switch(key)
	{
		case '1':
			render_type = 1;
			break;
		case '2':
			render_type = 2;
			break;
		case '3':
			render_type = 3;
			break;
		case '4':
			render_type = 4;
			break;
		case '5':
			render_type = 5;
			break;
		case '6':
			render_type = 6;
			break;
		case '7':
			render_type = 7;
			break;
		case '8':
			render_type = 8;
			break;
		case '9':
			render_type = 9;
			break;
		case '0':
			render_type = 0;
			break;
		case 'q':
			render_type = 11;
			break;
		case 'r':
			render_type = 14;
			break;
		case 'u':
			render_type = 17;
			break;
	}

	if(key == 27)
	{
		glutDestroyWindow(glutGetWindow());
		freeData(); //Osvobozhdaem pam'at'!!!!!!
		free(Bound);
		exit(1);
	}

	glutPostRedisplay();
}

void SpecialKeys(int key, int x, int y)
{
	if(key == GLUT_KEY_UP)
		eps = eps * 1.5;

	if(key == GLUT_KEY_DOWN)
		eps = eps / 1.5;

	freeData(); //Osvobozhdaem pam'at'!!!!!!
	GM = findGameMatrix(Bound, eps);
	Aq = find_A_EquilibriumWMiss(1, GM);
	Ar = find_A_EquilibriumWMiss(2, GM);
	Bq = find_B_EquilibriumWMiss(1, Aq, GM);
	Br = find_B_EquilibriumWMiss(2, Ar, GM);
	Cq = find_C_EquilibriumWMiss(1, Aq, GM);
	Cr = find_C_EquilibriumWMiss(2, Ar, GM);

	findEquilibriumWMissPTest('A', GM, Aq, ITERATION_NUM_CPU);

//	printf("\nGame matrix:");
//	printGameMatrix(GM);
	
	AqGPU = findEquilibriumGPU('A', 1, GM, 1);
	BqGPU = findEquilibriumGPU('B', 1, GM, 0);
	CqGPU = findEquilibriumGPU('C', 1, GM, 0);

	glutPostRedisplay();
}


void MouseFunc(int button, int state, int x, int y)
{
	if(x < 0.5 && y < 0.5)
	{
		render_type = 0;
	}
}

void ChangeSize(int w, int h)
{
	GLfloat fAspect;
	
	if(h == 0)
		h = 1;
	
	glViewport(0, 0, w, h);

	fAspect = (GLfloat)w / (GLfloat)h;

	glMatrixMode(GL_PROJECTION);
	glLoadIdentity();
    gluPerspective(35.0f, fAspect, 1.0f, 50.0f);

	glMatrixMode(GL_MODELVIEW);
	glLoadIdentity();
}

int main(int argc, char** argv)
{
	inputData(&Bound);
	GM = findGameMatrix(Bound, eps);
	Aq = find_A_EquilibriumWMiss(1, GM);
	Ar = find_A_EquilibriumWMiss(2, GM);
	Bq = find_B_EquilibriumWMiss(1, Aq, GM);
	Br = find_B_EquilibriumWMiss(2, Ar, GM);
	Cq = find_C_EquilibriumWMiss(1, Aq, GM);
	Cr = find_C_EquilibriumWMiss(2, Ar, GM);

	findEquilibriumWMissPTest('A', GM, Aq, ITERATION_NUM_CPU);

/*	printf("\nGame matrix:");
	printGameMatrix(GM);
	printf("\nAq-equilibrium matrix:");
	printBoolMatrixQR(GM.qDim, GM.rDim, Aq);
	printf("\nAr-equilibrium matrix:");
	printBoolMatrixQR(GM.qDim, GM.rDim, Ar);
*/

//	copyGameMatrixGPU(GM, GMDev);

	AqGPU = findEquilibriumGPU('A', 1, GM, 1);
	BqGPU = findEquilibriumGPU('B', 1, GM, 0);
	CqGPU = findEquilibriumGPU('C', 1, GM, 0);
//----------------------------------------OpenGL----------------------------------------
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	glutInitWindowSize(800, 600);
	glutCreateWindow("Equilibrium on Plane");
	glutReshapeFunc(ChangeSize);
    glutSpecialFunc(SpecialKeys);
	glutKeyboardFunc(KeyFunc);
//	glutMouseFunc(MouseFunc);
	glutDisplayFunc(RenderScene);
	SetupRC();

	glutMainLoop();
//------------------------------------------------------------------------------------
	return 1;
}