// compare.cpp : This file contains the 'main' function. Program execution begins and ends there.
#include <iostream>
#include <ctime>
#include <Windows.h>
#include <Psapi.h>
#include <condition_variable>
#include <processthreadsapi.h>
#include <WinBase.h>
#include <time.h>
#include <xmmintrin.h>
#include <tchar.h>

using namespace std;

void compare() {
	int i, j, k;

	LARGE_INTEGER start, stop, freq;

	double vRez[4], rezMultVect = 0;
	double v1[4] = { 13.20,2.3,3.4,5.6 };
	double v2[4] = { 14.5,5.8,4.9,6.1 };
	double m1[4][4] = { {2.1,3.2,4.5,1.9}, {2.9,8.2,4.5,3.1}, {1.1,2.2,3.3,4.4}, {5.5,6.6,7.7,8.8} };
	double m2[4][4] = { {6.5,6.4,6.3,6.2}, {2.7,8.4,5.7,5.2}, {1.3,8.3,7.4,3.6}, {8.9,4.6,6.4,7.8} };

	double vRez2[4];

	double mRez[4][4] = { {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0}, {0.0,0.0,0.0,0.0} };

	float vfRez[4], rezfMultVect = 0;
	float vf1[4] = { 2.3,1.3,6.4,9.6 };
	float vf2[4] = { 5.2,7.3,1.4,2.6 };
	float mf1[4][4] = { {1.1,2.3,4.5,5.9}, {6.9,7.2,8.5,9.1}, {10.1,11.2,12.3,13.4}, {14.5,15.6,16.7,17.8} };
	float mf2[4][4] = { {10.1,9.2,8.5,7.9}, {6.9,5.2,4.5,3.1}, {2.1,1.2,8.3,7.4}, {6.5,5.6,4.7,3.8} };


	QueryPerformanceFrequency(&freq);


	//adunare normala vector-vector
	QueryPerformanceCounter(&start);

	for (i = 0; i < 4; i++) {
		vRez[i] = v1[i] + v2[i];
	}

	QueryPerformanceCounter(&stop);
	cout << "Adunarea normala vector-vector: " << (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart << endl;


	//adunare SEE vector-vector
	QueryPerformanceCounter(&start);
	__m128 x, y;
	x = _mm_load_ps(vf1);
	_mm_storeu_ps(vf1, x);
	y = _mm_load_ps(vf2);
	_mm_add_ps(x, y);

	QueryPerformanceCounter(&stop);
	cout << "Adunarea SSE vector-vector: " << (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart << endl;

	//inmultire normala vector-vector
	QueryPerformanceCounter(&start);

	for (i = 0; i < 4; i++) {
		vRez[i] = v1[i] * v2[i];
	}

	QueryPerformanceCounter(&stop);
	cout << "Inmultirea normala vector-vector: " << (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart << endl;


	//inmultire SEE vector-vector
	QueryPerformanceCounter(&start);
	x = _mm_load_ps(vf1);
	_mm_storeu_ps(vf1, x);
	y = _mm_load_ps(vf2);
	_mm_mul_ps(x, y);

	QueryPerformanceCounter(&stop);
	cout << "Inmultirea SSE vector-vector: " << (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart << endl;


	//inmultire normala matrice-vector
	QueryPerformanceCounter(&start);

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			vRez2[i] += m1[i][j] * v2[i];
		}
	}

	QueryPerformanceCounter(&stop);
	cout << "Inmultirea normala matrice-vector: " << (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart << endl;

	//inmultire SEE matrice-vector
	QueryPerformanceCounter(&start);
	__m128 a[4];
	for (j = 0; j < 4; j++) {
		k = 0;
		for (i = 0; i < 4; i++) {
			vfRez[i] = mf1[i][j];
		}
		a[k] = _mm_load_ps(vfRez);
		k++;
	}
	y = _mm_load_ps(vf1);

	for (i = 0; i < 4; i++) {
		_mm_mul_ps(a[k], y);
	}

	QueryPerformanceCounter(&stop);
	cout << "Inmultirea SSE matrice-vector: " << (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart << endl;


	//inmultire normala matrice-matrice
	QueryPerformanceCounter(&start);

	for (i = 0; i < 4; i++) {
		for (j = 0; j < 4; j++) {
			mRez[i][j] = m1[i][j] * m2[i][j];
		}
	}

	QueryPerformanceCounter(&stop);
	cout << "Inmultirea normala matrice-matrice: " << (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart << endl;


	//inmultire SEE matrice-matrice
	__m128 b[4];

	QueryPerformanceCounter(&start);
	for (j = 0; j < 4; j++) {
		k = 0;
		for (i = 0; i < 4; i++) {
			vfRez[i] = mf1[i][j];
		}
		a[k] = _mm_load_ps(vfRez);
		k++;
	}

	for (j = 0; j < 4; j++) {
		k = 0;
		for (i = 0; i < 4; i++) {
			vfRez[i] = mf2[i][j];
		}
		b[k] = _mm_load_ps(vfRez);
		k++;
	}

	for (i = 0; i < 4; i++) {
		_mm_mul_ps(a[k], b[k]);
	}


	QueryPerformanceCounter(&stop);
	cout << "Inmultirea SSE matrice-matrice: " << (double)(stop.QuadPart - start.QuadPart) / (double)freq.QuadPart << endl;


}

int main()
{
   
	compare();
	return 0;

}
