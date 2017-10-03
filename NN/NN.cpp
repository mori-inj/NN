#include <windows.h>
#include <gdiplus.h>
#include <time.h>
#include "function.h"
#include "gdx.h"
//#include "resource.h"
using namespace Gdiplus;
#pragma comment(lib, "gdiplus")
#pragma warning(disable:4996)

const int WIDTH = 800;
const int HEIGHT = 600;

int OUTPUT_CNT;
//Model model;

#define GDX_MODE
GDX model;
int class0_data_size, class1_data_size;
FILE *fp0, *fp1;
vector<Data> input_data_list, output_data_list;
vector<Data> input_test_data_list, output_test_data_list;

HINSTANCE g_hInst;
HWND hWndMain;
HWND plotWindowHwnd;
LPCTSTR lpszClass = TEXT("GdiPlusStart");

void OnPaint(HDC hdc, int ID, int x, int y);
void OnPaintA(HDC hdc, int ID, int x, int y, double alpha);
LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK WndProcPlot(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

int WINAPI WinMain(HINSTANCE hInstance, HINSTANCE hPrevInstance, LPSTR lpszCmdLine, int nCmdShow)
{
	HWND     hWnd;
	MSG		 msg;
	WNDCLASS WndClass;

	g_hInst = hInstance;

	ULONG_PTR gpToken;
	GdiplusStartupInput gpsi;
	if (GdiplusStartup(&gpToken, &gpsi, NULL) != Ok)
	{
		MessageBox(NULL, TEXT("GDI+ 라이브러리를 초기화할 수 없습니다."), TEXT("알림"), MB_OK);
		return 0;
	}


	WndClass.style = CS_HREDRAW | CS_VREDRAW;
	WndClass.lpfnWndProc = WndProc;
	WndClass.cbClsExtra = 0;
	WndClass.cbWndExtra = 0;
	WndClass.hInstance = hInstance;
	WndClass.hIcon = LoadIcon(NULL, IDI_APPLICATION);
	WndClass.hCursor = LoadCursor(NULL, IDC_ARROW);
	WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	WndClass.lpszMenuName = NULL;
	WndClass.lpszClassName = L"NN";
	RegisterClass(&WndClass);

	WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	WndClass.lpfnWndProc = (WNDPROC)WndProcPlot;
	WndClass.lpszClassName = L"Plot";
	RegisterClass(&WndClass);

	hWnd = CreateWindow(
		L"NN",
		L"NN",
		WS_OVERLAPPEDWINDOW,
		GetSystemMetrics(SM_CXFULLSCREEN) / 2 - WIDTH/2,
		GetSystemMetrics(SM_CYFULLSCREEN) / 2 - HEIGHT/2,
		WIDTH,
		HEIGHT,
		NULL,
		NULL,
		hInstance,
		NULL
		);
	ShowWindow(hWnd, nCmdShow);
	UpdateWindow(hWnd);
	while (GetMessage(&msg, NULL, 0, 0))
	{
		TranslateMessage(&msg);
		DispatchMessage(&msg);
	}
	return (int)msg.wParam;
}

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	HDC hdc, MemDC;
	PAINTSTRUCT ps;

	HBITMAP hBit, OldBit;
	HPEN hPen, oldPen;
	RECT crt;


	static long double precision;


	switch (iMsg)
	{
	case WM_CREATE:
		if(1) {
		//build model
		int l[5] = {5, 4, 3}; //data1
		//int l[5] = {40, 20, 10, 5, 2}; //data4
		srand((unsigned)time(NULL));

		model.add_input_layer(2);
		model.add_output_layer(1);
		for(int i=0; i<3; i++) {
			model.add_layer(l[i], 
				//[](LD x) -> LD{return sigmoid(x);},
				//[](LD x) -> LD{return deriv_sigmoid(x);}
				//[](LD x) -> LD{return ReLU(x);},
				//[](LD x) -> LD{return deriv_ReLU(x);}
				[](LD x) -> LD{return PReLU(x);},
				[](LD x) -> LD{return deriv_PReLU(x);}
				//[](LD x) -> LD{return exponential_converge(x);},
				//[](LD x) -> LD{return deriv_exponential_converge(x);}
			);
		}
#ifndef GDX_MODE
		model.add_all_weights();
#endif
		
#ifdef GDX_MODE
		model.read_bias_and_weights("C:/AI/NN/data1/weight.txt");
#endif

		model.print();

		//read data
		int SIZE;

		fp0 = fopen("C:/AI/NN/data1/temp_class0_5000","r");
		fp1 = fopen("C:/AI/NN/data1/temp_class1_5000","r");
		
		fscanf(fp0, "%d", &class0_data_size);
		fscanf(fp1, "%d", &class1_data_size);
		for(int i=0; i<class0_data_size; i++) {
			long double x0, x1;
			fscanf(fp0, "%Lf %Lf",&x0, &x1);
			vector<long double> input_data, output_data;
			input_data.push_back(x0);
			input_data.push_back(x1);
			output_data.push_back(0);
			input_data_list.push_back(input_data);
			output_data_list.push_back(output_data);
		}
		for(int i=0; i<class1_data_size; i++) {
			long double x0, x1;
			fscanf(fp1, "%Lf %Lf",&x0, &x1);
			vector<long double> input_data, output_data;
			input_data.push_back(x0);
			input_data.push_back(x1);
			output_data.push_back(1);
			input_data_list.push_back(input_data);
			output_data_list.push_back(output_data);
		}
		SIZE = (int)input_data_list.size();
		for(int i=0; i<SIZE*10; i++) {
			int idx0 = rand()%SIZE;
			int idx1 = rand()%SIZE;

			auto temp_input = input_data_list[idx0];
			input_data_list[idx0] = input_data_list[idx1];
			input_data_list[idx1] = temp_input;

			auto temp_output = output_data_list[idx0];
			output_data_list[idx0] = output_data_list[idx1];
			output_data_list[idx1] = temp_output;
		}

		//read test data
		fp0 = fopen("C:/AI/NN/data1/temp_class0_500","r");
		fp1 = fopen("C:/AI/NN/data1/temp_class1_500","r");

		fscanf(fp0, "%d", &class0_data_size);
		fscanf(fp1, "%d", &class1_data_size);
		for(int i=0; i<class0_data_size; i++) {
			long double x0, x1;
			fscanf(fp0, "%Lf %Lf",&x0, &x1);
			vector<long double> input_data, output_data;
			input_data.push_back(x0);
			input_data.push_back(x1);
			output_data.push_back(0);
			input_test_data_list.push_back(input_data);
			output_test_data_list.push_back(output_data);
		}
		for(int i=0; i<class1_data_size; i++) {
			long double x0, x1;
			fscanf(fp1, "%Lf %Lf",&x0, &x1);
			vector<long double> input_data, output_data;
			input_data.push_back(x0);
			input_data.push_back(x1);
			output_data.push_back(1);
			input_test_data_list.push_back(input_data);
			output_test_data_list.push_back(output_data);
		}
		SIZE = (int)input_test_data_list.size();
		for(int i=0; i<SIZE*10; i++) {
			int idx0 = rand()%SIZE;
			int idx1 = rand()%SIZE;

			auto temp_input = input_test_data_list[idx0];
			input_test_data_list[idx0] = input_test_data_list[idx1];
			input_test_data_list[idx1] = temp_input;

			auto temp_output = output_test_data_list[idx0];
			output_test_data_list[idx0] = output_test_data_list[idx1];
			output_test_data_list[idx1] = temp_output;
		}


#ifdef GDX_MODE
		//gene part
		model.init_X(0,1);
		printf("X: (%Lf, %Lf)\n",model.get_X()[0], model.get_X()[1]); fflush(stdout);
#endif



		}
		//SetTimer(hWnd, 1, 10, 0);
		InvalidateRect(hWnd, NULL, FALSE);
		break;

	case WM_TIMER:
	{
#ifndef GDX_MODE
		//for training NN
		printf("train: %Lf (%Lf%%), test: %Lf (%Lf%%)\n",
			model.get_error(input_data_list, output_data_list), 
			precision = model.get_precision(input_data_list, output_data_list)*100, 
			model.get_error(input_test_data_list, output_test_data_list),
			model.get_precision(input_test_data_list, output_test_data_list)*100);
		fflush(stdout);
		if(precision <= 90) {
			model.train(0.01, input_data_list, output_data_list);
		} else if(precision <= 95) {
			model.train(0.001, input_data_list, output_data_list);
		} else {
			model.train(0.0001, input_data_list, output_data_list);
		}

		if(precision >= 99.9) {
			model.print_bias_and_weights();
			exit(0);
		} else {
			InvalidateRect(hWnd, NULL, FALSE);
		}
#endif




#ifdef GDX_MODE
		//GD on X
		model.calc_grad_X([](LD x) -> LD{return deriv_PReLU(x);});
		//model.calc_grad_X([](LD x) -> LD{return deriv_exponential_converge(x);});
		model.update_grad_X(0.001);
		InvalidateRect(hWnd, NULL, FALSE);
#endif

		break;
	}

	case WM_PAINT:
	{
		hdc = BeginPaint(hWnd, &ps);
		GetClientRect(hWnd, &crt);

		MemDC = CreateCompatibleDC(hdc);
		hBit = CreateCompatibleBitmap(hdc, crt.right, crt.bottom);
		OldBit = (HBITMAP)SelectObject(MemDC, hBit);
		//hBrush = CreateSolidBrush(RGB(255, 255, 255));
		//oldBrush = (HBRUSH)SelectObject(MemDC, hBrush);
		hPen = CreatePen(PS_SOLID, 5, RGB(255, 255, 255));
		oldPen = (HPEN)SelectObject(MemDC, hPen);

		//FillRect(MemDC, &crt, hBrush);
		SetBkColor(MemDC, RGB(255, 255, 255));

		/*RECT temp;
		GetWindowRect(hWnd, &temp);
		SendMessage(plotWindowHwnd, WM_CLOSE, NULL, NULL);
				plotWindowHwnd = CreateWindow(
							L"Plot",
							L"Plot",
							WS_BORDER | WS_CHILD | WS_POPUPWINDOW | WS_OVERLAPPEDWINDOW,
							temp.left - 300,
							temp.top,
							300,
							300,
							hWnd,
							(HMENU)0,
							g_hInst,
							NULL
							);
		ShowWindow(plotWindowHwnd, SW_SHOW);*/


		const int TOP = 100, BOTTOM = 500, LEFT = 200, RIGHT = 600;
		//draw axis
		hPen = CreatePen(PS_SOLID, 2, RGB(255,255,255));
		oldPen = (HPEN)SelectObject(MemDC, hPen);
		MoveToEx(MemDC, LEFT, (TOP+BOTTOM)/2, NULL); 
		LineTo(MemDC, RIGHT, (TOP+BOTTOM)/2);
		MoveToEx(MemDC, (LEFT+RIGHT)/2, TOP, NULL); 
		LineTo(MemDC, (LEFT+RIGHT)/2, BOTTOM);
		SelectObject(MemDC, oldPen);
		DeleteObject(hPen);

		//draw data
		int SIZE = (int)input_data_list.size();
		for(int i=0; i<SIZE/10; i++) {
			long double x0 = input_data_list[i][0];
			long double x1 = input_data_list[i][1];
			long double y = output_data_list[i][0];
			long double h = model.get_output(input_data_list[i])[0];

			int red, green, blue;
			if(y >= 0.5) {
				red = 255;
				green = 0;
				blue = 255;
			} else {
				red = 0;
				green = 255;
				blue = 255;
			}

			if(h>= 0.5) {
					
			} else {
				red /= 2;
				green /= 2;
				blue /= 2;
			}

			x0 = x0 * (RIGHT-LEFT) + LEFT;
			x1 = -x1 * (BOTTOM-TOP) + BOTTOM;
			long double r = 2;
			//Ellipse(MemDC, x0-r, x1-r, x0+r, x1+r);
			hPen = CreatePen(PS_SOLID, 1, RGB(red,green,blue));
			oldPen = (HPEN)SelectObject(MemDC, hPen);

			MoveToEx(MemDC, (int)(x0-r), (int)(x1-r), NULL); 
			LineTo(MemDC, (int)(x0+r), (int)(x1-r));
			LineTo(MemDC, (int)(x0+r), (int)(x1+r));
			LineTo(MemDC, (int)(x0-r), (int)(x1+r));
			LineTo(MemDC, (int)(x0-r), (int)(x1-r));

				
			SelectObject(MemDC, oldPen);
			DeleteObject(hPen);
		}

#ifdef GDX_MODE
		//draw X
		hPen = CreatePen(PS_SOLID, 2, RGB(255, 255, 0));
		oldPen = (HPEN)SelectObject(MemDC, hPen);

		vector<LD> X = model.get_X();
		printf("X: (%Lf, %Lf)\n",model.get_X()[0], model.get_X()[1]); fflush(stdout);
		long double x0 = X[0];
		long double x1 = X[1];
		x0 = x0 * (RIGHT-LEFT) + LEFT;
		x1 = -x1 * (BOTTOM-TOP) + BOTTOM;
		long double r = 3;

		MoveToEx(MemDC, (int)(x0-r), (int)(x1-r), NULL); 
		LineTo(MemDC, (int)(x0+r), (int)(x1-r));
		LineTo(MemDC, (int)(x0+r), (int)(x1+r));
		LineTo(MemDC, (int)(x0-r), (int)(x1+r));
		LineTo(MemDC, (int)(x0-r), (int)(x1-r));

		SelectObject(MemDC, oldPen);
		DeleteObject(hPen);
#endif




		BitBlt(hdc, 0, 0, crt.right, crt.bottom, MemDC, 0, 0, SRCCOPY);
		SelectObject(MemDC, OldBit);
		DeleteDC(MemDC);
		//SelectObject(MemDC, oldPen);
		//DeleteObject(hPen);
		//SelectObject(MemDC, oldBrush);
		//DeleteObject(hBrush);
		DeleteObject(hBit);
		EndPaint(hWnd, &ps);

		PostMessage(hWnd, WM_TIMER, NULL, NULL);
		break;
	}

	case WM_DESTROY:
		model.print_bias_and_weights();
		PostQuitMessage(0);
		break;
	/*case WM_QUIT:
	case WM_CLOSE:
		model.print_weights();
		break;*/
	}
	return DefWindowProc(hWnd, iMsg, wParam, lParam);
}

LRESULT CALLBACK WndProcPlot(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	HDC hdc, MemDC;
	PAINTSTRUCT ps;

	HBITMAP hBit, OldBit;
	RECT crt;

	HPEN hPen, oldPen;

	static long double zoom = 10;
	static long double precision;

	switch(iMsg)
	{
		case WM_CREATE:
		{
			//SetTimer(hWnd, 1, 200, 0);
			break;
		}

		case WM_TIMER:
		{
			printf("train: %Lf (%Lf%%), test: %Lf (%Lf%%)\n",
				model.get_error(input_data_list, output_data_list), 
				precision = model.get_precision(input_data_list, output_data_list)*100, 
				model.get_error(input_test_data_list, output_test_data_list),
				model.get_precision(input_test_data_list, output_test_data_list)*100);
			if(precision <= 90) {
				model.train(0.01, input_data_list, output_data_list);
			} else if(precision <= 95) {
				model.train(0.001, input_data_list, output_data_list);
			} else {
				model.train(0.0001, input_data_list, output_data_list);
			}

			InvalidateRect(hWnd, NULL, FALSE);
			break;
		}

		case WM_MOUSEWHEEL:
		{
			break;
		}

		case WM_PAINT:
		{
			hdc = BeginPaint(hWnd, &ps);
			GetClientRect(hWnd, &crt);

			MemDC = CreateCompatibleDC(hdc);
			hBit = CreateCompatibleBitmap(hdc, crt.right, crt.bottom);
			OldBit = (HBITMAP)SelectObject(MemDC, hBit);
			SetBkColor(MemDC, RGB(255, 255, 255));

			//draw axis
			hPen = CreatePen(PS_SOLID, 2, RGB(255,255,255));
			oldPen = (HPEN)SelectObject(MemDC, hPen);
			MoveToEx(MemDC, 20, 130, NULL); 
			LineTo(MemDC, 260, 130);
			MoveToEx(MemDC, 140, 250, NULL); 
			LineTo(MemDC, 140, 10);
			SelectObject(MemDC, oldPen);
			DeleteObject(hPen);

			//draw data
			int SIZE = (int)input_data_list.size();
			for(int i=0; i<SIZE/10; i++) {
				long double x0 = input_data_list[i][0];
				long double x1 = input_data_list[i][1];
				long double y = output_data_list[i][0];
				long double h = model.get_output(input_data_list[i])[0];

				int red, green, blue;
				if(y >= 0.5) {
					red = 255;
					green = 0;
					blue = 255;
				} else {
					red = 0;
					green = 255;
					blue = 255;
				}

				if(h>= 0.5) {
					
				} else {
					red /= 2;
					green /= 2;
					blue /= 2;
				}

				x0 = x0 * 240 + 20;
				x1 = -x1 * 240 + 250;
				long double r = 2;
				//Ellipse(MemDC, x0-r, x1-r, x0+r, x1+r);
				hPen = CreatePen(PS_SOLID, 1, RGB(red,green,blue));
				oldPen = (HPEN)SelectObject(MemDC, hPen);

				MoveToEx(MemDC, (int)(x0-r), (int)(x1-r), NULL); 
				LineTo(MemDC, (int)(x0+r), (int)(x1-r));
				LineTo(MemDC, (int)(x0+r), (int)(x1+r));
				LineTo(MemDC, (int)(x0-r), (int)(x1+r));
				LineTo(MemDC, (int)(x0-r), (int)(x1-r));

				
				SelectObject(MemDC, oldPen);
				DeleteObject(hPen);
			}



			BitBlt(hdc, 0, 0, crt.right, crt.bottom, MemDC, 0, 0, SRCCOPY);
			SelectObject(MemDC, OldBit);
			DeleteDC(MemDC);
			DeleteObject(hBit);
			EndPaint(hWnd, &ps);
			
			PostMessage(plotWindowHwnd, WM_TIMER, NULL, NULL);
			break;
		}
		case WM_DESTROY:
		{
			plotWindowHwnd = NULL;
			break;
		}
	}
	return DefWindowProc(hWnd, iMsg, wParam, lParam);
}