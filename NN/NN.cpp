#include <Windowsx.h>
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
#ifdef GDX_MODE
vector<Data> X_history;
#endif
GDX model;
FILE *fp0, *fp1;
vector<Data> input_data_list, output_data_list;
vector<Data> input_test_data_list, output_test_data_list;

HINSTANCE g_hInst;
HWND hWndMain;
LPCTSTR lpszClass = TEXT("GdiPlusStart");

void OnPaint(HDC hdc, int ID, int x, int y);
void OnPaintA(HDC hdc, int ID, int x, int y, double alpha);
LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

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

	static bool is_clicked = false;
	static int xpos, ypos;
	static vector<pair<pair<int, int>,bool> > dots;
	static long double precision;


	switch (iMsg)
	{
	case WM_CREATE:
		if(1) {
		//build model
		int l[5] = {256, 256};//{600, 300, 150, 75, 37};
		srand((unsigned)time(NULL));

		model.add_input_layer(28*28);
		model.add_output_layer(10);
		for(int i=0; i<2; i++) {
			model.add_layer(l[i], 
				[](LD x) -> LD{return PReLU(x);},
				[](LD x) -> LD{return deriv_PReLU(x);}
			);
		}
#ifndef GDX_MODE
		model.add_all_weights();
#endif
		
#ifdef GDX_MODE
		model.read_bias_and_weights("C:/AI/NN/mnist_weight_99.txt");
		//model.read_bias_and_weights("C:/AI/NN/mnist_weight_gdx.txt");
#endif

		//model.print();
		printf("model initialize done\n"); fflush(stdout);

		/*
		//read data
		int SIZE;
		fp0 = fopen("C:/AI/NN/mnist/train-images-idx3-ubyte","r");
		fp1 = fopen("C:/AI/NN/mnist/train-labels-idx1-ubyte","r");

		for(int i=0; i<16; i++) {
			unsigned char temp;
			fseek(fp0, i, SEEK_SET);
			fread(&temp, 1, 1, fp0);
		}
		for(int i=0; i<8; i++) {
			unsigned char temp;
			fseek(fp1, i, SEEK_SET);
			fread(&temp, 1, 1, fp1);
		}
		
		for(int i=0; i<60000; i++) {
			vector<LD> input_data;
			vector<LD> output_data;
			for(int j=0; j<28*28; j++) {
				unsigned char data;
				fseek(fp0, 16+i*28*28+j, SEEK_SET);
				fread(&data, 1, 1, fp0);
				input_data.push_back((long double)data);
			}
			
			output_data.push_back(1);

			input_data_list.push_back(input_data);
			output_data_list.push_back(output_data);
		}

		for(int i=0; i<60000; i++) {
			vector<LD> input_data;
			vector<LD> output_data;
			for(int j=0; j<28*28; j++) {
				input_data.push_back((long double)(rand()%256));
			}
			
			output_data.push_back(0);

			input_data_list.push_back(input_data);
			output_data_list.push_back(output_data);
		}
		
		//shuffle
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

		fclose(fp0);
		fclose(fp1);
		printf("read train done\n"); fflush(stdout);


		//read test data
		fp0 = fopen("C:/AI/NN/mnist/t10k-images-idx3-ubyte","r");
		fp1 = fopen("C:/AI/NN/mnist/t10k-labels-idx1-ubyte","r");

		for(int i=0; i<16; i++) {
			unsigned char temp;
			fseek(fp0, i, SEEK_SET);
			fread(&temp, 1, 1, fp0);
		}
		for(int i=0; i<8; i++) {
			unsigned char temp;
			fseek(fp1, i, SEEK_SET);
			fread(&temp, 1, 1, fp1);
		}
		
		for(int i=0; i<10000; i++) {
			vector<LD> input_data;
			vector<LD> output_data;
			for(int j=0; j<28*28; j++) {
				unsigned char data;
				fseek(fp0, 16 + i*28*28 + j, SEEK_SET);
				fread(&data, 1, 1, fp0);
				input_data.push_back((long double)data);
			}
			
			output_data.push_back(1);
			
			input_test_data_list.push_back(input_data);
			output_test_data_list.push_back(output_data);
		}

		for(int i=0; i<10000; i++) {
			vector<LD> input_data;
			vector<LD> output_data;
			for(int j=0; j<28*28; j++) {
				
				input_data.push_back((long double)(rand()%256));
			}
			
			output_data.push_back(0);
			
			input_test_data_list.push_back(input_data);
			output_test_data_list.push_back(output_data);
		}
		
		//shuffle
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

		fclose(fp0);
		fclose(fp1);
		printf("read test done\n"); fflush(stdout);
		*/

		//read data
		int SIZE;
		fp0 = fopen("C:/AI/NN/mnist/train-images-idx3-ubyte","r");
		fp1 = fopen("C:/AI/NN/mnist/train-labels-idx1-ubyte","r");

		for(int i=0; i<16; i++) {
			unsigned char temp;
			fseek(fp0, i, SEEK_SET);
			fread(&temp, 1, 1, fp0);
		}
		for(int i=0; i<8; i++) {
			unsigned char temp;
			fseek(fp1, i, SEEK_SET);
			fread(&temp, 1, 1, fp1);
		}
		
		for(int i=0; i<600; i++) {
			vector<LD> input_data;
			vector<LD> output_data;
			for(int j=0; j<28*28; j++) {
				unsigned char data;
				fseek(fp0, 16+i*28*28+j, SEEK_SET);
				fread(&data, 1, 1, fp0);
				input_data.push_back((long double)data);
			}
			unsigned char label;
			fseek(fp1, 8+i, SEEK_SET);
			fread(&label, 1, 1, fp1);
			for(int j=0; j<10; j++) {
				output_data.push_back(label==j);
			}
			input_data_list.push_back(input_data);
			output_data_list.push_back(output_data);
		}
		
		//shuffle
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

		fclose(fp0);
		fclose(fp1);
		printf("read train done\n"); fflush(stdout);


		//read test data
		fp0 = fopen("C:/AI/NN/mnist/t10k-images-idx3-ubyte","r");
		fp1 = fopen("C:/AI/NN/mnist/t10k-labels-idx1-ubyte","r");

		for(int i=0; i<16; i++) {
			unsigned char temp;
			fseek(fp0, i, SEEK_SET);
			fread(&temp, 1, 1, fp0);
		}
		for(int i=0; i<8; i++) {
			unsigned char temp;
			fseek(fp1, i, SEEK_SET);
			fread(&temp, 1, 1, fp1);
		}
		
		for(int i=0; i<100; i++) {
			vector<LD> input_data;
			vector<LD> output_data;
			for(int j=0; j<28*28; j++) {
				unsigned char data;
				fseek(fp0, 16 + i*28*28 + j, SEEK_SET);
				fread(&data, 1, 1, fp0);
				input_data.push_back((long double)data);
			}
			unsigned char label;
			fseek(fp1, 8+i, SEEK_SET);
			fread(&label, 1, 1, fp1);
			for(int j=0; j<10; j++) {
				output_data.push_back(label==j);
			}
			input_test_data_list.push_back(input_data);
			output_test_data_list.push_back(output_data);
		}
		
		//shuffle
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

		fclose(fp0);
		fclose(fp1);
		printf("read test done\n"); fflush(stdout);


#ifdef GDX_MODE
		//gene part
		//model.init_X(0,255, 1);
		
		/*for(int i=0; (int)output_data_list.size(); i++) {
			if(output_data_list[i][0]) {
				model.set_X(0,255,input_data_list[i], 1);
				X_history.push_back(model.get_X());
				break;
			}
		}*/


		model.set_X(0,255,input_data_list[i], 1);
		X_history.push_back(model.get_X());
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
			model.get_error(input_data_list, output_data_list), //model.get_error_prll(input_data_list, output_data_list, 100), 
			precision = model.get_precision(input_data_list, output_data_list)*100, //model.get_precision_prll(input_data_list, output_data_list,100)*100, 
			model.get_error(input_test_data_list, output_test_data_list), //model.get_error_prll(input_test_data_list, output_test_data_list, 100),
			model.get_precision(input_test_data_list, output_test_data_list)*100); //model.get_precision_prll(input_test_data_list, output_test_data_list,100)*100);
		fflush(stdout);
		if(precision <= 80) {
			for(int i=0; i<10; i++) model.train(0.001, input_data_list, output_data_list);
		} else if(precision <= 85) {
			for(int i=0; i<20; i++) model.train(0.0001, input_data_list, output_data_list);
		} else {
			for(int i=0; i<40; i++) model.train(0.00001, input_data_list, output_data_list);
		}

		if(precision >= 99.2) {
			if(model.get_precision_all(input_data_list, output_data_list)*100 >= 99) {
				model.print_bias_and_weights();
				exit(0);
			} else {
				InvalidateRect(hWnd, NULL, FALSE);
			}
		} else {
			InvalidateRect(hWnd, NULL, FALSE);
		}
#endif




#ifdef GDX_MODE
		//GD on X
		model.calc_grad_X([](LD x) -> LD{return deriv_PReLU(x);});
		//model.calc_grad_X([](LD x) -> LD{return deriv_exponential_converge(x);});
		Data output = model.get_output(model.get_X());
		Data linear_output = model.get_linear_output(model.get_X());
		printf("g(%Lf) = %Lf\n",linear_output[model.TARGET_CLASS], output[model.TARGET_CLASS]); fflush(stdout);

		model.update_grad_X(10000);
		X_history.push_back(model.get_X());
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

#ifdef GDX_MODE
		//draw X
		int x_coord = 50, y_coord = 50;
		for(int idx = 0; idx < (int)X_history.size(); idx+=1) {
			for(int i=0; i<28; i++) {
				for(int j=0; j<28; j++) {
					LD clr = X_history[idx][i*28+j];
					hPen = CreatePen(PS_SOLID, 1, RGB(clr,clr,clr));
					oldPen = (HPEN)SelectObject(MemDC, hPen);
					MoveToEx(MemDC, x_coord+j, y_coord+i, NULL); 
					LineTo(MemDC, x_coord+j+1, y_coord+i);
					SelectObject(MemDC, oldPen);
					DeleteObject(hPen);
				}
			}
			x_coord += 28;
			if(x_coord > 750) {
				x_coord = 50;
				y_coord += 28;
			}
		}
#endif

		BitBlt(hdc, 0, 0, crt.right, crt.bottom, MemDC, 0, 0, SRCCOPY);
		SelectObject(MemDC, OldBit);
		DeleteDC(MemDC);
		DeleteObject(hBit);
		EndPaint(hWnd, &ps);

		PostMessage(hWnd, WM_TIMER, NULL, NULL);
		break;
	}

	
	case WM_DESTROY:
		model.print_bias_and_weights();
		PostQuitMessage(0);
		break;
	}
	return DefWindowProc(hWnd, iMsg, wParam, lParam);
}
