#include <windows.h>
#include <gdiplus.h>
#include <time.h>
#include "fnn.h"
//#include "resource.h"
using namespace Gdiplus;
#pragma comment(lib, "gdiplus")
#pragma warning(disable:4996)

const int WIDTH = 800;
const int HEIGHT = 600;

int OUTPUT_CNT;
Model model;
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
	RECT crt;


	switch (iMsg)
	{
	case WM_CREATE:
		if(1) {
		//build model
		srand((unsigned)time(NULL));
		model.add_new_input_node();
		model.add_new_input_node();
		model.add_new_output_node();

		for(int i=0; i<4; i++) {
			model.add_new_node();
			model.add_weight(0, i+3);
			model.add_weight(1, i+3);
			model.add_weight(i+3, 2);
		}
		model.print();

		//read data
		int SIZE;

		fp0 = fopen("../data/temp_class0_5000","r");
		fp1 = fopen("../data/temp_class1_5000","r");
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
		fp0 = fopen("../data/temp_class0_500","r");
		fp1 = fopen("../data/temp_class1_500","r");

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

		//train
		RECT temp;
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
		ShowWindow(plotWindowHwnd, SW_SHOW);


		printf("before train: %Lf (%Lf%%), test: %Lf (%Lf%%)\n",
			model.get_error(input_data_list, output_data_list), 
			model.get_precision(input_data_list, output_data_list)*100, 
			model.get_error(input_test_data_list, output_test_data_list),
			model.get_precision(input_test_data_list, output_test_data_list)*100);
		model.train(0.1, 30, input_data_list, output_data_list);
		
		printf("after 1 train: %Lf (%Lf%%), test: %Lf (%Lf%%)\n",
			model.get_error(input_data_list, output_data_list), 
			model.get_precision(input_data_list, output_data_list)*100, 
			model.get_error(input_test_data_list, output_test_data_list),
			model.get_precision(input_test_data_list, output_test_data_list)*100);
		model.train(0.01, 40, input_data_list, output_data_list);
		
		printf("after 2 train: %Lf (%Lf%%), test: %Lf (%Lf%%)\n",
			model.get_error(input_data_list, output_data_list), 
			model.get_precision(input_data_list, output_data_list)*100, 
			model.get_error(input_test_data_list, output_test_data_list),
			model.get_precision(input_test_data_list, output_test_data_list)*100);
		model.train(0.001, 60, input_data_list, output_data_list);
		
		printf("after 3 train: %Lf (%Lf%%), test: %Lf (%Lf%%)\n",
			model.get_error(input_data_list, output_data_list), 
			model.get_precision(input_data_list, output_data_list)*100, 
			model.get_error(input_test_data_list, output_test_data_list),
			model.get_precision(input_test_data_list, output_test_data_list)*100);

		//result
		printf("train error: %Lf (%Lf%%)\ntest error: %Lf (%Lf%%)\n",
			model.get_error(input_data_list, output_data_list), 
			model.get_precision(input_data_list, output_data_list)*100, 
			model.get_error(input_test_data_list, output_test_data_list),
			model.get_precision(input_test_data_list, output_test_data_list)*100);
		}

		SetTimer(hWnd, 1, 10, 0);
		break;

	case WM_TIMER:
		InvalidateRect(hWnd, NULL, FALSE);
		break;

	case WM_PAINT:
		hdc = BeginPaint(hWnd, &ps);
		GetClientRect(hWnd, &crt);

		MemDC = CreateCompatibleDC(hdc);
		hBit = CreateCompatibleBitmap(hdc, crt.right, crt.bottom);
		OldBit = (HBITMAP)SelectObject(MemDC, hBit);
		//hBrush = CreateSolidBrush(RGB(255, 255, 255));
		//oldBrush = (HBRUSH)SelectObject(MemDC, hBrush);
		//hPen = CreatePen(PS_SOLID, 5, RGB(255, 255, 255));
		//oldPen = (HPEN)SelectObject(MemDC, hPen);

		//FillRect(MemDC, &crt, hBrush);
		SetBkColor(MemDC, RGB(255, 255, 255));



		BitBlt(hdc, 0, 0, crt.right, crt.bottom, MemDC, 0, 0, SRCCOPY);
		SelectObject(MemDC, OldBit);
		DeleteDC(MemDC);
		//SelectObject(MemDC, oldPen);
		//DeleteObject(hPen);
		//SelectObject(MemDC, oldBrush);
		//DeleteObject(hBrush);
		DeleteObject(hBit);
		EndPaint(hWnd, &ps);
		break;

	case WM_DESTROY:
		PostQuitMessage(0);
		break;
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

	switch(iMsg)
	{
		case WM_CREATE:
		{
			//SetTimer(hWnd, 1, 200, 0);
			break;
		}

		case WM_TIMER:
		{
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


			BitBlt(hdc, 0, 0, crt.right, crt.bottom, MemDC, 0, 0, SRCCOPY);
			SelectObject(MemDC, OldBit);
			DeleteDC(MemDC);
			DeleteObject(hBit);
			EndPaint(hWnd, &ps);
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