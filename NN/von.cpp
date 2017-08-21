#include <Windows.h>
#include <gdiplus.h>
#include <Windowsx.h>
#include <time.h>
#include <iostream>
//#include <vld.h>
#include "gui.h"

using namespace Gdiplus;
using namespace std;
#pragma comment(lib, "gdiplus")
#pragma warning(disable:4018)

HINSTANCE g_hInst;
HWND hWndMain;
LPCTSTR lpszClass = TEXT("GdiPlusStart");

static GUI gui(&g_hInst);

LRESULT CALLBACK WndProc(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK WndProcPlot(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK WndProcInput(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK WndProcInputGUI(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);
LRESULT CALLBACK WndProcOutput(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam);

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
	WndClass.lpszClassName = L"VoN";
	RegisterClass(&WndClass);

	//
	WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	WndClass.lpfnWndProc = (WNDPROC)WndProcPlot;
	WndClass.lpszClassName = L"Plot";
	RegisterClass(&WndClass);
	//

	//
	WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	WndClass.lpfnWndProc = (WNDPROC)WndProcInput;
	WndClass.lpszClassName = L"Input";
	RegisterClass(&WndClass);
	//

	//
	WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	WndClass.lpfnWndProc = (WNDPROC)WndProcInputGUI;
	WndClass.lpszClassName = L"InputGUI";
	RegisterClass(&WndClass);
	//

	//
	WndClass.hbrBackground = (HBRUSH)GetStockObject(WHITE_BRUSH);
	WndClass.lpfnWndProc = (WNDPROC)WndProcOutput;
	WndClass.lpszClassName = L"Output";
	RegisterClass(&WndClass);
	//

	hWnd = CreateWindow(
		L"VoN",
		L"VoN",
		WS_OVERLAPPEDWINDOW,
		GetSystemMetrics(SM_CXFULLSCREEN) / 2 - WIDTH/2 + 100,
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

	int xpos,ypos;

	switch (iMsg)
	{
		case WM_CREATE:
		{
			gui = GUI(&g_hInst);
			SetTimer(hWnd, 1, 10, 0);
			srand((unsigned)time(NULL));
			break;
		}

		case WM_TIMER:
		{
			InvalidateRect(hWnd, NULL, FALSE);
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

			gui.print(MemDC, hWnd);


			BitBlt(hdc, 0, 0, crt.right, crt.bottom, MemDC, 0, 0, SRCCOPY);
			SelectObject(MemDC, OldBit);
			DeleteDC(MemDC);
			
			DeleteObject(hBit);
			EndPaint(hWnd, &ps);
			break;
		}


		case WM_LBUTTONDOWN:
		{
			xpos = GET_X_LPARAM(lParam);
			ypos = GET_Y_LPARAM(lParam);
			gui.LDown(xpos, ypos);
			break;
		}


		case WM_LBUTTONUP:
		{	
			xpos = GET_X_LPARAM(lParam);
			ypos = GET_Y_LPARAM(lParam);
			gui.LUp(xpos, ypos, hWnd);
			break;
		}

		case WM_RBUTTONDOWN:
		{
			xpos = GET_X_LPARAM(lParam);
			ypos = GET_Y_LPARAM(lParam);
			gui.RDown(xpos, ypos, &hWnd);
			break;
		}


		case WM_RBUTTONUP:
		{
			xpos = GET_X_LPARAM(lParam);
			ypos = GET_Y_LPARAM(lParam);
			gui.RUp(xpos, ypos);
			break;
		}


		case WM_MOUSEMOVE:
		{
			xpos = GET_X_LPARAM(lParam);
			ypos = GET_Y_LPARAM(lParam);
			gui.mouse_move(xpos, ypos);
			break;
		}

		case WM_COMMAND:
		{
			gui.command(LOWORD(wParam));
			break;
		}

		case WM_CHAR:
		{
			switch(wParam)
			{
				case 1:
				{
					gui.keyboard_down("all");
					break;
				}
				case 3:
				{
					gui.keyboard_down("copy");
					break;
				}
				case 22:
				{
					gui.keyboard_down("paste");
					break;
				}
				case 24:
				{
					gui.keyboard_down("cut");
					break;
				}
			}
			break;
		}

		case WM_KEYUP:
		{
			switch(wParam)
			{
				case VK_DELETE:
				case VK_BACK:
				{
					gui.keyboard_up("delete");
					break;
				}
				case VK_SHIFT:
				{
					gui.keyboard_up("shift");
					break;
				}
			}
			break;
		}

		case WM_KEYDOWN:
		{
			switch(wParam)
			{
				case VK_SHIFT:
				{
					gui.keyboard_down("shift");
					break;
				}
			}
			break;
		}


		case WM_DESTROY:
		{
			PostQuitMessage(0);
			break;
		}
	}
	return DefWindowProc(hWnd, iMsg, wParam, lParam);
}

LRESULT CALLBACK WndProcPlot(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	return gui.WndProcPlot(hWnd, iMsg, wParam, lParam);
}

LRESULT CALLBACK WndProcInput(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	return gui.WndProcInput(hWnd, iMsg, wParam, lParam);
}

LRESULT CALLBACK WndProcInputGUI(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	return gui.WndProcInputGUI(hWnd, iMsg, wParam, lParam);
}

LRESULT CALLBACK WndProcOutput(HWND hWnd, UINT iMsg, WPARAM wParam, LPARAM lParam)
{
	return gui.WndProcOutput(hWnd, iMsg, wParam, lParam);
}