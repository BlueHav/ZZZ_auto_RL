# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:50 2020

@author: pang
"""
#模拟键盘输出
import ctypes
import time

SendInput = ctypes.windll.user32.SendInput


W = 0x11
A = 0x1E
S = 0x1F
D = 0x20

Q = 0x10#终结技
E = 0x12
LSHIFT = 0x2A
C = 0x2E#切换上一位
space = 0x39#切换下一位

esc = 0x01

# C struct redefinitions 
PUL = ctypes.POINTER(ctypes.c_ulong)
class KeyBdInput(ctypes.Structure):
    _fields_ = [("wVk", ctypes.c_ushort),
                ("wScan", ctypes.c_ushort),
                ("dwFlags", ctypes.c_ulong),
                ("time", ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class HardwareInput(ctypes.Structure):
    _fields_ = [("uMsg", ctypes.c_ulong),
                ("wParamL", ctypes.c_short),
                ("wParamH", ctypes.c_ushort)]

class MouseInput(ctypes.Structure):
    _fields_ = [("dx", ctypes.c_long),
                ("dy", ctypes.c_long),
                ("mouseData", ctypes.c_ulong),
                ("dwFlags", ctypes.c_ulong),
                ("time",ctypes.c_ulong),
                ("dwExtraInfo", PUL)]

class Input_I(ctypes.Union):
    _fields_ = [("ki", KeyBdInput),
                 ("mi", MouseInput),
                 ("hi", HardwareInput)]

class Input(ctypes.Structure):
    _fields_ = [("type", ctypes.c_ulong),
                ("ii", Input_I)]

# Actuals Functions

def PressKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))

def ReleaseKey(hexKeyCode):
    extra = ctypes.c_ulong(0)
    ii_ = Input_I()
    ii_.ki = KeyBdInput( 0, hexKeyCode, 0x0008 | 0x0002, 0, ctypes.pointer(extra) )
    x = Input( ctypes.c_ulong(1), ii_ )
    ctypes.windll.user32.SendInput(1, ctypes.pointer(x), ctypes.sizeof(x))
    
    
def finishattack():
    PressKey(Q)
    time.sleep(0.05)
    ReleaseKey(Q)
    #time.sleep(0.1)
    
def specialattack():
    PressKey(E)
    time.sleep(0.05)
    ReleaseKey(E)
    #time.sleep(0.1)
    
def go_forward():
    PressKey(W)
    time.sleep(0.4)
    ReleaseKey(W)
    
def go_back():
    PressKey(S)
    time.sleep(0.4)
    ReleaseKey(S)
    
def go_left():
    PressKey(A)
    time.sleep(0.4)
    ReleaseKey(A)
    
def go_right():
    PressKey(D)
    time.sleep(0.4)
    ReleaseKey(D)
    
def dodge():#闪避
    PressKey(LSHIFT)
    time.sleep(0.1)
    ReleaseKey(LSHIFT)
    #time.sleep(0.1)
def turn_up():
    PressKey(C)
    time.sleep(0.3)
    ReleaseKey(C)
    
def turn_down():
    PressKey(space)
    time.sleep(0.3)
    ReleaseKey(space)
    
def press_esc():
    PressKey(esc)
    time.sleep(0.3)
    ReleaseKey(esc)
    
# 定义鼠标事件标志
MOUSEEVENTF_LEFTDOWN = 0x0002
MOUSEEVENTF_LEFTUP = 0x0004

# 模拟鼠标左键按下
def press_mouse_left_button():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTDOWN, 0, 0, 0, 0)

# 模拟鼠标左键释放
def release_mouse_left_button():
    ctypes.windll.user32.mouse_event(MOUSEEVENTF_LEFTUP, 0, 0, 0, 0)

# 模拟长按鼠标左键
def hold_mouse_left_button():
    press_mouse_left_button()
    time.sleep(3)
    release_mouse_left_button()

def change_break():
    press_mouse_left_button()
    release_mouse_left_button()
