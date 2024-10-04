# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 10:37:50 2020

@author: pang
"""

# import Quartz
import time
# from Quartz.CoreGraphics import CGEventCreateKeyboardEvent, CGEventPost, kCGHIDEventTap

import pydirectinput
import pyautogui

def right():
    pyautogui.press('D')
def left():
    pyautogui.press('A')
def up():
    pyautogui.press('W')
def down():
    pyautogui.press('S')
def attack():
    pyautogui.click(x=500, y=500)
def longattack():
    pyautogui.mouseDown(button='left')
    time.sleep(3)
    pyautogui.mouseUp(button='left',x=500, y=500)
def jumppersonright():
    pyautogui.press('space')
def jumppersonleft():
    pyautogui.press('C')
def dodge():#闪避
    pyautogui.press('Lshift')
def specialattack():
    pyautogui.press('E')
def finishattack():
    pyautogui.press('Q')

def restart_click():
    pyautogui.click(x=936, y=723)

