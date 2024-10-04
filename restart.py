# -*- coding: utf-8 -*-
"""
Created on Sat Jul  4 18:31:36 2020

@author: pang
"""
import directkeys
import time
import pyautogui
def restart():
    print("死,restart")
    directkeys.press_esc()
    time.sleep(5)
    print("quit")
    pyautogui.click(500, 500, button='left',clicks=2,interval=1)
    time.sleep(2)
    pyautogui.click(937, 727, button='left',clicks=1,interval=1)
    time.sleep(3)
    pyautogui.click(720, 458, button='left',clicks=2,interval=1)
    time.sleep(10)
    print("in")
    directkeys.go_forward()
    directkeys.go_forward()
    directkeys.go_forward()
    directkeys.go_forward()
    directkeys.go_forward()
    directkeys.go_forward()
    print("开始新一轮")

if __name__ == "__main__":  
    time.sleep(3)
    restart()

