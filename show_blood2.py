import cv2
import numpy as np
import pyautogui
#1280，720
def capture_specific_area(x, y, width, height):
    # 截取屏幕上指定区域的图像
    screenshot = pyautogui.screenshot(region=(x, y, width, height))

    # 将 PIL 图像转换为 NumPy 数组，以便 OpenCV 可以处理
    frame = np.array(screenshot)

    # OpenCV 默认使用 BGR 颜色空间，而 pyautogui 截图为 RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    return frame

def detect_health_bars(image, ally_positions):
    # 定义颜色阈值范围（绿色血条）
    lower_green = np.array([34, 80, 0])
    upper_green = np.array([68, 255, 255])
    # 设置最大宽度和高度阈值
    max_width, max_height = 200, 20
    min_width, min_height = 5,2
    ally_enemy_part=210
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 根据颜色阈值获取血条区域的掩码
    mask = cv2.inRange(hsv, lower_green, upper_green)
    
    # 查找所有符合颜色的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    output_image = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)
    cv2.imshow("Contours", output_image)
    # 用于存放血条区域的坐标
    detected_bars = []

    # 处理上半部分的我方血条
    for pos in ally_positions:
        x, y, w, h = pos
        if y + h <ally_enemy_part:  # 确保位置在上半部分
            roi = image[y:y+h, x:x+w]  # 获取该区域的图像
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_roi = cv2.inRange(roi_hsv, lower_green, upper_green)
            green_length = np.sum(mask_roi > 0, axis=1).max()
            print(green_length)
            health_percentage = int((green_length /w) * 100)
            if health_percentage > 100:
                health_percentage = 100
            # 如果在该区域内检测到绿色区域（即血条存在），则认为该血条未空
            if green_length > 0:
                detected_bars.append(('ally', pos, health_percentage))
            else :
                detected_bars.append(('ally', pos, 0))

    # 处理下半部分的敌方血条（不定位置）
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)

        # 确保位置在下半部分并且大小符合要求
        if y>=ally_enemy_part and w <= max_width and h <= max_height and w >= min_width and h >= min_height:
            detected_bars.append(('enemy', (x, y, w, h),100))
    return detected_bars

# 假设我方血条的固定位置，格式为(x, y, w, h)
ally_positions = [
    (175, 30, 230, 10), # 我方血条1位置
    (460, 30, 55, 10),  # 我方血条2位置
    (570, 30, 55, 10)   # 我方血条3位置
]

x1, y1, w1, h1 = 758,79,335,10
def detect_energy_bars(image):
    lower_energy = np.array([0,0, 50])
    upper_energy = np.array([179,50, 200])
    roi = image[y1:y1+h1, x1:x1+w1]  # 获取该区域的图像
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_roi = cv2.inRange(roi_hsv, lower_energy, upper_energy)
    energy_length = np.sum(mask_roi > 0, axis=1).max()
    print(energy_length)
    energy_percentage = int((energy_length / w1) * 100)
    # if energy_percentage > 100:

    #     energy_percentage = 100

    return energy_percentage

# 读取本地图片
image_path = '4.png'
image = cv2.imread(image_path)

detected_energy=detect_energy_bars(image)
# 调用函数识别血条
#detected_bars = detect_health_bars(image, ally_positions)
cv2.rectangle(image, (x1, y1), (x1+w1, y1+h1), (0, 255, 0), 2)
cv2.putText(image, str(detected_energy), (x1+30, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)   

# 在图像上标记检测到的血条
# for bar_type, (x, y, w, h),health in detected_bars:
#     if bar_type == 'ally':
#         # 我方血条用蓝色矩形框标记
#         cv2.rectangle(image, (x, y), (x+w, y+h), (255, 0, 0), 2)
#         cv2.putText(image, str(health), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
#     elif bar_type == 'enemy':
#         # 敌方血条用红色矩形框标记
#         cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
#         cv2.putText(image, str(health), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

# 显示标记后的图像
cv2.imshow('Health Bars Detection', image)
cv2.waitKey(5000)
cv2.destroyAllWindows()