import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import random
from collections import deque
import numpy as np
import cv2
REPLAY_SIZE = 2000
INITIAL_EPSILON = 0.5
FINAL_EPSILON = 0.01
BATCH_SIZE = 16
GAMMA = 0.9
class NET(nn.Module):
    def __init__(self, observation_height, observation_width, action_space) -> None:
        super(NET, self).__init__()
        self.state_dim = observation_width * observation_height
        self.state_w = observation_width
        self.state_h = observation_height
        self.action_dim = action_space
        self.relu = nn.ReLU()
        self.net = nn.Sequential( 
            nn.Conv2d(1, 32, kernel_size=[5,5],stride=1,padding='same'), 
            nn.ReLU(),  
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(32, 64, kernel_size=[5,5],stride=1,padding='same'), 
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.fc1 = nn.Linear(int((self.state_w/4) * (self.state_h/4) * 64), 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, self.action_dim)

    def forward(self,x):
        x = self.net(x)
        x = x.reshape(-1, int((self.state_w/4) * (self.state_h/4) * 64))
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class DQN(object):
    def __init__(self,observation_height, observation_width, action_space, model_file,log_file):
        self.model_file = model_file
        self.target_net = NET(observation_height,observation_width,action_space)
        self.target_net.to("cpu")
        self.eval_net = NET(observation_height,observation_width,action_space)
        self.eval_net.to("cpu")
        self.replay_buffer = deque()
        self.epsilon = INITIAL_EPSILON
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=0.001)
        self.loss = nn.MSELoss()
        self.action_dim = action_space

    def choose_action(self, state):
        # the output is a tensor, so the [0] is to get the output as a list
        # use epsilon greedy to get the action
        if random.random() <= self.epsilon:
            # if lower than epsilon, give a random value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            return random.randint(0, self.action_dim - 1)
        else:
            # print("choose_action")
            # print(state.shape)
            Q_value = self.eval_net(state)
            # if bigger than epsilon, give the argmax value
            self.epsilon -= (INITIAL_EPSILON - FINAL_EPSILON) / 10000
            # print("Q_value: ",Q_value.shape)
            # print("Q_value: ",Q_value)
            return torch.argmax(Q_value)
    def store_data(self, state, action, reward, next_state, done):
        one_hot_action = np.zeros(self.action_dim)
        one_hot_action[action] = 1
        self.replay_buffer.append((state, one_hot_action, reward, next_state, done))
        if len(self.replay_buffer) > REPLAY_SIZE:
            self.replay_buffer.popleft()
    def train(self):
        self.target_net.to("cpu")
        minibatch = random.sample(self.replay_buffer, BATCH_SIZE)
        state_batch = [data[0] for data in minibatch][0]
        action_batch = [data[1] for data in minibatch]
        reward_batch = [data[2] for data in minibatch]
        next_state_batch = [data[3] for data in minibatch]
        y_batch = []
        Q_value_batch = self.target_net(torch.tensor(next_state_batch).to(dtype=torch.float).to("cpu"))
        for i in range(0, BATCH_SIZE):
            done = minibatch[i][4]
            # see if the station is the final station
            if done:
                y_batch.append(reward_batch[i])
            else:
                # the Q value caculate use the max directly
                y_batch.append(reward_batch[i] + GAMMA * torch.max(Q_value_batch[i]))
        # 假设 self.Q_value 是模型输出的Q值，self.action_input 是动作的one-hot编码表示
        # print(Q_value_batch.shape)
        # print(Q_value_batch.dtype)
        action_batch = torch.tensor(action_batch).to(dtype=torch.float).to("cpu")
        # print(action_batch.shape)
        # print(action_batch.dtype)
        Q_eval = self.eval_net(torch.tensor(state_batch).to(dtype=torch.float).to("cpu"))
        Q_action = torch.sum(Q_eval * action_batch, dim=1)
        y_batch = torch.tensor(y_batch).to(dtype=torch.float).to("cpu")
        print(y_batch.shape)
        loss = self.loss(Q_action, y_batch)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save_model(self):
        torch.save(self.target_net.state_dict(), self.model_file)
    def update_target(self):
        self.target_net.load_state_dict(self.eval_net.state_dict())


import cv2
import time
import directkeys
from restart import restart
import pyautogui
import numpy as np

DQN_model_path = "model_gpu_2"
DQN_log_path = "logs_gpu/"
WIDTH = 96
HEIGHT = 88
window_size = (00, 0, 1280, 720)#384,352  192,176 96,88 48,44 24,22
# station window_size

ally_positions = [
    (173, 68, 228, 10),  # 我方血条1位置175, 30, 230, 10
    (457, 68, 52, 10),   # 我方血条2位置460, 30, 55, 10
    (567, 68, 52, 10)    # 我方血条3位置
]
enemy_position = [
    (760,68,375,10)     # 敌方血条位置
]
# used to get boss and self blood

action_size = 11
# action[n_choose,j,k,m,r]
# j-attack, k-jump, m-defense, r-dodge, n_choose-do nothing

EPISODES = 3000
big_BATCH_SIZE = 16
UPDATE_STEP = 50
# times that evaluate the network
num_step = 0
# used to save log graph
target_step = 0
# used to update target Q network
paused = True
# used to stop training
def detect_health_bars(image, ally_positions):
    # 定义颜色阈值范围（绿色血条）
    lower_green = np.array([34, 80, 0])
    upper_green = np.array([68, 255, 255])
    # 设置最大宽度和高度阈值
    max_width, max_height = 200, 20
    min_width, min_height = 20, 3
    ally_enemy_part = 210
    num_enemy = 1
    pos_num=0
    # 将图像转换为HSV颜色空间
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 根据颜色阈值获取血条区域的掩码
    mask = cv2.inRange(hsv, lower_green, upper_green)
    # 查找所有符合颜色的轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 用于存放血条区域的坐标
    detected_bars = []

    # 处理上半部分的我方血条
    for pos in ally_positions:
        x, y, w, h = pos
        pos_num += 1
        if y + h < ally_enemy_part:  # 确保位置在上半部分
            roi = image[y:y+h, x:x+w]  # 获取该区域的图像
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_roi = cv2.inRange(roi_hsv, lower_green, upper_green)
            green_length = np.sum(mask_roi > 0, axis=1).max()
            health_percentage = int((green_length / w) * 100)
            if health_percentage > 100:
                health_percentage = 100
            # 如果在该区域内检测到绿色区域（即血条存在），则认为该血条未空
            if green_length > 0:
                detected_bars.append(('ally', pos_num,int(health_percentage)))
            else:
                detected_bars.append(('ally', pos_num, 0))

    for pos in enemy_position:
        x, y, w, h = pos
        if y + h < ally_enemy_part:
            roi = image[y:y+h, x:x+w]  # 获取该区域的图像
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            mask_roi = cv2.inRange(roi_hsv, lower_green, upper_green)
            green_length = np.sum(mask_roi > 0, axis=1).max()
            health_percentage = 100-int((green_length/ w) * 100)
            if health_percentage <0:
                health_percentage = 0
            # 如果在该区域内检测到绿色区域（即血条存在），则认为该血条未空
            if green_length > 0:
                detected_bars.append(('enemy', 0, int(health_percentage)))
            else:
                detected_bars.append(('enemy', 0, 0))

    # 处理下半部分的敌方血条（不定位置）
    for contour in contours:
        # 计算轮廓的边界框
        x, y, w, h = cv2.boundingRect(contour)
        # 确保位置在下半部分并且大小符合要求
        if y >= ally_enemy_part and w <= max_width and h <= max_height and w >= min_width and h >= min_height:
            num_enemy+=1
    detected_bars.append(('enemy',num_enemy, 100))
    return detected_bars

def detect_energy_bars(image):
    x1, y1, w1, h1 = 758,80,335,10
    lower_energy = np.array([0,0, 50])
    upper_energy = np.array([179,50, 200])
    roi = image[y1:y1+h1, x1:x1+w1]  # 获取该区域的图像
    roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask_roi = cv2.inRange(roi_hsv, lower_energy, upper_energy)
    energy_length = np.sum(mask_roi > 0, axis=1).max()
    energy_percentage = int((energy_length / w1) * 100)
    if energy_percentage > 100:
        energy_percentage = 100
    return energy_percentage

def take_action(action):
    print("action: ",action)
    if action == 0:     # n_choose
        pass
    elif action == 10:   # j
        directkeys.finishattack()
    elif action == 2:   # k
        directkeys.specialattack()
    elif action == 3:   # m
        directkeys.go_forward()
    elif action == 4:   # r
        directkeys.go_back()
    elif action == 5:   # j+k
        directkeys.go_left()
    elif action == 6:   # j+m
        directkeys.go_right()
    elif action == 7:   # j+r
        directkeys.dodge()
    elif action == 8:   # k+m
        directkeys.turn_up()
    elif action == 9:   # k+r
        directkeys.turn_down()
    elif action == 1:   # k+r
        directkeys.press_mouse_left_button()
        directkeys.release_mouse_left_button()
    
def action_judge(boss_blood, next_boss_blood, self_blood, next_self_blood_all, stop, emergence_break):
    # get action reward
    # emergence_break is used to break down training
    # 用于防止出现意外紧急停止训练防止错误训练数据扰乱神经网络
    if next_self_blood_all < 3:     # self dead
        if emergence_break < 2:
            reward = -10
            done = 1
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break
        else:
            reward = -10
            done = 1
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break
    elif next_self_blood_all - boss_blood > 15:   #boss dead
        if emergence_break < 2:
            reward = 20
            done = 0
            stop = 0
            emergence_break += 1
            return reward, done, stop, emergence_break
        else:
            reward = 20
            done = 0
            stop = 0
            emergence_break = 100
            return reward, done, stop, emergence_break
    else:
        self_blood_reward = 0
        boss_blood_reward = 0
        # print(next_self_blood - self_blood)
        # print(next_boss_blood - boss_blood)
        if next_self_blood_all - self_blood < -7:
            if stop == 0:

                self_blood_reward = -6
                stop = 1
                # 防止连续取帧时一直计算掉血
        else:
            stop = 0
        if next_boss_blood - boss_blood <= -3:
            boss_blood_reward = 4
        # print("self_blood_reward:    ",self_blood_reward)
        # print("boss_blood_reward:    ",boss_blood_reward)
        reward = self_blood_reward + boss_blood_reward
        done = 0
        emergence_break = 0
        return reward, done, stop, emergence_break
    
def capture_specific_area(x, y, width, height):
    # 截取屏幕上指定区域的图像
    screenshot = pyautogui.screenshot(region=(x, y, width, height))
    # 将 PIL 图像转换为 NumPy 数组，以便 OpenCV 可以处理
    frame = np.array(screenshot)
    # OpenCV 默认使用 BGR 颜色空间，而 pyautogui 截图为 RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

if __name__ == "__main__":
    restart()
    agent = DQN(HEIGHT, WIDTH,action_size, DQN_model_path, DQN_log_path)
    print("start")
    self_blood = [100.0, 100.0, 100.0]
    self_blood_all=300

    next_self_blood=[100.0, 100.0, 100.0]
    next_self_blood_all=300

    energy_break = 0
    next_energy_break = 0
    energy_flag=0
    next_energy_flag=0

    next_boss_blood=200
    next_boss_num=1
    boss_blood=200
    boss_num=1

    for episode in range(EPISODES):
        screen = capture_specific_area(0, 0, 1280, 720)
        detected_bars = detect_health_bars(screen, ally_positions)
        detected_energy=detect_energy_bars(screen)

        for i in range(len(detected_bars)):
            if detected_bars[i][0] == 'ally':
                # 我方血条用蓝色矩形框标记
                if(detected_bars[i][1] == 1):
                    self_blood[0] = detected_bars[i][2]
                elif(detected_bars[i][1] == 2):
                    self_blood[1] = detected_bars[i][2]
                elif(detected_bars[i][1] == 3):
                    self_blood[2] = detected_bars[i][2]
            elif detected_bars[i][0] == 'enemy':
                if(detected_bars[i][1] == 0):
                    # 敌方血条用红色矩形框标记
                    boss_blood = boss_blood-detected_bars[i][2]
                    energy_break = 100-detected_energy
                else:
                    boss_num = detected_bars[i][1]

        if energy_break == 100:
            energy_flag = 1
        if(self_blood[0]==0 or self_blood[1]==0 or self_blood[2]==0) and (energy_flag != 1):
            self_blood_all = 0
        elif(energy_flag == 1 and self_blood[0]==0 and self_blood[1]==0 and self_blood[2]==0):
            self_blood_all = self_blood_all
            time.sleep(2)
            energy_flag = 0
        else:
            self_blood_all = min(next_self_blood[0],(self_blood[1]+self_blood[0])/2,(self_blood[2]+self_blood[0])/2)

        target_step = 0
        # used to update target Q network
        done = 0
        total_reward = 0
        stop = 0    
        emergence_break = 0
        # 用于防止连续帧重复计算reward
        window_gray = cv2.cvtColor(capture_specific_area(0, 80, 1280, 640),cv2.COLOR_BGR2GRAY)
        state = cv2.resize(window_gray,(WIDTH,HEIGHT))
        last_time = time.time()
        state = torch.from_numpy(state).to(dtype=torch.float).to("cpu")
        while True:
            state = state.reshape(-1,1,HEIGHT,WIDTH)[0]
            # state = state.unsqueeze(0)
            # print("state_shape",state.shape)
            print('loop took {} seconds'.format(time.time()-last_time))
            target_step += 1
            action = agent.choose_action(state)
            # print("choose_action: ",action)
            take_action(action)

            window_gray = cv2.cvtColor(capture_specific_area(0, 80, 1280, 640),cv2.COLOR_BGR2GRAY)
            screen = capture_specific_area(0, 0, 1280, 720)
            detected_bars = detect_health_bars(screen, ally_positions)
            detected_energy=detect_energy_bars(screen)

            next_state = cv2.resize(window_gray,(WIDTH,HEIGHT))
            next_state = np.array(next_state).reshape(-1,1,HEIGHT,WIDTH)[0]

            for bar_type, num, health in detected_bars:
                if bar_type == 'ally':
                    if(num == 1):
                        next_self_blood[0] = health
                    elif(num == 2):
                        next_self_blood[1] = health
                    elif(num == 3):
                        next_self_blood[2] = health
                elif bar_type == 'enemy':
                    if(num == 0):
                        next_boss_blood = next_boss_blood-health
                        next_energy_break = 100-detected_energy
                    else:
                        next_boss_num = 1

            if next_energy_break == 100:
                next_energy_flag=1
            if(next_self_blood[0]==0 or next_self_blood[1]==0 or next_self_blood[2]==0) and (next_energy_flag != 1):
                next_self_blood_all = 0
            elif(next_energy_flag == 1 or (next_self_blood[0]==0 and next_self_blood[1]==0 and next_self_blood[2]==0)):
                next_self_blood_all = next_self_blood_all
                time.sleep(4)
                next_energy_flag = 0
            else:
                next_self_blood_all = min(next_self_blood[0],(next_self_blood[1]+next_self_blood[0])/2,(next_self_blood[2]+next_self_blood[0])/2)
            print("next_boss_blood: ",next_boss_blood)
            print("next_self_blood: ",next_self_blood_all)
            print("next_energy_break: ",next_energy_break)
            reward, done, stop, emergence_break = action_judge(boss_blood, next_boss_blood,
                                                               self_blood_all, next_self_blood_all,
                                                               stop, emergence_break)
            if emergence_break == 100:
                # emergence break , save model and paused
                # 遇到紧急情况，保存数据，并且暂停
                print("emergence_break")
                agent.save_model()
                paused = True

            agent.store_data(state, action, reward,next_state, done)
            if len(agent.replay_buffer) > big_BATCH_SIZE:
                num_step += 1
                # save loss graph
                print('train'+str(num_step))
                agent.train()
            if target_step % UPDATE_STEP == 0:
                agent.update_target()
            station = next_state
            self_bwlood = next_self_blood
            boss_blood = next_boss_blood
            energy_break = next_energy_break
            energy_flag = next_energy_flag
            total_reward += reward
            if done == 1:
                # print("done")
                break
            if episode % 10 == 0:
                agent.save_model()
            # save model
        print('episode: ', episode, 'Evaluation Average Reward:', total_reward/target_step)
        restart()
