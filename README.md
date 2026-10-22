智慧巡检-基于YOLOV5-V11的安全帽检测系统
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/d6eff6d265f64d55812c98c8e7bed8d0.png)

YOLOV5-V11目标检测通用系统，以安全帽检测为例，亦可改成通用的目标检测系统。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/71af1dc939e74d2f92bf3fe19635ae9d.png)
本项目GUI部分使用pyqt5制作，包括数据库、多线程、自定义组件等知识，亦可作为学习深度学习和pyqt5时的练手项目。功能以及特色如下：![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a5fd93ab394b465badd974d891eeec52.png)

1.带有登陆注册界面
2.输入数据格式：图片、文件夹、视频以及支持摄像头的实时检测，并自动保存最终检测结果。
3.特色功能：支持选区检测功能，通过鼠标左键绘制多边形来确定检测区域，并且最终生成的结果视频中会保存绘制的多边形。
4.支持后处理过程中交并比（iou）以及置信度（conf）的动态调节。
5.检测视频和摄像头时，支持暂停、恢复、停止等功能，且停止后自动保存检测结果。
6.支持检测结果实时动态显示。
7. 提供详细的项目说明以及环境配置教程，即使是深度学习小白也能轻松跑通项目。
8.项目提供项目系统代码，模型训练代码，模型训练结果以及数据集。
9.包含模型训练和界面元素修改教程。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/49461acacc744f71beabc17e83d19bca.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/a4a5a7cf33e6408c945ea06cb37b318c.png)
这是一个非常完整且功能丰富的计算机视觉项目，非常适合大学生毕业设计或研究生课题。它结合了深度学习（YOLOv5/v11）与桌面应用开发（PyQt5）。

由于整个项目包含登录、数据库、多线程、界面绘制等数千行代码，：**YOLOv5 检测线程**、**多边形区域检测逻辑**以及**主界面交互**。

你可以基于这些核心代码构建完整的项目。

---

### 1. 项目核心架构设计

项目主要依赖以下库：
```bash
pip install PyQt5 opencv-python torch torchvision matplotlib numpy
```

### 2. 核心代码：YOLOv5 检测线程 (`detect_thread.py`)

为了不卡死界面，检测过程必须放在子线程中。这个类负责读取视频流、进行推理并发射信号更新界面。

```python
import sys
import torch
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, Qt
from models.common import DetectMultiBackend
from utils.general import non_max_suppression, scale_coords
from utils.plots import Annotator, colors

class DetectThread(QThread):
    # 定义信号：发送处理后的图像、日志信息、统计结果
    send_img = pyqtSignal(np.ndarray)
    send_log = pyqtSignal(str)
    send_status = pyqtSignal(str)

    def __init__(self):
        super(DetectThread, self).__init__()
        self.weights = 'hatV11_best.pt'  # 默认模型路径
        self.source = '0'  # 默认摄像头
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.is_running = False
        self.is_paused = False
        self.polygon_points = []  # 存储选区多边形坐标
        self.model = None

    def load_model(self):
        # 加载 YOLOv5/v11 模型
        self.model = DetectMultiBackend(self.weights)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt

    def run(self):
        self.is_running = True
        self.load_model()
        
        # 打开视频源 (摄像头或视频文件)
        cap = cv2.VideoCapture(int(self.source) if self.source.isdigit() else self.source)
        
        while self.is_running:
            if self.is_paused:
                self.msleep(100)
                continue

            ret, frame = cap.read()
            if not ret:
                break

            # 推理逻辑
            img = self.preprocess(frame)
            pred = self.model(img)
            pred = non_max_suppression(pred, self.conf_thres, self.iou_thres)
            
            # 后处理与绘图
            result_frame = self.postprocess(frame, pred)
            
            # 发送图像到主界面
            self.send_img.emit(result_frame)

        cap.release()

    def preprocess(self, frame):
        # 图像预处理
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float() / 255.0
        img = img.unsqueeze(0)  # 添加 batch 维度
        return img

    def postprocess(self, frame, pred):
        det = pred[0]
        im0 = frame.copy()
        annotator = Annotator(im0, line_width=2, example=str(self.names))
        
        helmet_count = 0
        no_helmet_count = 0
        
        if len(det):
            # 将检测框映射回原图尺寸
            det[:, :4] = scale_coords(self.img.shape[2:], det[:, :4], im0.shape).round()

            for *xyxy, conf, cls in reversed(det):
                c = int(cls)
                label = f'{self.names[c]} {conf:.2f}'
                
                # 判断点是否在多边形内 (选区检测)
                center_point = (int((xyxy[0]+xyxy[2])/2), int((xyxy[1]+xyxy[3])/2))
                if len(self.polygon_points) > 0:
                    # cv2.pointPolygonTest 返回值 > 0 表示在内部
                    res = cv2.pointPolygonTest(np.array(self.polygon_points, np.int32), center_point, False)
                    if res < 0: continue # 如果在多边形外，则跳过不画

                # 统计逻辑
                if c == 0: # 假设类别0是 helmet
                    helmet_count += 1
                    color = (0, 255, 0) # 绿色
                else: # 假设类别1是 head (没戴安全帽)
                    no_helmet_count += 1
                    color = (0, 0, 255) # 红色
                    # 这里可以添加报警逻辑
                
                annotator.box_label(xyxy, label, color=color)

        # 绘制多边形选区
        if len(self.polygon_points) > 0:
            cv2.polylines(annotator.im, [np.array(self.polygon_points, np.int32)], True, (255, 0, 0), 2)

        return annotator.result()

    def stop(self):
        self.is_running = False
```

### 3. 主界面实现 (`main_window.py`)

这是 PyQt5 的核心部分，处理按钮点击、滑块调节和显示视频流。

```python
import sys
from PyQt5.QtWidgets import QMainWindow, QApplication, QFileDialog, QMessageBox
from PyQt5.QtGui import QImage, QPixmap
from ui_main import Ui_MainWindow # 假设你用 QtDesigner 设计了界面并转为 py
from detect_thread import DetectThread

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        
        # 初始化变量
        self.thread = DetectThread()
        
        # 连接信号与槽
        self.btn_start.clicked.connect(self.start_detection)
        self.btn_pause.clicked.connect(self.pause_detection)
        self.btn_stop.clicked.connect(self.stop_detection)
        self.slider_conf.valueChanged.connect(self.change_conf)
        self.thread.send_img.connect(self.update_image)
        
        # 鼠标事件用于绘制多边形
        self.video_label.mousePressEvent = self.mouse_press_event
        self.polygon_points = []

    def start_detection(self):
        # 获取文件路径
        source, _ = QFileDialog.getOpenFileName(self, "选择视频或图片", ".", "Video Files (*.mp4 *.avi);;Image Files (*.jpg *.png)")
        if source:
            self.thread.source = source
            self.thread.start()

    def update_image(self, img):
        # 将 OpenCV 图像转换为 Qt 图像显示
        show = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        showImage = QImage(show.data, show.shape[1], show.shape[0], QImage.Format_RGB888)
        self.video_label.setPixmap(QPixmap.fromImage(showImage))

    def change_conf(self, value):
        # 动态调节置信度
        self.thread.conf_thres = value / 100.0

    def mouse_press_event(self, event):
        # 简单的多边形绘制逻辑
        if event.button() == Qt.LeftButton:
            x, y = event.x(), event.y()
            self.polygon_points.append([x, y])
            # 这里需要重绘视频帧以显示绘制的点，略去具体绘图代码
```

### 4. 数据库与登录 (`login.py`)

使用 SQLite 进行简单的用户管理。

```python
import sqlite3
from PyQt5.QtWidgets import QDialog, QMessageBox

class LoginDialog(QDialog):
    def __init__(self):
        super().__init__()
        self.init_db()
        # 这里需要加载你的登录界面 UI

    def init_db(self):
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute('''CREATE TABLE IF NOT EXISTS users
                     (username text PRIMARY KEY, password text)''')
        conn.commit()
        conn.close()

    def login(self):
        username = self.input_user.text()
        password = self.input_pwd.text()
        
        conn = sqlite3.connect('users.db')
        c = conn.cursor()
        c.execute("SELECT * FROM users WHERE username=? AND password=?", (username, password))
        if c.fetchone():
            self.accept() # 登录成功，关闭对话框
        else:
            QMessageBox.warning(self, "错误", "用户名或密码错误")
```

### 5. 如何运行

1.  **模型准备**：你需要先训练一个 YOLOv5 模型（例如 `hatV11_best.pt`），类别为 `helmet` 和 `head`。
2.  **界面设计**：使用 Qt Designer 设计 `.ui` 文件，然后使用 `pyuic5` 转换为 `ui_main.py`。
3.  **整合**：将上述代码片段整合到你的项目中。
4.  **启动**：在 `main.py` 中先弹出 `LoginDialog`，验证通过后再显示 `MainWindow`。

这套代码结构清晰，涵盖了从底层检测到上层交互的所有关键点，非常适合用来做毕业设计或课程作业。
