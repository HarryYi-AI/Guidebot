#!/usr/bin/env python3
# 指定脚本使用系统环境中的 python3 解释器运行，Linux/树莓派上直接执行文件时会用到。
# -*-coding: utf-8 -*-
# 声明源代码文件使用 UTF-8 编码，方便文件里写中文注释。
"""
    @Project: python-learning-notes
    @File   : openpose_for_image_test.py
    @Author : panjq
    @E-mail : pan_jinquan@163.com
    @Date   : 2019-07-29 21:50:17
"""
# 上面的三引号字符串是文件说明，不参与程序逻辑，只记录项目、作者、日期等信息。

import time
# 导入 time 模块，用来统计每一帧处理耗时，从而计算 FPS。
import cv2 as cv
# 导入 OpenCV，并起别名 cv；后面摄像头、DNN 推理、画框、显示窗口都靠它。
import numpy as np
# 导入 NumPy，并起别名 np；这里主要用于生成随机颜色数组。

######################### Detection ##########################
# 下面这一段是目标检测部分，使用 SSD MobileNet 模型识别 COCO 数据集里的常见物体。

# load the COCO class names
# 打开类别名称文件，每一行是一个 COCO 类别名称，比如 person、car、chair 等。
with open('object_detection_coco.txt', 'r') as f:
    # 读取整个文件内容，并按换行符切分成列表，得到 class_names。
    class_names = f.read().split('\n')

# get a different color array for each of the classes
# 为每个类别随机生成一个 BGR 颜色，后面画检测框时不同类别显示不同颜色。
COLORS = np.random.uniform(0, 255, size=(len(class_names), 3))

# load the DNN modelimage
# 使用 OpenCV DNN 模块加载 TensorFlow 目标检测模型和配置文件。
model = cv.dnn.readNet(
    model='frozen_inference_graph.pb',
    # frozen_inference_graph.pb 是已经训练好的 SSD MobileNet 权重文件。
    config='ssd_mobilenet_v2_coco.txt',
    # ssd_mobilenet_v2_coco.txt 是模型结构/配置文件。
    framework='TensorFlow'
    # 告诉 OpenCV 这个模型来自 TensorFlow 框架。
)

######################### openpose ##########################
# 下面这一段是人体姿态估计部分，使用 OpenCV DNN 加载 OpenPose 风格的 TensorFlow 模型。

BODY_PARTS = {
    # BODY_PARTS 把人体关键点名称映射到模型输出通道编号。
    "Nose": 0,
    # 鼻子关键点编号。
    "Neck": 1,
    # 脖子关键点编号。
    "RShoulder": 2,
    # 右肩关键点编号。
    "RElbow": 3,
    # 右肘关键点编号。
    "RWrist": 4,
    # 右手腕关键点编号。
    "LShoulder": 5,
    # 左肩关键点编号。
    "LElbow": 6,
    # 左肘关键点编号。
    "LWrist": 7,
    # 左手腕关键点编号。
    "RHip": 8,
    # 右髋/右胯关键点编号。
    "RKnee": 9,
    # 右膝关键点编号。
    "RAnkle": 10,
    # 右脚踝关键点编号。
    "LHip": 11,
    # 左髋/左胯关键点编号。
    "LKnee": 12,
    # 左膝关键点编号。
    "LAnkle": 13,
    # 左脚踝关键点编号。
    "REye": 14,
    # 右眼关键点编号。
    "LEye": 15,
    # 左眼关键点编号。
    "REar": 16,
    # 右耳关键点编号。
    "LEar": 17,
    # 左耳关键点编号。
    "Background": 18
    # 背景通道编号，不属于人体关节。
}

POSE_PAIRS = [
    # POSE_PAIRS 定义哪些关键点之间需要连线，从而画出人体骨架。
    ["Neck", "RShoulder"],
    # 脖子连接右肩。
    ["Neck", "LShoulder"],
    # 脖子连接左肩。
    ["RShoulder", "RElbow"],
    # 右肩连接右肘。
    ["RElbow", "RWrist"],
    # 右肘连接右手腕。
    ["LShoulder", "LElbow"],
    # 左肩连接左肘。
    ["LElbow", "LWrist"],
    # 左肘连接左手腕。
    ["Neck", "RHip"],
    # 脖子连接右髋。
    ["RHip", "RKnee"],
    # 右髋连接右膝。
    ["RKnee", "RAnkle"],
    # 右膝连接右脚踝。
    ["Neck", "LHip"],
    # 脖子连接左髋。
    ["LHip", "LKnee"],
    # 左髋连接左膝。
    ["LKnee", "LAnkle"],
    # 左膝连接左脚踝。
    ["Neck", "Nose"],
    # 脖子连接鼻子。
    ["Nose", "REye"],
    # 鼻子连接右眼。
    ["REye", "REar"],
    # 右眼连接右耳。
    ["Nose", "LEye"],
    # 鼻子连接左眼。
    ["LEye", "LEar"]
    # 左眼连接左耳。
]

# 使用 OpenCV DNN 加载 OpenPose 的 TensorFlow 模型文件。
net = cv.dnn.readNetFromTensorflow("graph_opt.pb")


def Target_Detection(image):
    # 定义目标检测函数，输入一帧图像，输出画好检测框的图像。
    image_height, image_width, _ = image.shape
    # 获取图像高度、宽度和通道数；这里通道数用不到，所以用 _ 接收。

    # create blob from image
    # 把原始图像转换成神经网络需要的 blob 格式，并缩放到 300x300。
    blob = cv.dnn.blobFromImage(
        image=image,
        # 输入原始图像。
        size=(300, 300),
        # SSD MobileNet 模型要求的输入尺寸。
        mean=(104, 117, 123),
        # 减去训练时使用的均值，让输入分布更接近模型训练数据。
        swapRB=True
        # OpenCV 读图默认是 BGR，这里交换 R/B 通道以适配模型输入。
    )

    # 将预处理后的 blob 输入到目标检测网络。
    model.setInput(blob)

    # 执行前向推理，得到检测结果。
    output = model.forward()

    # loop over each of the detections
    # 遍历模型输出的每一个候选检测框。
    for detection in output[0, 0, :, :]:
        # extract the confidence of the detection
        # detection[2] 是该检测结果的置信度，越大表示模型越确定。
        confidence = detection[2]

        # draw bounding boxes only if the detection confidence is above...
        # ... a certain threshold, else skip
        # 只保留置信度大于 0.4 的结果，过滤掉不太可靠的检测。
        if confidence > .4:
            # get the class id
            # detection[1] 是类别编号，例如人、车、椅子等。
            class_id = detection[1]

            # map the class id to the class
            # 类别编号从 1 开始，而 Python 列表从 0 开始，所以这里减 1。
            class_name = class_names[int(class_id) - 1]

            # 根据类别编号取一个随机颜色，用来画框和文字。
            color = COLORS[int(class_id)]

            # get the bounding box coordinates
            # detection[3] 和 detection[4] 是归一化后的左上角 x、y 坐标。
            box_x = detection[3] * image_width
            # 把归一化 x 坐标转换为原图像素坐标。
            box_y = detection[4] * image_height
            # 把归一化 y 坐标转换为原图像素坐标。

            # get the bounding box width and height
            # detection[5] 和 detection[6] 是归一化后的右下角 x、y 坐标。
            box_width = detection[5] * image_width
            # 把归一化右下角 x 坐标转换为原图像素坐标。
            box_height = detection[6] * image_height
            # 把归一化右下角 y 坐标转换为原图像素坐标。

            # draw a rectangle around each detected object
            # 在图像上画出目标检测框，左上角是 (box_x, box_y)，右下角是 (box_width, box_height)。
            cv.rectangle(image, (int(box_x), int(box_y)), (int(box_width), int(box_height)), color, thickness=2)

            # put the class name text on the detected object
            # 在检测框上方写出类别名称。
            cv.putText(image, class_name, (int(box_x), int(box_y - 5)), cv.FONT_HERSHEY_SIMPLEX, 1, color, 2)

    # 返回已经画好检测框和类别文字的图像。
    return image


def openpose(frame):
    # 定义姿态估计函数，输入一帧图像，输出画好人体骨架的图像。
    frameHeight, frameWidth = frame.shape[:2]
    # 获取当前帧的高度和宽度，用于把模型输出坐标映射回原图。

    # 把图像转换成 OpenPose 模型需要的 blob，并设置为网络输入。
    net.setInput(
        cv.dnn.blobFromImage(
            frame,
            # 输入原始视频帧。
            1.0,
            # 缩放因子，这里保持像素值不额外缩放。
            (368, 368),
            # OpenPose 模型常用输入尺寸。
            (127.5, 127.5, 127.5),
            # 减去均值，把像素中心移到模型更适应的范围。
            swapRB=True,
            # 交换 R/B 通道，适配模型输入。
            crop=False
            # 不裁剪图像，保持整幅图缩放到指定大小。
        )
    )

    # 执行姿态估计网络的前向推理，得到所有输出通道。
    out = net.forward()

    # MobileNet output [1, 57, -1, -1], we only need the first 19 elements
    # 模型输出有 57 个通道，这里只取前 19 个关键点热力图通道。
    out = out[:, :19, :, :]

    # 确保模型输出的关键点通道数和 BODY_PARTS 定义的 19 个关键点一致。
    assert (len(BODY_PARTS) == out.shape[1])

    # 用列表保存每个人体关键点的坐标；识别不到的关键点保存为 None。
    points = []

    # 遍历 19 个关键点通道，逐个寻找每个关键点最可能的位置。
    for i in range(len(BODY_PARTS)):
        # Slice heatmap of corresponging body's part.
        # 取出第 i 个关键点对应的热力图，热力图中数值越大表示该位置越可能是该关键点。
        heatMap = out[0, i, :, :]

        # Originally, we try to find all the local maximums. To simplify a sample
        # we just find a global one. However only a single pose at the same time
        # could be detected this way.
        # 简化处理：只找整张热力图里置信度最高的一个点，因此这个示例基本只适合单人姿态。
        _, conf, _, point = cv.minMaxLoc(heatMap)

        # 将热力图上的 x 坐标按比例映射回原始画面宽度。
        x = (frameWidth * point[0]) / out.shape[3]

        # 将热力图上的 y 坐标按比例映射回原始画面高度。
        y = (frameHeight * point[1]) / out.shape[2]

        # Add a point if it's confidence is higher than threshold.
        # 如果该关键点置信度大于 0.2，就保存坐标；否则保存 None 表示没识别到。
        points.append((int(x), int(y)) if conf > 0.2 else None)

    # 遍历预定义的骨架连接关系，尝试把关键点连成人体骨架。
    for pair in POSE_PAIRS:
        # 当前连接的起点关键点名称。
        partFrom = pair[0]

        # 当前连接的终点关键点名称。
        partTo = pair[1]

        # 确保起点名称在 BODY_PARTS 字典里，防止写错名字。
        assert (partFrom in BODY_PARTS)

        # 确保终点名称在 BODY_PARTS 字典里，防止写错名字。
        assert (partTo in BODY_PARTS)

        # 根据起点名称找到它在 points 列表里的下标。
        idFrom = BODY_PARTS[partFrom]

        # 根据终点名称找到它在 points 列表里的下标。
        idTo = BODY_PARTS[partTo]

        # 只有当两个关键点都被识别出来时，才画线连接它们。
        if points[idFrom] and points[idTo]:
            # 在两个关键点之间画绿色线段，表示人体骨架的一条边。
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)

            # 在起点关键点位置画红色实心小圆点。
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

            # 在终点关键点位置画红色实心小圆点。
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    # 返回已经画好人体骨架的图像。
    return frame


if __name__ == '__main__':
    # 只有直接运行 test.py 时，下面的主程序才会执行；如果被别的文件 import，则不会执行。

    # 打开编号为 0 的摄像头，通常是电脑或小车上的默认 USB 摄像头。
    capture = cv.VideoCapture(0)

    # 获取当前 OpenCV 的版本号字符串，比如 3.x 或 4.x。
    cv_edition = cv.__version__

    # 如果是 OpenCV 3，用 XVID 编码格式设置摄像头视频流。
    if cv_edition[0] == '3':
        # 设置摄像头编码格式为 XVID。
        capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'XVID'))
    else:
        # 如果不是 OpenCV 3，按 OpenCV 4 的写法设置 MJPG 编码格式。
        capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter.fourcc('M', 'J', 'P', 'G'))

    # 设置摄像头画面宽度为 640 像素。
    capture.set(cv.CAP_PROP_FRAME_WIDTH, 640)

    # 设置摄像头画面高度为 480 像素。
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, 480)

    # state 用来控制当前模式；True 表示目标检测模式，False 表示 OpenPose 姿态估计模式。
    state = True

    # 只要摄像头处于打开状态，就持续读取视频帧。
    while capture.isOpened():
        # 记录当前帧开始处理的时间，用于后面计算 FPS。
        start = time.time()

        # 从摄像头读取一帧；ret 表示是否读取成功，frame 是图像数据。
        ret, frame = capture.read()

        # 等待 10 毫秒读取键盘按键；& 0xFF 用来兼容不同平台的按键编码。
        action = cv.waitKey(10) & 0xFF

        # 如果 state 为 True，执行目标检测。
        if state == True:
            # 对当前帧进行目标检测，并把检测框画到 frame 上。
            frame = Target_Detection(frame)

            # 在画面上写出当前模式名称 Detection。
            cv.putText(frame, "Detection", (240, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)
        else:
            # 如果 state 为 False，执行 OpenPose 姿态估计。
            frame = openpose(frame)

            # 在画面上写出当前模式名称 Openpose。
            cv.putText(frame, "Openpose", (240, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 1)

        # 如果按下 q 或 Q，跳出循环，结束程序。
        if action == ord('q') or action == ord('Q'):
            break

        # 如果按下 f 或 F，就在目标检测模式和姿态估计模式之间切换。
        if action == ord('f') or action == ord('F'):
            state = not state

        # 记录当前帧处理结束的时间。
        end = time.time()

        # 用 1 除以单帧耗时，得到每秒处理帧数 FPS。
        fps = 1 / (end - start)

        # 拼接要显示在画面上的 FPS 文本。
        text = "FPS : " + str(int(fps))

        # 把 FPS 写到画面左上角。
        cv.putText(frame, text, (20, 30), cv.FONT_HERSHEY_SIMPLEX, 0.9, (100, 200, 200), 1)

        # 显示处理后的当前帧。
        cv.imshow('frame', frame)

    # 循环结束后释放摄像头资源。
    capture.release()

    # 关闭所有 OpenCV 创建的显示窗口。
    cv.destroyAllWindows()
