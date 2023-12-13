import cv2 as cv
import numpy as np
import mediapipe as mp
import face1

 
# 手指检测
# point1-手掌0点位置，point2-手指尖点位置，point3手指根部点位置
def finger_stretch_detect(point1, point2, point3):
    result = 0
    #计算向量的L2范数
    dist1 = np.linalg.norm((point2 - point1), ord=2)
    dist2 = np.linalg.norm((point3 - point1), ord=2)
    if dist2 > dist1:
        result = 1
    
    return result
    
# 检测手势
def detect_hands_gesture(result):
    gesture="Can't recognize..."
    if (result[0] == 0) and (result[1] == 1)and (result[2] == 1) and (result[3] == 0) and (result[4] == 0):
        gesture = "Yes"    
    elif (result[0] == 1) and (result[1] == 0)and (result[2] == 1) and (result[3] == 1) and (result[4] == 1):
        gesture = "OK"
    return gesture
 
 
def detect(frame):
    
    # 加载手部检测函数
    mpHands = mp.solutions.hands
    hands = mpHands.Hands()
    # 加载绘制函数，并设置手部关键点和连接线的形状、颜色
    mpDraw = mp.solutions.drawing_utils
    handLmsStyle = mpDraw.DrawingSpec(color=(0, 0, 255), thickness=int(5))
    handConStyle = mpDraw.DrawingSpec(color=(0, 255, 0), thickness=int(10))
    
    figure = np.zeros(5)
    landmark = np.empty((21, 2))
    #mediaPipe的图像要求是RGB，所以此处需要转换图像的格式
    frame_RGB = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
    result = hands.process(frame_RGB)
    #读取视频图像的高和宽
    frame_height = frame.shape[0]
    frame_width  = frame.shape[1]
    
    #print(result.multi_hand_landmarks)
    #如果检测到手
    if result.multi_hand_landmarks:
        #为每个手绘制关键点和连接线
        for i, handLms in enumerate(result.multi_hand_landmarks):
            mpDraw.draw_landmarks(frame, 
                                    handLms, 
                                    mpHands.HAND_CONNECTIONS,
                                    landmark_drawing_spec=handLmsStyle,
                                    connection_drawing_spec=handConStyle)

            for j, lm in enumerate(handLms.landmark):
                xPos = int(lm.x * frame_width)
                yPos = int(lm.y * frame_height)
                landmark_ = [xPos, yPos]
                landmark[j,:] = landmark_

            # 通过判断手指尖与手指根部到0位置点的距离判断手指是否伸开(拇指检测到17点的距离)
            for k in range (5):
                if k == 0:
                    figure_ = finger_stretch_detect(landmark[17],landmark[4*k+2],landmark[4*k+4])
                else:    
                    figure_ = finger_stretch_detect(landmark[0],landmark[4*k+2],landmark[4*k+4])

                figure[k] = figure_
            print(figure,'\n')

            gesture_result = detect_hands_gesture(figure)
            cv.putText(frame, f"{gesture_result}", (30, 60*(i+1)), cv.FONT_HERSHEY_COMPLEX, 2, (255 ,255, 0), 5)
    cv.imwrite('./gestures.png', frame)       
    
 
if __name__ == '__main__':
    # 接入USB摄像头时，注意修改cap设备的编号
    flag=0
    number=int(input("\n1 for registration,2 for recognition: "))
    if number==2:
        face1.face_recogntion()
        ret,frame=cv.VideoCapture(0).read()
        detect(frame)
        cv.VideoCapture(0).release()
    elif number==1:
        face1.face_registration()
        