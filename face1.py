import os

import cv2
import insightface
import numpy as np
from sklearn import preprocessing


class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
        """
        人脸识别工具类
        :param gpu_id: 正数为GPU的ID，负数为使用CPU
        :param face_db: 人脸库文件夹
        :param threshold: 人脸识别阈值
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        """
        self.gpu_id = gpu_id
        self.face_db = face_db
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size

        # 加载人脸识别模型，当allowed_modules=['detection', 'recognition']时，只单纯检测和识别
        self.model = insightface.app.FaceAnalysis(root='./',
                                                  allowed_modules=None,
                                                  providers=['CUDAExecutionProvider'])
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        # 人脸库的人脸特征
        self.faces_embedding = list()
        # 加载人脸库中的人脸
        self.load_faces(self.face_db)

    # 加载人脸库中的人脸
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                user_name = file.split(".")[0]
                face = self.model.get(input_image)[0]
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                self.faces_embedding.append({
                    "user_name": user_name,
                    "feature": embedding
                })
    @staticmethod
    def my_draw_on(img, faces,results):
        import cv2
        dimg = img.copy()
        images_array=list()
        for i in range(len(faces)):
            face = faces[i]
            box = face.bbox.astype(int)
            color = (0, 0,255 )
            cv2.rectangle(dimg, (box[0], box[1]), (box[2], box[3]), color, 2)
            if face.kps is not None:
                kps = face.kps.astype(int)
                #print(landmark.shape)
                for l in range(kps.shape[0]):
                    color = (0, 0, 255)
                    if l == 0 or l == 3:
                        color = (0, 255, 0)
                    cv2.circle(dimg, (kps[l][0], kps[l][1]), 1, color, 2)
            img_cut=dimg[box[1]:box[3]+1,box[0]:box[2]+1].copy()
            cv2.putText(dimg,'%s'%results[i], (box[0], box[1]),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,255,0),1)
            images_array.append(img_cut)
        return dimg,images_array
    # 人脸识别
    def recognition(self, image):
        faces = self.model.get(image)
        results = list()
        for face in faces:
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            user_name = "unknown"
            for com_face in self.faces_embedding:
                r = self.feature_compare(embedding, com_face["feature"], self.threshold)
                if r:
                    user_name = com_face["user_name"]
            results.append(user_name)
        img,images_section=FaceRecognition.my_draw_on(image,faces,results)
        return img,images_section,results
    @staticmethod
    def feature_compare(feature1, feature2, threshold):
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        if dist < threshold:
            return True
        else:
            return False

    def register(self, image, user_name):
        faces = self.model.get(image)
        if len(faces) != 1:
            return '图片检测不到人脸'
        # 判断人脸是否存在
        embedding = np.array(faces[0].embedding).reshape((1, -1))
        embedding = preprocessing.normalize(embedding)
        is_exits = False
        for com_face in self.faces_embedding:
            r = self.feature_compare(embedding, com_face["feature"], self.threshold)
            if r:
                is_exits = True
        if is_exits:
            return '该用户已存在'
        # 符合注册条件保存图片，同时把特征添加到人脸特征库中
        cv2.imencode('.jpg', image)[1].tofile(os.path.join(self.face_db, '%s.jpg' % user_name))
        self.faces_embedding.append({
            "user_name": user_name,
            "feature": embedding
        })
        return "success"
#封装人脸识别，输入一张合照，如果全能识别出熟悉的人，则输出一张框出各个人脸的合照以及各个人分别的照片
def face_recogntion():
    flag=0
    if not os.path.exists("./recognition"):
            os.makedirs('./recognition')
    if not os.path.exists("./sections"):
            os.makedirs('./sections')   
    face_recognitio = FaceRecognition()
    cap = cv2.VideoCapture(0)
    while flag==0:
        ret,frame=cap.read() 
        if not (frame is None):
            img_rec ,sections_images,results= face_recognitio.recognition(frame)
            if (not ("unknown" in results ))and len(results)==2:
                cv2.imwrite(f'./recognition/results.png', img_rec)
                for i in range(2):
                    cv2.imwrite(f'./sections/{i}_output.png', sections_images[i])
                flag=1
    
def face_registration():
    cap = cv2.VideoCapture(0)
    ret,frame=cap.read() 
    face_recognitio = FaceRecognition()
    name=input("Please tell me his or her name?")
    result = face_recognitio.register(frame, user_name=name)
    print(result)