import cv2
import mediapipe as mp
cap = cv2.VideoCapture(0)
cap.set(4,1366)
cap.set(3,768)
mpDraw = mp.solutions.drawing_utils
mpFaceMesh = mp.solutions.face_mesh
mpHands = mp.solutions.hands
hands = mpHands.Hands()
faceMesh = mpFaceMesh.FaceMesh()
drawSpec = mpDraw.DrawingSpec(thickness=1,circle_radius=1)
classNames = []
classFile = 'coco.names'
with open(classFile,'rt') as f:
	classNames = f.read().rstrip('\n').split('\n')
configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'
net = cv2.dnn_DetectionModel(configPath,weightsPath)
net.setInputSize(320,320)
net.setInputScale(1.0/ 127.5)
net.setInputMean((127.5,127.5,127.5))
net.setInputSwapRB(True)
while True:
	success,img = cap.read()
	classIds, confs, bbox = net.detect(img,confThreshold=0.5)
	print(classIds,bbox)
	if len(classIds) != 0:
		for classIds,confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
			cv2.rectangle(img,box,color=(0,255,0),thickness=2)
			cv2.putText(img,classNames[classIds-1],(box[0]+10,box[1]+30),cv2.FONT_ITALIC,1,(0,255,0),2)
			cv2.putText(img,f"{str(round(confidence*100,2))}%",(box[0]+10,box[1]+60),cv2.FONT_ITALIC,1,(0,255,0),2)
			print(classNames[classIds-1])
	imgRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
	results = faceMesh.process(imgRGB)
	resultsa = hands.process(imgRGB)
	if results.multi_face_landmarks:
		for faceLms in results.multi_face_landmarks:
			mpDraw.draw_landmarks(img,faceLms,mpFaceMesh.FACEMESH_CONTOURS,drawSpec,drawSpec)
	if resultsa.multi_hand_landmarks:
		for handLms in resultsa.multi_hand_landmarks:
			mpDraw.draw_landmarks(img,handLms,mpHands.HAND_CONNECTIONS)
	cv2.imshow("Output",img)
	cv2.waitKey(1)