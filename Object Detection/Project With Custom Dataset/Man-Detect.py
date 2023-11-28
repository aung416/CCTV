from ultralytics import YOLO
import cv2
import cvzone
import math

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4,480)

# cap = cv2.VideoCapture("../images/jahangir1.jpg")


model = YOLO('../Yolo-Weights/custom.pt')
model = YOLO('../Yolo-Weights/yolov8n.pt')

classNames=  [ 'aung','biazid', 'habib', 'jahangir', 'mahafuj', 'nadim', 'naim']



while True:
    success, video = cap.read()
    results = model(video, stream= True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1,y1,x2,y2 = box.xyxy[0]
            x1,y1,x2,y2 = int(x1),int(y1),int(x2), int(y2)
            # cv2.rectangle(video,(x1,y1),(x2,y2), (0,255,0),3)
            w = x2 - x1
            h = y2 - y1
            cvzone.cornerRect(video,(x1,y1,w,h))
            conf = math.ceil(box.conf[0]*100)/100
            cvzone.putTextRect(video, f'{conf}', (max(0,x1),max(35,y1)))

            cls = int(box.cls[0])
            cvzone.putTextRect(video, f'{classNames[cls]} {conf}', (max(0,x1),max(35,y1)),scale=1.5,thickness=2)

    cv2.imshow("Image", video)
    cv2.waitKey(1)
