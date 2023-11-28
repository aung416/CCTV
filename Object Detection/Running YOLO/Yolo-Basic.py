from ultralytics import YOLO
import  cv2

model = YOLO('../Yolo-Weights/best (2).pt')
results = model("images/jahangir1.jpg", show = True)

cv2.waitKey(0)
