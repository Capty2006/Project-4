import numpy as np
import cv2

#รายชื่อAIที่สามารถอ่านได้
CLASSES = ["BACKGROUND", "AEROPLANE", "BICYCLE", "BIRD", "BOAT",
	"BOTTLE", "BUS", "CAR", "CAT", "CHAIR", "COW", "DININGTABLE",
	"DOG", "HORSE", "MOTORBIKE", "PERSON", "POTTEDPLANT", "SHEEP",
	"SOFA", "TRAIN", "TVMONITOR"]
COLORS = np.random.uniform(0,100, size=(len(CLASSES), 3))
net = cv2.dnn.readNetFromCaffe("MobileNetSSD.prototxt","MobileNetSSD.caffemodel")


#เลือกเปิดวีดิโอหรือกล้อง
cap = cv2.VideoCapture(0)

while True:
    #อ่านในแต่ละเฟรม
    ret, frame = cap.read()
    if ret:
        (h,w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 0.007843, (300,300), 127.5)
        net.setInput(blob)
        #feedเข้าmodelพร้อมได้ผลลัพธ์ทั้งหมดเก็บในตัวแปร detections
        detections = net.forward()

        for i in np.arange(0, detections.shape[2]):
            percent = detections[0,0,i,2]
            #ปัดทศนิยมเปอร์เซ็นต์ที่สูงกว่า 0.5
            if percent > 0.5:
                class_index = int(detections[0,0,i,1])
                box = detections[0,0,i,3:7]*np.array([w,h,w,h])
                (startX, startY, endX, endY) = box.astype("int")
                 
                 #กรอบและชื่อ ปรับตามที่ต้องการได้
                label = "{} [{:.2f}%]".format(CLASSES[class_index], percent*100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[class_index], 2)
                cv2.rectangle(frame, (startX-1, startY-30), (endX+1, startY), COLORS[class_index], cv2.FILLED)
                y = startY - 15 if startY-15>15 else startY+15
                cv2.putText(frame, label, (startX+20, y+5), cv2.FONT_HERSHEY_DUPLEX, 0.6, (255,255,255), 1)
				

        cv2.imshow("Frame", frame)
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break
#ปิดกล้องและเครียร์memoly   
cap.release()
cv2.destroyWindow()