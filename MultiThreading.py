import threading
import os
import cv2
from tensorflow import keras
import numpy as np

def Predict(DIR,frame_name):
	model = keras.models.load_model("CNNgender.tf")
	for file in os.listdir(DIR):
	    frame = cv2.imread((os.path.join(DIR,file)),cv2.IMREAD_GRAYSCALE)
	    x = cv2.resize(frame, (150, 150))
	    x = np.array(x).reshape(-1, 150, 150, 1)
	    x = x/255
	    l1 = model.predict(x)
	    if l1<0.5:
	        print("Male")
	    else:
	        print("Female")
	    cv2.imshow(frame_name,frame)
	    cv2.waitKey(1)
   

if __name__ == '__main__':
    thread1 = threading.Thread(target=Predict ,args=("TestImage_0","0"))
    thread2 = threading.Thread(target=Predict, args=("TestImage_1","1"))
    thread1.start()
    thread2.start()
    thread1.join()
    thread2.join()