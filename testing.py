import cv2
import os
import matplotlib.pyplot as plt
import tensorflow.keras as tf
import numpy as np

my_model = tf.models.load_model('digits_cnn3.model.h5')

list1 = []

def digits_prediction2(folder = ""):
    img_files = os.listdir(folder)
    #my_list = []
    
    for image in img_files:
        img = cv2.imread(folder+"/"+image,0)
        img = cv2.resize(img,(28,28))
        img = cv2.bitwise_not(img)
        plt.imshow(img,cmap='gray')
        
        display_img = plt.show()
        ex1 = my_model.predict(np.array([img]))
        var = np.argmax(ex1)
#         my_list.append(var)
        json = {"file name": image, 'predicted number ': var }
        list1.append(json)
        
    return list1
