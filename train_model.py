import cv2
import numpy as np
import os
from PIL import Image
recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'
def getImagesWithId(path):
    immagePaths= [os.path.join(path, f) for f in os.listdir(path)]
    print(immagePaths)
    faces= []
    IDs= []
    for immagePath in immagePaths:
        faceImg = Image.open(immagePath).convert('L')
        # faceNP = np.array(faceImg, 'unit8')
        faceNP = np.array(faceImg)
        print(faceNP)
        Id= int(os.path.split(immagePath)[-1].split('.')[1])
        faces.append(faceNP)
        IDs.append(Id)
        cv2.imshow('training', faceNP)
        cv2.waitKey(10)
    return faces,IDs
faces,IDs=getImagesWithId(path)
recognizer.train(faces, np.array(IDs))
if not os.path.exists('recognizer'):
    os.makedirs(recognizer)
recognizer.save('recognizer/trainingData.yml')
cv2.destroyAllWindows()


