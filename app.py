import streamlit as st
def intro():
    import streamlit as st

    st.write("# Welcome to Face Recognite by The Kiet! 👋")
    st.sidebar.success("Select a step to train.")

def mapping_demo():
    import streamlit as st
    import cv2
    import numpy as np
    import csv
    import os
    import av
    from streamlit_webrtc import webrtc_streamer

    header = ["ID","Name"]
    def writeToCSV(id,name):
        f = open("people.csv","a",newline='')
        writer = csv.writer(f)
        tup=[id,name]
        writer.writerow(tup)
        f.close()
    
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    class VideoProcessor:
        def recv(self,frame):
            sampleNum = 0
            # while(True):
            frm = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(cv2.cvtColor(frm,cv2.COLOR_BGR2GRAY),1.3, 5)
            for x,y,w,h in faces:
                while(True):
                    cv2.rectangle(frm, (x, y), (x+w, y+h), (255, 0, 0), 2)
                    if not os.path.exists('dataSet'):
                        os.makedirs('dataSet')
                    sampleNum +=1
                    cv2.imwrite('dataSet/'+str(name)+'.' + str(id)+'.' + str(sampleNum) + '.jpg', gray[y:y+h,x:x+w])
                    if(sampleNum>=100):
                        break
            return av.VideoFrame.from_ndarray(frm,format='bgr24')
    id = st.text_input("Id cần nhập")
    name=st.text_input("Tên cần nhập")
    # insertOrUpdate(id,name)
    if st.button('Click bỏ vô csv'):
        writeToCSV(id,name)
    webrtc_streamer(key='key', video_processor_factory=VideoProcessor, rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    st.write('nếu như webcam trên streamlit dùng không được thì dùng camera sẵn trên desktop')
    if st.button('nhấn vào để lấy data set'):
        cap =cv2.VideoCapture(0)
        sampleNum = 0
        while(True):
            ret,frame =cap.read()
            gray= cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                if not os.path.exists('dataSet'):
                    os.makedirs('dataSet')
                sampleNum +=1
                cv2.imwrite('dataSet/'+str(name)+'.' + str(id)+'.' + str(sampleNum) + '.jpg', gray[y:y+h,x:x+w])
            cv2.imshow('frame',frame)
            cv2.waitKey(1) #& 0xFF == ord('q')):
            if(sampleNum>=150):
                    break
        cap.release()
        cv2.destroyAllWindows
    # cv2.destroyAllWindows
def plotting_demo():
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
            # cv2.imshow('training', faceNP)
            # cv2.waitKey(10)
        return faces,IDs
    faces,IDs=getImagesWithId(path)
    if st.button('Nhấn vào để training'):
        recognizer.train(faces, np.array(IDs))
        if not os.path.exists('recognizer'):
            os.makedirs(recognizer)
        recognizer.save('recognizer/trainingData.yml')
    # cv2.destroyAllWindows()

def data_frame_demo():
    import cv2
    import csv
    from streamlit_webrtc import webrtc_streamer
    import streamlit as st
    import av
    from PIL import Image
    face_cascade =cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('recognizer/trainingData.yml')
    def getId(id):
        url = open("people.csv", "r")
        profile = None
        read_file = csv.reader(url)
        for row in read_file:
            if(row[0] == str(id)):
                profile = row
        url.close()
        return profile

    fontface= cv2.FONT_HERSHEY_SIMPLEX
    class VideoProcessor:
        def recv(self,frame):
            frm = frame.to_ndarray(format="bgr24")
            gray  = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frm, (x, y), (x + w, y + h), (0,255, 0), 2)
                roi_gray =gray[y:y+h,x:x+w]
                id, confidence = recognizer.predict(roi_gray)
                if confidence < 40:
                    # profile = getProfile(id)
                    profile = getId(id)
                    # print(profile)
                    if(profile != None):
                        cv2.putText(frm,""+profile[1], (x+10,y+h+30), fontface, 1,(0,255,0),2)
                else:
                    cv2.putText(frm,"Unknow",(x+10,y+h+30),fontface,1,(0,0,255),2)
                    # cv2.imshow('Image',frame)
            return av.VideoFrame.from_ndarray(frm,format='bgr24')
            
    st.write("Nhận diện khuôn mặt")
    webrtc_streamer(key='key', video_processor_factory=VideoProcessor, rtc_configuration={  # Add this line
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    })
    st.write('nếu như webcam trên streamlit dùng không được thì dùng camera sẵn trên desktop')
    if st.button('Nhấn vào đây để test'):
        cap = cv2.VideoCapture(0)
        while(True):
            ret, frame = cap.read()
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,scaleFactor=1.5, minNeighbors=5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255, 0), 2)
                roi_gray =gray[y:y+h,x:x+w]
                id, confidence = recognizer.predict(roi_gray)
                if confidence < 40:
                    # profile = getProfile(id)
                    profile = getId(id)
                    # print(profile)
                    if(profile != None):
                        cv2.putText(frame,""+profile[1], (x+10,y+h+30), fontface, 1,(0,255,0),2)
                else:
                    cv2.putText(frame,"Unknow",(x+10,y+h+30),fontface,1,(0,0,255),2)
                    # cv2.imshow('Image',frame)
                if (cv2.waitKey(1)) & 0xFF==ord('q'):
                    break
            cv2.imshow('frame',frame)
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()
page_names_to_funcs = {
    "Introduce": intro,
    "Step 1: Get data set": mapping_demo,
    "Step 2: Training data": plotting_demo,
    "Step 3: Test": data_frame_demo
}

demo_name = st.sidebar.selectbox("Choose a demo", page_names_to_funcs.keys())
page_names_to_funcs[demo_name]()