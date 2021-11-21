import argparse
import cv2

def main(args):
    camera_index:int=args.camera_index

    capture=cv2.VideoCapture(camera_index)
    face_cascade=cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
    eye_cascade=cv2.CascadeClassifier("./haarcascade_eye.xml")

    while(True):
        _,frame=capture.read()

        gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        faces=face_cascade.detectMultiScale(gray)

        for (x,y,w,h) in faces:
            face_gray=gray[y:y+h,x:x+w]
            eyes=eye_cascade.detectMultiScale(face_gray)

            if len(eyes)==0:
                continue

            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),4)

        cv2.imshow("Face Detection",frame)

        if cv2.waitKey(1)!=-1:
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--camera_index",type=int,default=0)
    args=parser.parse_args()

    main(args)
