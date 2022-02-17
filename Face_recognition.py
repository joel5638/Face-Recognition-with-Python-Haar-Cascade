import os
import glob
import cv2
import sys
import argparse
import uuid

images = "Database/"

im = sorted(glob.glob(images + "*.*"), key=os.path.getmtime)

ap = argparse.ArgumentParser()
ap.add_argument("-i", "--input", required=True)
args = vars(ap.parse_args())

input_image = args["input"]

def find_hist(image):
	hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8],[0, 256, 0, 256, 0, 256])
	hist = cv2.normalize(hist, hist).flatten()
	return hist

def face_detect(image):
  image  = cv2.imread(image)
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
  faces = faceCascade.detectMultiScale(gray,scaleFactor=1.3,minNeighbors=3,minSize=(30, 30))
  height, width = image.shape[:2]
  for (x, y, w, h) in faces:
    r = max(w, h) / 2
    centerx = x + w / 2
    centery = y + h / 2
    nx = int(centerx - r)
    ny = int(centery - r)
    nr = int(r * 2)
    faceimg = image[ny:ny+nr, nx:nx+nr]
  return faceimg,faces,(x, y, w, h)

def find_match(hist1, hist2):
	match = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
	return match

inpu,_,_ = face_detect(input_image)

in_image  = cv2.imread(input_image)

input_image = find_hist(inpu)

for i in im:
  faceimg,faces,(x, y, w, h) = face_detect(i)
  label = i.split('.')[0]
  label = label.split('/')[1]
  sample = find_hist(faceimg)
  match_result = find_match(sample, input_image)
  if (match_result*100)>=75:
    print("===========match===========",(match_result*100),"label_name",label)
    image = cv2.imread(i)
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    cv2.putText(image,label,(x,x+w), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0))
    cv2.imshow("Input image",in_image)
    cv2.imshow("Output Database :- "+ label,image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
  else:
  	print("-----no match-----",(match_result*100),"label_name",label)