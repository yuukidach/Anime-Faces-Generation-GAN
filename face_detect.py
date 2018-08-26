import cv2
import sys
import os.path
from glob2 import glob
'''
def detect(filename, cascade_file = "./lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    
    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor = 1.1,
                                     minNeighbors = 5,
                                     minSize = (24, 24))
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x-(w>>1), y-(h>>1)), (x + w+(w>>1), y + h+(h>>1)), (0, 0, 255), 2)

    cv2.imshow("AnimeFaceDetect", image)
    cv2.waitKey(0)

if len(sys.argv) != 2:
    sys.stderr.write("usage: python3 face_detect.py <filename>\n")
    sys.exit(-1)
    
detect(sys.argv[1])
'''

def detect(filename, cascade_file="lbpcascade_animeface.xml"):
    if not os.path.isfile(cascade_file):
        raise RuntimeError("%s: not found" % cascade_file)

    cascade = cv2.CascadeClassifier(cascade_file)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    faces = cascade.detectMultiScale(gray,
                                     # detector options
                                     scaleFactor=1.1,
                                     minNeighbors=5,
                                     minSize=(48, 48))
    for i, (x, y, w, h) in enumerate(faces):
        face = image[y-(h>>2): y + h+(h>>2), x-(w>>2):x + w+(w>>2),:]
        face = cv2.resize(face, (128, 128))
        save_filename = '%s-%d.jpg' % (os.path.basename(filename).split('.')[0], i)
        cv2.imwrite("getchu_samples/" + save_filename, face)


if __name__ == '__main__':
    if os.path.exists('getchu_samples') is False:
        os.makedirs('getchu_samples')
    try:
        file_list = next(os.walk("./anime_faces1"))[2]
    except IOError:
        print('文件列表为空')
    
    for filename in file_list:
        try:
            detect(os.path.join("./anime_faces1",filename))
        except:
            continue