# -*- coding: utf-8 -*-
"""
Created on Mon Jun  4 16:19:50 2018

@author: Jordan
"""
#import pickle
import face_recognition
import scipy , cv2, os, time, ctypes
from PIL import Image
import numpy as np

def idbreakout(target,header):
    # Load the jpg file into a numpy array
    image = face_recognition.load_image_file(header + target)

    # Find all the faces in the image using the default HOG-based model.
    # This method is fairly accurate, but not as accurate as the CNN model and not GPU accelerated.

    face_locations = face_recognition.face_locations(image)

    #print('I found {} face(s) in this photograph.'.format(len(face_locations)))   
    global idpic

    for face_location in face_locations:

    # Print the location of each face in this image
        top, right, bottom, left = face_location
        #print('A face is located at pixel location Top: {}, Left: {}, Bottom: {}, Right: {}'.format(top, left, bottom, right))

    # You can access the actual face itself like this:
        face_image = image[top-200:bottom+100, left-100:right+100]    
        pil_image = Image.fromarray(face_image)
        pil_image.save( header +"face" + target)
        #pil_image.show()
        
    idpic = (header +"face" + target)
    return idpic
    
#idbreakout('JordanID.jpg','Test/')        

def Screenshot(video,folder):
    print('runnning...')
    cap = cv2.VideoCapture(video)
    makemydir(folder)
    
    global count
    count = 0   
    while True:
        rect, frame = cap.read()
        time.sleep(.5)
       # width = int(frame.shape[1] * .75)
        #height = int(frame.shape[0] * .75)
        #dim = (width, height)
        #resize = cv2.resize(frame, dim) 
        
        cv2.imwrite('{0}/Screenshot{1}.png' .format(folder,count),frame)
        count = count +1
        
        if count > 3: 
            #ctypes.windll.user32.MessageBoxW(0, '{} screenshots has been taken' .format(count), "Screenshot", 0)
            idbreakout('JordanID.jpg',folder+ '/')
            face_distancecompiler(folder,count,idpic)
            break
        #hit ESC button to escape method
        if cv2.waitKey(1) & 0xFF == 27:
            break

Screenshot('newtest.mp4', 'Jordan')


def face_distancecompiler(folder,count,ID):
    video_encodings = []
    distancelist = []
    
    for i in range(count):
        try:
            globals()['loading{}' .format(i)] = face_recognition.load_image_file('{0}/Screenshot{1}.png' .format(folder,i))        
            globals()['encoding{}' .format(i)] = face_recognition.face_encodings(globals()['loading{}' .format(i)])[0]  
        except IndexError:
            rotate('{0}/Screenshot{1}.png' .format(folder,i))
            globals()['loading{}' .format(i)] = face_recognition.load_image_file('{0}/Screenshot{1}.png' .format(folder,i))        
            globals()['encoding{}' .format(i)] = face_recognition.face_encodings(globals()['loading{}' .format(i)])[0]
            print('error caught and saved rotated')
        video_encodings.append(globals()['encoding{}' .format(i)])
       
    try: 
        image_to_test = face_recognition.load_image_file(ID)
        image_to_test_encoding = face_recognition.face_encodings(image_to_test,num_jitters=00)[0]
    except IndexError:
        resize(ID)
        image_to_test = face_recognition.load_image_file(str(1080) +ID)
        image_to_test_encoding = face_recognition.face_encodings(image_to_test,num_jitters=00)[0]
        print('error caught and saved derez')
    
    face_distances = face_recognition.face_distance(video_encodings, image_to_test_encoding)      
    for i, face_distance in enumerate(face_distances):      
        distancelist.append(face_distance) 
    mean = np.mean(distancelist)
    if mean <= .5:
        print('Percieved idenity congruency with a mean of:' + str(mean))
    else:
        print('No percieved idenity congruency with a mean of:' + str(mean))

##################HELPER FUNCTIONS

def makemydir(folder):
  try:
    os.makedirs(folder)
  except OSError:
    pass

def resize(image):
    name = image
    size = (1080,1440)
    resx, resy = size
    im = cv2.imread(image)
    resized = cv2.resize(im,size)
    cv2.imwrite(  str(resx) + name,resized)

def rotate(image):
    tip = Image.open( image)
    x, y = tip.size
    tip60 = tip.rotate(270)
    tip60.save( image)

###################################################################################
def encodeme(encode,test):
    picme = face_recognition.load_image_file(encode)
    known_encoding = face_recognition.face_encodings(picme)[0]


    # my_face_encoding now contains a universal 'encoding' of my facial features that can be compared to any other picture of a face!

    unknown = face_recognition.load_image_file(test)
    
    unknown_face_encoding = face_recognition.face_encodings(unknown)[0]

    known_encodings = [
            known_encoding]
    # Now we can see the two face encodings are of the same person with `compare_faces`!

    results = face_recognition.compare_faces([known_encoding], unknown_face_encoding, tolerance=0.5)
    face_distances = face_recognition.face_distance(known_encodings, unknown_face_encoding)
    
    
    im = Image.open(encode)
    im2 = Image.open(test)
    
    im.show()
    im2.show()

    if results[0] == True:
        print('True, euclidean distance of faces is ',face_distances[0])
    else:
        print('False, euclidean distance of faces is ',face_distances[0])
#encodeme('Mannon.jpg','truetest.png')
  ###############################################################################################      
