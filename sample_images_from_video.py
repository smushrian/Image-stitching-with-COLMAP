import cv2

# Opens the Video file
cap= cv2.VideoCapture('C:/Users/hurr/Documents/SKOLARBETE/Design_project_in_systems_control_and_mechatronics/videos/my_video-1.mkv')

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i % 500 == 0: # this is the line I added to make it only save one frame every 20
        cv2.imwrite('video1frame'+str(i)+'.jpg',frame)
    i+=1

cap.release()

cap= cv2.VideoCapture('C:/Users/hurr/Documents/SKOLARBETE/Design_project_in_systems_control_and_mechatronics/videos/my_video-2.mkv')

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i % 500 == 0: # this is the line I added to make it only save one frame every 20
        cv2.imwrite('video2frame'+str(i)+'.jpg',frame)
    i+=1

cap.release()

cap= cv2.VideoCapture('C:/Users/hurr/Documents/SKOLARBETE/Design_project_in_systems_control_and_mechatronics/videos/my_video-3.mkv')

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i % 500 == 0: # this is the line I added to make it only save one frame every 20
        cv2.imwrite('video3frame'+str(i)+'.jpg',frame)
    i+=1

cap.release()

cap= cv2.VideoCapture('C:/Users/hurr/Documents/SKOLARBETE/Design_project_in_systems_control_and_mechatronics/videos/my_video-4.mkv')

i=0
while(cap.isOpened()):
    ret, frame = cap.read()
    if ret == False:
        break
    if i % 500 == 0: # this is the line I added to make it only save one frame every 20
        cv2.imwrite('video4frame'+str(i)+'.jpg',frame)
    i+=1

cap.release()
cv2.destroyAllWindows()