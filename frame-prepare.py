import cv2

capture = cv2.VideoCapture('CNTRL.mp4')
frame_shift = 25
frameNr = 0

def get_frame_name(n):
    fr_num = str(n).zfill(4)
    return f'frames/frame_{fr_num}.png'


while (True):
    capture.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
    success, frame = capture.read()
    if success:
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cv2.imwrite(get_frame_name(frameNr), gray_frame)
 
    else:
        break
    # frame += frame_shift
    frameNr = frameNr + frame_shift
 
capture.release()