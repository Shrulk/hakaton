import cv2

capture = cv2.VideoCapture('CNTRL.mp4')
frame_shift = 10
frameNr = 25 * 6    # Пропускаем первые 6 секунд видео (там стропы ещё не натянуты)

def get_frame_name(n):
    fr_num = str(n).zfill(4)
    # return f'frames/frame_{fr_num}.png'
    return f'frames_to_be_marked/frame_{fr_num}.png'

# check work dir
# import os
# print(os.path.abspath(os.curdir))

while (True):
    # Пропускаем кадры от 1:40 до 2:36 (от момента отцепления первого груза до захвата второго)
    # И пропускаем кадры после 4:41, т.к. там работы уже закончены
    if (frameNr > 25 * 100 and frameNr < 25 * (2*60 + 36)) or (frameNr > 25 * (4*60 +41)):
        frameNr = frameNr + frame_shift
        if frameNr > 8000: break

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