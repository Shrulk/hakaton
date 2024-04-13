import cv2

capture = cv2.VideoCapture('CNTRL.mp4')
frame_shift = 10
frameNr = 0    # Пропускаем первые 6 секунд видео (там стропы ещё не натянуты)

def get_frame_name(n):
    fr_num = str(n).zfill(4)
    # return f'frames/frame_{fr_num}.png'
    return f'D:/Education/10_sem/ml/hakaton/marked_frames/colored_frames/frame_{fr_num}.png'

# check work dir
# import os
# print(os.path.abspath(os.curdir))
import os

files = os.listdir("marked_frames")
nums = []
for file in files:
    if file in ["xml_s", "colored_frames"]: continue
    nums.append(int(file[-9:-5]))

# while (True):
    # Пропускаем кадры от 1:40 до 2:36 (от момента отцепления первого груза до захвата второго)
    # И пропускаем кадры после 4:41, т.к. там работы уже закончены
    # if (frameNr > 25 * 100 and frameNr < 25 * (2*60 + 36)) or (frameNr > 25 * (4*60 +41)):
    #     frameNr = frameNr + frame_shift
    #     continue
    # if frameNr > 8000: break

for frameNr in nums:
    capture.set(cv2.CAP_PROP_POS_FRAMES, frameNr)
    success, frame = capture.read()
    if success:
        # gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imwrite(get_frame_name(frameNr), frame)

    # else:
    #     break
    # # frame += frame_shift
    # frameNr = frameNr + frame_shift
 
capture.release()