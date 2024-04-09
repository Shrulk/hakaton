
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


from utils import pred_lines

def detect_lines(image):
    image_t = image[:, :, 0]
    lsd = cv2.createLineSegmentDetector()
    lines = lsd.detect(image_t)[0]
    
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # cv2.line(image, (x1, y1), (x2, y2), [0, 255, 0], 2)
        plt.plot([x1, x2], [y1, y2], color='red')
    
    return image

# img_input = cv2.imread('frames/frame_0325.png')
img_input = cv2.imread('example3.jpg')


# Blur
# img_input = cv2.GaussianBlur(img_input, (3,3), 5)

################################################### Method 1 ###################################################
# img_output = detect_lines(img_input)
# plt.imshow(img_output, cmap='gray')
# plt.show()


################################################### Method 2 ###################################################
# Load tflite model
interpreter = tf.lite.Interpreter(model_path="mlsd/tflite_models/M-LSD_512_large_fp32.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

lines = pred_lines(img_input, interpreter, input_details, output_details, input_shape=[512, 512], score_thr=0.1, dist_thr=5.0, dist_max=100.0)
img_output = img_input.copy()

# draw lines
for line in lines:
    x_start, y_start, x_end, y_end = [int(val) for val in line]
#   cv2.line(img_output, (x_start, y_start), (x_end, y_end), [0,0,255], 1)
    plt.plot([x_start, x_end], [y_start, y_end], color='cyan')


plt.imshow(img_output, cmap='gray')
plt.show()
# cv2.imshow('test', cv2.resize(img_output, (1600,900)))
# cv2.waitKey(0)
# cv2.destroyAllWindows()