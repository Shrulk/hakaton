import cv2
import numpy as np
import sys
import matplotlib.pyplot as plt


image = cv2.imread('frames/frame_1875.png')[:, :, 0]
plt.imshow(image, cmap='gray')
plt.show()


lower_threshold = 145
high_threshold = 150
L2gradients = True
edges = cv2.Canny(image, lower_threshold, high_threshold, L2gradient=L2gradients)
print(np.where(edges == 255))

lines_list =[]
lines = cv2.HoughLinesP(
            edges,   # Input edge image
            1,   # Distance resolution in pixels
            np.pi/180,   # Angle resolution in radians
            threshold=100,   # Min number of votes for valid line
            minLineLength=60,   # Min allowed length of line
            maxLineGap=10   # Max allowed gap between line for joining them
            )
for points in lines:
    # Extracted points nested in the list
    x1, y1, x2, y2 = points[0]
    # Draw the lines joing the points
    # On the original image
    plt.plot([x1, x2], [y1, y2], color='red')
    # Maintain a simples lookup list for points
    lines_list.append([(x1, y1), (x2, y2)])

plt.imshow(image, cmap='gray')
# plt.scatter(np.where(edges == 255)[1], np.where(edges == 255)[0], s=0.1, color='red')
plt.show()
