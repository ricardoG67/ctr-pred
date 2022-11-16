import cv2

image = cv2.imread('image.png')

print(image.shape)
print(type(image))

vertical = (image.shape)[0]
vertical = int(vertical)

horizontal = (image.shape)[1]
horizontal = int(horizontal)

# Puntos
vertical_p1 = vertical/5
vertical_p2 = (2*vertical)/5
vertical_p3 = (3*vertical)/5
vertical_p4 = (4*vertical)/5

horizontal_p1 = (horizontal)/5
horizontal_p2 = (2*horizontal)/5
horizontal_p3 = (3*horizontal)/5
horizontal_p4 = (4*horizontal)/5


cv2.line(image, (0,int(horizontal_p1)), (horizontal, int(horizontal_p1)), (0,0,255), 2)
cv2.line(image, (0,int(horizontal_p2)), (horizontal, int(horizontal_p2)), (0,0,255), 2)
cv2.line(image, (0,int(horizontal_p3)), (horizontal, int(horizontal_p3)), (0,0,255), 2)
cv2.line(image, (0,int(horizontal_p4)), (horizontal, int(horizontal_p4)), (0,0,255), 2)

cv2.line(image, (int(vertical_p1),0), (int(vertical_p1), vertical), (0,0,255), 2)
cv2.line(image, (int(vertical_p2),0), (int(vertical_p2), vertical), (0,0,255), 2)
cv2.line(image, (int(vertical_p3),0), (int(vertical_p3), vertical), (0,0,255), 2)
cv2.line(image, (int(vertical_p4),0), (int(vertical_p4), vertical), (0,0,255), 2)

cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows()

img_encode = cv2.imencode('.png', image)
print(img_encode)

img_encode = img_encode[1]

import numpy as np
data_encode = np.array(img_encode)
byte_encode = data_encode.tobytes()

print(data_encode)
