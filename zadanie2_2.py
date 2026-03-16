import math
import cv2
from ximea import xiapi
import numpy as np
import cv2 as cv

def nothing(x):
    pass
### runn this command first echo 0|sudo tee /sys/module/usbcore/parameters/usbfs_memory_mb  ###

# create instance for first connected camera
cam = xiapi.Camera()

# start communication
# to open specific device, use:
# cam.open_device_by_SN('41305651')
# (open by serial number
print('Opening first camera...')
cam.open_device()

# settings
cam.set_exposure(30000)
cam.set_param("imgdataformat","XI_RGB32")
cam.set_param("auto_wb",1)

print('Exposure was set to %i us' %cam.get_exposure())

# create instance of Image to store image data and metadata
img = xiapi.Image()

# start data acquisitionq
print('Starting data acquisition...')
cam.start_acquisition()
images = []

cv2.namedWindow('Source')
cv2.createTrackbar('R', 'Source', 0, 255, nothing)
cv2.createTrackbar('G', 'Source', 0, 255, nothing)
cv2.createTrackbar('B', 'Source', 0, 255, nothing)


while True:
    cam.get_image(img)
    image = img.get_image_data_numpy()
    image = cv2.resize(image, (288, 240))

    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 7)
    rows = gray.shape[0]

    dst = cv2.Canny(gray, 50, 200)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)

    circles = cv2.HoughCircles(dst, cv2.HOUGH_GRADIENT, 1, rows / 4,
                              param1=100, param2=30,
                              minRadius=20, maxRadius=150)
    if circles is not None:
        circles = np.uint16(np.around(circles))
        for i in circles[0, :]:
            center = (i[0], i[1])
            # circle center
            cv2.circle(cdst, center, 1, (0, 100, 100), 3)
            # circle outline
            radius = i[2]
            cv2.circle(cdst, center, radius, (255, 0, 255), 3)
            print("Circle")

    ret, thresh1 = cv2.threshold(gray, 160, 255, 1)
    thresh2 = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,11,2)
    thresh3 = cv2.adaptiveThreshold(gray,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

    combined = cv2.bitwise_and(thresh1, thresh2)
    combined = cv2.bitwise_and(combined, thresh3)
    contours, h = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 250 or area > 20000:
            continue

        perimeter = cv2.arcLength(cnt, True)
        if perimeter < 40:
            continue

        x, y, w, h = cv2.boundingRect(cnt)
        if w < 12 or h < 12:
            continue

        approx = cv2.approxPolyDP(cnt, 0.01 * perimeter, True)
        if len(approx) == 5:
            print("Pentagon")
            cv2.drawContours(image, [cnt], 0, 255, 2)
            M = cv2.moments(cnt)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)

        elif len(approx) == 3:
            print("Triangle")
            cv2.drawContours(image, [cnt], 0, (0, 255, 0), 2)
            M = cv2.moments(cnt)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(image, (cx, cy), 3, (0, 0, 255), -1)

        elif len(approx) == 4:
            ratio = float(w) / h
            if 0.9 <= ratio <= 1.1:
                print("Square")
                cv2.drawContours(image, [cnt], 0, (10, 100, 255), 2)

            else:
                print("Rectangle")
                cv2.drawContours(image, [cnt], 0, (0, 0, 255), 2)

            M = cv2.moments(cnt)

            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])
                cv2.circle(image, (cx, cy), 3, (0, 0, 0), -1)

      # get current positions of four trackbars
    r = cv2.getTrackbarPos('R', 'Source')
    g = cv2.getTrackbarPos('G', 'Source')
    b = cv2.getTrackbarPos('B', 'Source')

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower1 = np.array([0, 120, 70])
    upper1 = np.array([10, 255, 255])

    lower2 = np.array([170, 120, 70])
    upper2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower1, upper1)
    mask2 = cv2.inRange(hsv, lower2, upper2)

    mask = cv2.bitwise_or(mask1, mask2)

    image[mask != 0] = [b, g, r]

    cv2.imshow("Source", image)
    cv2.imshow("Detected Lines (in red) - Standard Hough Line Transform", cdst)
    cv2.waitKey(1)

    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        cam.stop_acquisition()
        cam.close_device()
        print('Camera closed.')
        break