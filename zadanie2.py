import time
import cv2
from ximea import xiapi
from keyboard import is_pressed
import numpy as np
import cv2 as cv
import glob

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

try:
    while True:
        cam.get_image(img)
        image = img.get_image_data_numpy()
        image = cv2.resize(image, (240, 240))
        cv2.imshow("image", image)
        cv2.waitKey(1)
        if is_pressed('space'):
            for i in range(10):
                cam.get_image(img)
                image = img.get_image_data_numpy()
                image = cv2.resize(image, (240, 240))
                cv2.imshow("image", image)
                time.sleep(3)
                images.append(image)
                cv2.imwrite(f"image_{i}.jpg", image)
            break

finally:
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    objp = np.zeros((7 * 5, 3), np.float32)
    objp[:, :2] = np.mgrid[0:7, 0:5].T.reshape(-1, 2)

    # Arrays to store object points and image points from all the images.
    objpoints = []  # 3d point in real world space
    imgpoints = []  # 2d points in image plane.

    images = glob.glob('*.jpg')
    if images == []:
        print('No images found.')

    for fname in images:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Find the chess board corners
        ret, corners = cv2.findChessboardCorners(gray, (7, 5), None)

        if ret == True:
            objpoints.append(objp)
            print("Found chessboard")
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            imgpoints.append(corners2)

            # Draw and display the corners
            imgShow = img.copy()
            cv2.drawChessboardCorners(imgShow, (7, 5), corners2, ret)
            cv2.imshow('imgShow', imgShow)
            cv2.waitKey(0)

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    print("Camera matrix:\n", mtx)

    fx = mtx[0, 0]
    fy = mtx[1, 1]
    cx = mtx[0, 2]
    cy = mtx[1, 2]
    print("fx =", fx)
    print("fy =", fy)
    print("cx =", cx)
    print("cy =", cy)

    h, w = img.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w, h), 1, (w, h))

    dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

    # crop the image
    x, y, w, h = roi
    dst = dst[y:y + h, x:x + w]
    cv2.imwrite('calibresult.png', dst)

    mapx, mapy = cv2.initUndistortRectifyMap(mtx, dist, None, newcameramtx, (w, h), 5)
    dst = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)

    cv2.imshow("original", img)
    cv2.imshow("undistorted", dst)
    cv2.waitKey(0)

    np.savez("calibration_data.npz", camera_matrix=mtx, dist_coeffs=dist)
    print("Kalibrácia uložená.")

    cam.stop_acquisition()
    cam.close_device()
    print('Camera closed.')