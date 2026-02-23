import cv2
from ximea import xiapi
from keyboard import is_pressed
import numpy as np
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
cam.set_exposure(50000)
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
        if is_pressed('space'):
            for i in range(4):
                cam.get_image(img)
                image = img.get_image_data_numpy()

                if image.ndim == 3 and image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

                image = cv2.resize(image, (240, 240))
                images.append(image)

            kernel = np.array([[0, -1, 0],
                               [-1, 5, -1],
                               [0, -1, 0]], dtype=np.float32)

            images[0] = cv2.filter2D(images[0], -1, kernel, borderType=cv2.BORDER_REPLICATE)

            src = images[1]
            h, w = src.shape[0], src.shape[1]
            rot = np.zeros((w, h, 3), dtype=src.dtype)

            for i in range(h):
                for j in range(w):
                    rot[j, h - 1 - i] = src[i, j]

            images[1] = rot

            red_only = images[2].copy()
            red_only[:, :, 0] = 0  # Blue
            red_only[:, :, 1] = 0  # Green
            images[2] = red_only

            mosaic = np.vstack((
                np.hstack((images[0], images[1])),
                np.hstack((images[2], images[3]))
            ))

            cv2.imshow("zadanie1", mosaic)

            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break

finally:
    cam.stop_acquisition()
    cam.close_device()
    cv2.destroyAllWindows()
    print('Camera closed.')