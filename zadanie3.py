import cv2
import numpy as np
from ximea import xiapi
from numpy.lib.stride_tricks import sliding_window_view

cam = xiapi.Camera()

print('Opening first camera...')
cam.open_device()

cam.set_exposure(30000)
cam.set_param("imgdataformat", "XI_RGB32")
cam.set_param("auto_wb", 1)

print('Exposure was set to %i us' % cam.get_exposure())

img = xiapi.Image()

print('Starting data acquisition...')
cam.start_acquisition()

#parametre
resize_width = 400
gaussian_size = 5
sigma = 1.4
low_ratio = 0.08
high_ratio = 0.18

while True:

    cam.get_image(img)
    frame = img.get_image_data_numpy()

    #resize
    h0, w0 = frame.shape[:2]
    scale = resize_width / w0
    frame = cv2.resize(frame, (resize_width, int(h0 * scale)))

    #grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)

    #gaus kernel
    k = gaussian_size // 2
    ax = np.arange(-k, k + 1, dtype=np.float32)
    xx, yy = np.meshgrid(ax, ax)
    gaussian_kernel = np.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    gaussian_kernel /= np.sum(gaussian_kernel)

    #gaus filter
    padded = np.pad(gray, ((k, k), (k, k)), mode="edge")
    windows = sliding_window_view(padded, (gaussian_size, gaussian_size))
    blurred = np.tensordot(windows, gaussian_kernel, axes=((2, 3), (0, 1)))

    #sobel
    sobel_x = np.array([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=np.float32)

    sobel_y = np.array([
        [1, 2, 1],
        [0, 0, 0],
        [-1, -2, -1]
    ], dtype=np.float32)

    padded_blur = np.pad(blurred, ((1, 1), (1, 1)), mode="edge")
    windows_blur = sliding_window_view(padded_blur, (3, 3))

    gx = np.tensordot(windows_blur, sobel_x, axes=((2, 3), (0, 1)))
    gy = np.tensordot(windows_blur, sobel_y, axes=((2, 3), (0, 1)))

    #magnituda a direction
    magnitude = np.sqrt(gx**2 + gy**2)
    if magnitude.max() > 0:
        magnitude = magnitude / magnitude.max() * 255.0

    direction = np.rad2deg(np.arctan2(gy, gx))
    direction[direction < 0] += 180

    h, w = magnitude.shape

    #NMS
    nms = np.zeros_like(magnitude, dtype=np.float32)

    for i in range(1, h - 1):
        ang = direction[i]

        #cely riadok
        mag0 = magnitude[i]

        #horizontálne
        left = magnitude[i, :-2]
        right = magnitude[i, 2:]

        #diagonálne 45
        ul = magnitude[i - 1, 2:]
        dr = magnitude[i + 1, :-2]

        #vertikálne 90
        up = magnitude[i - 1, 1:-1]
        down = magnitude[i + 1, 1:-1]

        #diagonálne 135
        ur = magnitude[i - 1, :-2]
        dl = magnitude[i + 1, 2:]

        #masky
        m0 = ((ang >= 0) & (ang < 22.5)) | ((ang >= 157.5) & (ang <= 180))
        m45 = (ang >= 22.5) & (ang < 67.5)
        m90 = (ang >= 67.5) & (ang < 112.5)
        m135 = (ang >= 112.5) & (ang < 157.5)

        row = np.zeros_like(mag0)

        #horizontálne
        row[1:-1][m0[1:-1]] = np.where(
            (mag0[1:-1][m0[1:-1]] >= left[m0[1:-1]]) &
            (mag0[1:-1][m0[1:-1]] >= right[m0[1:-1]]),
            mag0[1:-1][m0[1:-1]], 0
        )

        #45
        row[1:-1][m45[1:-1]] = np.where(
            (mag0[1:-1][m45[1:-1]] >= ul[m45[1:-1]]) &
            (mag0[1:-1][m45[1:-1]] >= dr[m45[1:-1]]),
            mag0[1:-1][m45[1:-1]], 0
        )

        #90
        row[1:-1][m90[1:-1]] = np.where(
            (mag0[1:-1][m90[1:-1]] >= up[m90[1:-1]]) &
            (mag0[1:-1][m90[1:-1]] >= down[m90[1:-1]]),
            mag0[1:-1][m90[1:-1]], 0
        )

        #135
        row[1:-1][m135[1:-1]] = np.where(
            (mag0[1:-1][m135[1:-1]] >= ur[m135[1:-1]]) &
            (mag0[1:-1][m135[1:-1]] >= dl[m135[1:-1]]),
            mag0[1:-1][m135[1:-1]], 0
        )

        nms[i] = row

    #treshold
    high_threshold = nms.max() * high_ratio
    low_threshold = high_threshold * low_ratio

    strong = 255
    weak = 75

    thresholded = np.zeros((h, w), dtype=np.uint8)
    strong_mask = nms >= high_threshold
    weak_mask = (nms >= low_threshold) & (nms < high_threshold)

    thresholded[strong_mask] = strong
    thresholded[weak_mask] = weak

    #hystereza
    edges = thresholded.copy()

    strong_mask = edges == strong
    weak_mask = edges == weak

    changed = True
    while changed:
        padded_strong = np.pad(strong_mask, ((1, 1), (1, 1)), mode='constant')

        neighbors = (
            padded_strong[:-2, :-2] | padded_strong[:-2, 1:-1] | padded_strong[:-2, 2:] |
            padded_strong[1:-1, :-2] | padded_strong[1:-1, 1:-1] | padded_strong[1:-1, 2:] |
            padded_strong[2:, :-2] | padded_strong[2:, 1:-1] | padded_strong[2:, 2:]
        )

        new_strong = weak_mask & neighbors

        if np.any(new_strong):
            strong_mask |= new_strong
            weak_mask &= ~new_strong
        else:
            changed = False

    edges[:, :] = 0
    edges[strong_mask] = 255

    #zobrazenie
    cv2.imshow("Original", frame)
    cv2.imshow("Edges", edges.astype(np.uint8))

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        cam.stop_acquisition()
        cam.close_device()
        print('Camera closed.')
        break