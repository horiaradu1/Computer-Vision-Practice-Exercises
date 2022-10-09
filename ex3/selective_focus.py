import numpy as np
import cv2
import sys
from matplotlib import pyplot as plt

# Calibration data
cam0=np.array([[5806.559, 0, 1429.219], [0, 5806.559, 993.403], [0, 0, 1]])
cam1=np.array([[5806.559, 0, 1543.51 ], [0, 5806.559, 993.403], [0, 0, 1]])
doffs=114.291
baseline=174.019
width=2960
height=2016
focal_length = 5806.559

# Camera Focal Length Parameters
width_sensor_size = 22.2
height_sensor_size = 14.8
width_camera_resolution = 3088
height_camera_resolution = 2056

# Parameters
global img, imgL, imgR
imgL = []
imgR = []
global num_disparities, block_size, img_type, th1, th2, aperture
num_disparities = 0
block_size = 0
img_type = 1
th1 = 50
th2 = 150
aperture = 0
global num_disparities_val, block_size_val
num_disparities_val = 0
block_size_val = 5
global disparity, depth
disparity = None
depth = None
global k, depth_threshold
k = 0
depth_threshold = 0
# ================================================

def getDisparityMap(imL, imR, numDisparities, blockSize):
    stereo = cv2.StereoBM_create(numDisparities=numDisparities, blockSize=blockSize)

    disparity = stereo.compute(imL, imR)
    disparity = disparity - disparity.min() + 1 # Add 1 so we don't get a zero depth, later
    disparity = disparity.astype(np.float32) / 16.0 # Map is fixed point int with 4 fractional bits

    return disparity # floating point image

# ================================================

def plot(disparity, baseline, doffs, focal):
    # This just plots some sample points.  Change this function to
    # plot the 3D reconstruction from the disparity map and other values
    z = baseline * (focal / (disparity + doffs))
    x = np.zeros(z.shape)
    y = np.zeros(z.shape)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            x[i,j] = z[i,j] * (j / focal) - baseline / 2
            y[i,j] = - z[i,j] * (i / focal) + disparity.shape[0] / 2

    x = x.flatten()
    y = y.flatten()
    z = z.flatten()

    threshold_id = []
    for id, val in enumerate(z):
        if val > 8500:
            threshold_id.append(id)

    x = np.delete(x, threshold_id)
    y = np.delete(y, threshold_id)
    z = np.delete(z, threshold_id)

    # Plt depths
    ax = plt.axes(projection ='3d')
    ax.scatter(x, z, y, s=1)
    ax.view_init(elev=0, azim=-90)

    # Labels
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')

    # plt.savefig('myplot.pdf', bbox_inches='tight') # Can also specify an image, e.g. myplot.png
    plt.show()

# ================================================
# Trackbar functions

def print_stats_vals():
    print('------------------------------------------------------')
    print("imgtype: {}".format(img_type))
    print("th1: {}".format(th1))
    print("th2: {}".format(th2))
    print("aperture: {}".format(3 + aperture * 2))
    print("num_disparities: {}".format(num_disparities_val))
    print("block_size: {}".format(block_size_val))
    print("k: {}".format(k))
    print("depth_threshold: {}".format(depth_threshold))
    print('------------------------------------------------------')

def image_type_trackbar(val):
    global img_type
    global img
    img_type = val
    img = [imgL[val], imgR[val]]
    show_disparity()

def th1_trackbar(val):
    global th1
    global img
    th1 = val
    imgL[1] = edge_detection(imgL[0], th1, th2, 3 + aperture * 2)
    imgR[1] = edge_detection(imgR[0], th1, th2, 3 + aperture * 2)
    img = [imgL[1], imgR[1]]
    show_disparity()
    
def th2_trackbar(val):
    global th2
    global img
    th2 = val
    imgL[1] = edge_detection(imgL[0], th1, th2, 3 + aperture * 2)
    imgR[1] = edge_detection(imgR[0], th1, th2, 3 + aperture * 2)
    img = [imgL[1], imgR[1]]
    show_disparity()
    
def aperture_trackbar(val):
    global aperture
    global img
    aperture = val
    imgL[1] = edge_detection(imgL[0], th1, th2, 3 + aperture * 2)
    imgR[1] = edge_detection(imgR[0], th1, th2, 3 + aperture * 2)
    img = [imgL[1], imgR[1]]
    show_disparity()
    
def num_disparities_trackbar(val):
    global num_disparities_val
    num_disparities_val = val * 16
    show_disparity()
    
def block_size_trackbar(val):
    global block_size_val
    if val % 2 == 0:
        block_size_val = val + 5
    else:
        block_size_val = val + 4
    show_disparity()
    
def k_trackbar(val):
    global k
    k = val / 100
    show_disparity()

def depth_trackbar(val):
    global depth_threshold
    depth_threshold = val
    show_disparity()
    
# ================================================

def focal_length_px_to_mm(focal_length_px, sensor_size_mm, image_resolution):
    return focal_length_px * sensor_size_mm / image_resolution

def edge_detection(gray, th1, th2, aperture):
    # Apply Canny edge detection
    edges = cv2.Canny(gray, th1, th2, apertureSize=aperture)
    return edges

def show_disparity():
    global disparity
    global depth
    disparity = getDisparityMap(img[0], img[1], num_disparities_val, block_size_val)
    disparity = np.interp(disparity, (disparity.min(), disparity.max()), (0.0, 1.0))

    depth = 1 / (disparity + k)
    depth = np.interp(depth, (depth.min(), depth.max()), (0, 255))
    
    mask = (depth < depth_threshold)[:, :, np.newaxis]
    mask_gray = (depth < depth_threshold)
    foreground = img_left
    foreground_gray = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY)
    background = cv2.cvtColor(background, cv2.COLOR_GRAY2BGR)

    background_color = cv2.GaussianBlur(background, (15, 15), 256)
    background_gray  = cv2.GaussianBlur(cv2.cvtColor(foreground, cv2.COLOR_BGR2GRAY), (15, 15), 256)

    focused_color = ((mask * foreground) + ((1 - mask) * background_color)).astype(np.uint8)
    focused_color = np.interp(focused_color, (focused_color.min(), focused_color.max()), (0.0, 1.0))

    focused_gray  = ((mask_gray * foreground_gray) + ((1 - mask_gray) * background_gray)).astype(np.uint8)
    focused_gray  = np.interp(focused_gray, (focused_gray.min(), focused_gray.max()), (0.0, 1.0))

    print_stats_vals()
    cv2.imshow('Disparity', disparity)
    cv2.imshow('Depth', depth.astype(np.uint8))
    cv2.imshow('Focused_color', focused_color)
    cv2.imshow('Focused_gray', focused_gray)
    
# ================================================

if __name__ == '__main__':

    # Load left image
    filename = 'girlL.png'
    imgL.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    img_left = cv2.imread(filename, cv2.IMREAD_COLOR)
    if imgL[0] is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    # Load right image
    filename = 'girlR.png'
    imgR.append(cv2.imread(filename, cv2.IMREAD_GRAYSCALE))
    img_right = cv2.imread(filename, cv2.IMREAD_COLOR)
    if imgR[0] is None:
        print('\nError: failed to open {}.\n'.format(filename))
        sys.exit()

    imgL.append(edge_detection(imgL[0], th1, th2, 3 + aperture * 2))
    imgR.append(edge_detection(imgR[0], th1, th2, 3 + aperture * 2))

    cam0_x = focal_length_px_to_mm(cam0[0,0], width_sensor_size, width_camera_resolution)
    cam0_y = focal_length_px_to_mm(cam0[1,1], height_sensor_size, height_camera_resolution)

    cam1_x = focal_length_px_to_mm(cam1[0,0], width_sensor_size, width_camera_resolution)
    cam1_y = focal_length_px_to_mm(cam1[1,1], height_sensor_size, height_camera_resolution)

    print("cam0 x focal length in mm: ", cam0_x)
    print("cam0 y focal length in mm: ", cam0_y)
    print("cam1 x focal length in mm: ", cam1_x)
    print("cam1 y focal length in mm: ", cam1_y)

    ###############################################################################

    # Create a window to display the image in
    cv2.namedWindow('Disparity', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Depth', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Focused_color', cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow('Focused_gray', cv2.WINDOW_AUTOSIZE)

    cv2.createTrackbar('num_disparities', 'Disparity', num_disparities, 50, num_disparities_trackbar)
    cv2.createTrackbar('block_size', 'Disparity', block_size, 250, block_size_trackbar)
    cv2.createTrackbar('image_type', 'Disparity', img_type, 1, image_type_trackbar)
    # cv2.createTrackbar('th1', 'Disparity', th1, 255, th1_trackbar)
    # cv2.createTrackbar('th2', 'Disparity', th2, 255, th2_trackbar)
    # cv2.createTrackbar('aperture', 'Disparity', aperture, 2, aperture_trackbar)
    cv2.createTrackbar('k', 'Depth', 0, 500, k_trackbar)
    cv2.createTrackbar('depth_threshold', 'Focused_color', depth_threshold, 200, depth_trackbar)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # plot(disparity, baseline, doffs, focal_length)

    ###############################################################################

    

