import sys
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
import scipy.spatial
import cv2

def HarrisStrengthFunction(img):
    """ Compute Harris corner strength function """
    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3, borderType=cv2.BORDER_REFLECT)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3, borderType=cv2.BORDER_REFLECT)
    xy = x * y

    x = cv2.GaussianBlur(x, (5,5), 0.5, borderType=cv2.BORDER_REFLECT)
    y = cv2.GaussianBlur(y, (5,5), 0.5, borderType=cv2.BORDER_REFLECT)
    xy = cv2.GaussianBlur(xy, (5,5), 0.5, borderType=cv2.BORDER_REFLECT)
    xx = x**2
    yy = y**2

    detM = (xx * yy) - (xy ** 2)
    traceM = xx + yy
    R = detM - 0.05 * (traceM ** 2)
    
    orientation =np.arctan2(y, x)
    orientation = orientation * 180 / np.pi

    return R, orientation

def HarrisPointsDetector(img, blur_img=False, th=0):
    """ Detect Harris points in image """
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    if blur_img:
        img_gray = cv2.GaussianBlur(img_gray, (7,7), 7, borderType=cv2.BORDER_REFLECT)

    R, orientation = HarrisStrengthFunction(img_gray)

    max_filter = scipy.ndimage.maximum_filter(R, size=7)
    local_maxima = (R >= th) & (R == max_filter)
    output = local_maxima * 255

    kp = []
    for j in range(output.shape[0]):
        for i in range(output.shape[1]):
            if output[j, i] > 0:
                kp.append(cv2.KeyPoint(i, j, 60, orientation[j, i]))

    return kp, R

def featureDescriptor(img, kp):
    """ Compute feature descriptors for keypoints """
    orb = cv2.ORB_create()
    kp, des = orb.compute(img, kp)
    return des, kp

def orbDetect(img, method='harris'):
    """ Detect keypoints in image and get descriptors for them """
    if method == 'harris':
        orb = cv2.ORB_create(scoreType=cv2.ORB_HARRIS_SCORE)
    elif method == 'fast':
        orb = cv2.ORB_create(scoreType=cv2.ORB_FAST_SCORE)
    else:
        return None
    kp = orb.detect(img, None)
    kp, des = orb.compute(img, kp)
    return kp, des

def SSDFeatureMatcher(des1, des2):
    """ Match features using SSD distance """
    dist = scipy.spatial.distance.cdist(des1, des2, 'euclidean')
    matches = []

    for i in range(len(des1)):
        m = np.argmin(dist[i])
        matches.append(cv2.DMatch(i, m, dist[i, m]))

    return matches

def RatioFeatureMatcher(des1, des2, ratio_threshold):
    """ Match features using ratio test """
    dist = scipy.spatial.distance.cdist(des1, des2, 'euclidean')
    matches = []

    for i in range(len(des1)):
        ordered = np.argsort(dist[i])
        first, second = (dist[i][ordered[0]], dist[i][ordered[1]])

        ratio = float(first) / float(second)
        if ratio < ratio_threshold:
            matches.append(cv2.DMatch(i, ordered[0], dist[i, ordered[0]]))
    return matches

def getAllFileNames(path):
    """ Get all file names in a directory """
    import os
    files = []
    for f in os.listdir(path):
        if os.path.isfile(os.path.join(path, f)):
            files.append(f)
    return files

## Parameters
threshold = 2e4
do_blur = True
# threshold = 9e6
# do_blur = False
ratio_threshold = 0.85

## Get images
image = cv2.imread("bernieSanders.jpg", cv2.IMREAD_COLOR)
image_copy = image.copy()
image_harris = image.copy()
image_fast = image.copy()

## Detect Harris points
kp, R = HarrisPointsDetector(image, do_blur, threshold)
des, kp = featureDescriptor(image, kp)

## Plot various threshold values with keypoint numbers
# threshold_values = [-10000, -5000, -1000, -500, -400, -300, -200, -100]
# # threshold_values = []
# threshold_values.extend([x*0.1 for x in range(-500, 501)])
# threshold_values.extend([60, 70, 80, 90, 100, 200, 300, 400, 500, 1000, 2500, 5000, 7500, 10000])
# # threshold_values.extend([1e3, 2e3, 3e3, 4e3, 5e3, 6e3, 7e3, 8e3, 9e3, 1e4, 2e4, 3e4, 4e4, 5e4, 6e4, 7e4, 8e4, 9e4, 1e5, 2e5, 3e5, 4e5, 5e5, 6e5, 7e5, 8e5, 9e5, 1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6, 9e6, 1e7])
# axis_x = []
# axis_y = []
# for t in threshold_values:
#     axis_x.append(t)
#     axis_y.append((R >= t).sum())

# plt.plot(axis_x, axis_y, color='b', alpha=1)
# plt.xlabel("Threshold values")
# plt.ylabel("Keypoints number")
# plt.show()

## Detect Harris points using ORB built-in functions
kp_harris, des_harris = orbDetect(image, 'harris')
kp_fast, des_fast = orbDetect(image, 'fast')

## Draw keypoints on image
cv2.drawKeypoints(image_copy, kp, image_copy, color=(0, 0, 255), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(image_harris, kp_harris, image_harris, color=(0, 0, 255), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
cv2.drawKeypoints(image_fast, kp_fast, image_fast, color=(0, 0, 255), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)

## Plot images with keypoints
cv2.cvtColor(image, cv2.COLOR_BGR2RGB, image)
cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB, image_copy)
cv2.cvtColor(image_harris, cv2.COLOR_BGR2RGB, image_harris)
cv2.cvtColor(image_fast, cv2.COLOR_BGR2RGB, image_fast)
fig, ax = plt.subplots(1, 2, figsize=(20, 16))
ax[0].imshow(image_copy)
ax[0].set_title("OWN IMPLEMENTED HARRIS")
ax[1].imshow(image_harris)
ax[1].set_title("ORB HARRIS")
plt.savefig('IMPLEMENTED|HARRIS.png', dpi=100)
#plt.show()
fig, ax = plt.subplots(1, 2, figsize=(20, 16))
ax[0].imshow(image_copy)
ax[0].set_title("OWN IMPLEMENTED HARRIS")
ax[1].imshow(image_fast)
ax[1].set_title("ORB FAST")
plt.savefig('IMPLEMENTED|FAST.png', dpi=100)
#plt.show()
cv2.cvtColor(image_copy, cv2.COLOR_RGB2BGR, image_copy)
cv2.cvtColor(image_harris, cv2.COLOR_RGB2BGR, image_harris)
cv2.cvtColor(image_fast, cv2.COLOR_RGB2BGR, image_fast)

## Repeat above steps for each image plus do matching with original image
files = getAllFileNames('images')
for f in files:
    try:
        img = cv2.imread("images/" + f, cv2.IMREAD_COLOR)
        kp_img, _ = HarrisPointsDetector(img, do_blur, threshold)
        des_img, kp_img = featureDescriptor(img, kp_img)
        kp_img1, des_img1 = orbDetect(img, 'harris')
        kp_img2, des_img2 = orbDetect(img, 'fast')

        matches = SSDFeatureMatcher(des, des_img)
        ratios = RatioFeatureMatcher(des, des_img, ratio_threshold)
        matches_harris = SSDFeatureMatcher(des_harris, des_img1)
        ratio_harris = RatioFeatureMatcher(des_harris, des_img1, ratio_threshold)
        matches_fast = SSDFeatureMatcher(des_fast, des_img2)
        ratio_fast = RatioFeatureMatcher(des_fast, des_img2, ratio_threshold)

        img_copy = cv2.drawMatches(image_copy, kp, img, kp_img, matches, None)
        img_harris = cv2.drawMatches(image_harris, kp_harris, img, kp_img1, matches_harris, None)
        img_fast = cv2.drawMatches(image_fast, kp_fast, img, kp_img2, matches_fast, None)

        img_copy_ratio = cv2.drawMatches(image_copy, kp, img, kp_img, ratios, None)
        img_harris_ratio = cv2.drawMatches(image_harris, kp_harris, img, kp_img1, ratio_harris, None)
        img_fast_ratio = cv2.drawMatches(image_fast, kp_fast, img, kp_img2, ratio_fast, None)

        cv2.drawKeypoints(image_copy, kp, image_copy, color=(0, 0, 255), flags=cv2.DrawMatchesFlags_DRAW_RICH_KEYPOINTS)
        cv2.cvtColor(img_copy, cv2.COLOR_BGR2RGB, img_copy)
        cv2.cvtColor(img_copy_ratio, cv2.COLOR_BGR2RGB, img_copy_ratio)

        cv2.cvtColor(img_harris, cv2.COLOR_BGR2RGB, img_harris)
        cv2.cvtColor(img_harris_ratio, cv2.COLOR_BGR2RGB, img_harris_ratio)
        cv2.cvtColor(img_fast, cv2.COLOR_BGR2RGB, img_fast)
        cv2.cvtColor(img_fast_ratio, cv2.COLOR_BGR2RGB, img_fast_ratio)
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        fig, ax = plt.subplots(1, 2, figsize=(25, 16))
        # ax.imshow(img_copy)
        # ax.set_title("SSD OWN IMPLEMENTED HARRIS " + f)
        ax[0].imshow(img_harris)
        ax[0].set_title("SSD ORB HARRIS " + f)
        ax[1].imshow(img_fast)
        ax[1].set_title("SSD ORB FAST " + f)
        plt.savefig('matches/' + f + '_SSD.png', dpi=100)
        plt.close()
        #plt.show()

        fig, ax = plt.subplots(1, 2, figsize=(25, 16))
        # ax.imshow(img_copy_ratio)
        # ax.set_title("RATIO OWN IMPLEMENTED HARRIS " + f)
        ax[0].imshow(img_harris_ratio)
        ax[0].set_title("RATIO ORB HARRIS " + f)
        ax[1].imshow(img_fast_ratio)
        ax[1].set_title("RATIO ORB FAST " + f)
        plt.savefig('matches/' + f + '_RATIO.png', dpi=100)
        plt.close()
        #plt.show()
    except ValueError:
        ## bernieMoreblurred.jpg image not readable due to a ValueError, because no keypoints were found
        print(sys.exc_info()[0], " at file ", f)