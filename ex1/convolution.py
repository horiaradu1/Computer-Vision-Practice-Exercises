from matplotlib import pyplot as plt
import math

def read_image_and_pad(path, width, height, padding):
    """ Read .bmp file and pad directly """
    # Calculate starting bit (multiple of 4)
    if width%4 == 0:
        width_offset = width
    else:
        width_offset = width+(4-width%4)
    # Skip offset of the .bmp file input bytestring
    file = open(path, "rb")
    file.seek(10)
    start = int.from_bytes(file.read(4), "little")
    file.seek(18)
    file.read(8)
    file.seek(start-1)

    img = []
    line = []
    index = 0
    for _ in range(padding):
        line = []
        for _ in range(width + 2*padding):
            line.append(0)
        img.insert(0, line)
    line = []
    for _ in range(padding):
        line.append(0)
    while True:
        if index == width_offset:
            index = 0
            for _ in range(padding):
                line.append(0)
            img.insert(0, line)
            line = []
            for _ in range(padding):
                line.append(0)
        index += 1
        bit_string = file.read(1)

        if len(bit_string) == 0:
            for _ in range(padding):
                line = []
                for _ in range(width + 2*padding):
                    line.append(0)
                img.insert(0, line)
            break
        if index < 4-width%4 or index >= width_offset:
            continue

        pixel = ord(bit_string)
        line.append(pixel)

    file.close()

    return img

def padd_img(img, width, height, padding):
    """ Return input image padded"""
    padded_img = []
    for _ in range(padding):
        padded_img.append([0 for _ in range(width)])
    
    padded_img.extend(img.copy())
    padded_img = [x+ [0]*padding for x in padded_img]
    padded_img = [[0]*padding +x for x in padded_img]
    for _ in range(padding):
        padded_img.append([0 for _ in range(width+2*padding)])

    return padded_img

def convolution(img, width, height, padding, kernel):
    """ Convolution function on already padded image and input kernel 
    Need to enter pad number and kernel"""
    kernel_size = len(kernel)
    kh = len(kernel)
    kw = len(kernel[0])
    h = height - kh + 2 * padding + 1
    w = width - kw + 2 * padding + 1
    img_out = [
        [0 for _ in range(w)]
        for _ in range(h)
    ]

    # Find the highest value in the matrix for normalization
    max_value = 0
    for i in range(h):
        for j in range(w):
            for k in range(kernel_size):
                for l in range(kernel_size):
                    img_out[i][j] += (img[i + k][j + l] * kernel[k][l])
            max_value = max(max_value, img_out[i][j])

    return img_out, max_value

def gradient_compute(img, kernel_name, width, height, padding):
    """ Compute the horizontal and vertical gradient images 
    and then do the gradient magnitude of them
    (combine - square root of sum of squares)
    Two kernels to choose from """
    if kernel_name == "sobel":
        # Sobel Kernels
        kernel_horizontal = list()
        kernel_horizontal.extend([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
        kernel_vertical = list()
        kernel_vertical.extend([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    elif kernel_name == "scharr":
        # Scharr Kernels
        kernel_horizontal = list()
        kernel_horizontal.extend([[-3, -10, -3], [0, 0, 0], [3, 10, 3]])
        kernel_vertical = list()
        kernel_vertical.extend([[-3, 0, 3], [-10, 0, 10], [-3, 0, 3]])
    else:
        exit()

    # Apply the convolution function to the image both horizontally and vertically
    img_horizontal, max_1 = convolution(img, width, height, padding, kernel_horizontal)
    img_vertical, max_2 = convolution(img, width, height, padding, kernel_vertical)

    # Calculate the gradient magnitude between horizontal and vertical gradient images
    gradient_magnitude = magnitude_compute(img_horizontal, img_vertical)

    for i in range(len(img_horizontal)):
        for j in range(len(img_horizontal[0])):
            img_horizontal[i][j] /= max_1

    for i in range(len(img_vertical)):
        for j in range(len(img_vertical[0])):
            img_vertical[i][j] /= max_2

    max_val = 0
    for i in range(len(gradient_magnitude)):
        for j in range(len(gradient_magnitude[0])):
            max_val = max(max_val, gradient_magnitude[i][j])

    # Normalize the image to from 0-1 to 0-255
    for i in range(len(gradient_magnitude)):
        for j in range(len(gradient_magnitude[0])):
            gradient_magnitude[i][j] = int(gradient_magnitude[i][j] / (max_val * 1/255))

    return gradient_magnitude, img_horizontal, img_vertical

def magnitude_compute(img1, img2):
    """ Combine two images """
    img_mag = [
        [0 for _ in range(len(img1[0]))]
        for _ in range(len(img1))
    ]
    
    for i in range(len(img1)):
        for j in range(len(img1[0])):
            img_mag[i][j] = math.sqrt(img1[i][j] ** 2 + img2[i][j] ** 2)
    
    return img_mag

def histogram(img, width, height):
    """ Return a histogram of a grayscale image (0-255) """
    histogram = [0 for _ in range(256)]
    for i in range(height):
        for j in range(width):
            histogram[img[i][j]]+=1
    return histogram

def threshold(img, width, height, threshold):
    """ Edge detection thresholding function """
    h = height
    w = width
    img_out = [
        [0 for j in range(w)]
        for i in range(h)
    ]

    for i in range(h):
        for j in range(w):
            if (img[i][j] >= threshold):
                img_out[i][j] = 255
            else:
                img_out[i][j] = 0

    return img_out

def gaussian_blur(kernel_size, std_dev):
    """ Return a gaussian blur kernel based on the product of two Gaussian functions """
    kernel = [
        [0 for _ in range(kernel_size)]
        for _ in range(kernel_size)
    ]
    for i in range(kernel_size):
        for j in range(kernel_size):
            kernel[i][j] = (1/(2 * math.pi * std_dev**2)) * math.e**(-(((j - kernel_size//2)**2 + (i - kernel_size//2)**2) / (2 * std_dev**2)))
    return kernel

def weightead_mean(img, width, height, padding, name, kernel_size, std_dev):
    """ Apply a blur on the image (gaussian or other) """
    if (name=="gaussian"):
        kernel = gaussian_blur(kernel_size, std_dev)
    else:
        kernel = list()
        kernel.extend([[1, 2, 1], [2, 4, 2], [1, 2, 1]])

    img, max_val = convolution(img, width, height, padding, kernel)

    for i in range(len(img)):
        for j in range(len(img[0])):
            img[i][j] /= max_val
    
    return img

# -- PARAMETERS --
path = "kitty.bmp"
width = 206
height = 231

kernel_name = "sobel"
kernel_blur = "gaussian"

gaussian_kernel_size = 9
gaussian_std_dev = 10


padding_blur = gaussian_kernel_size // 2
padding_gradient = 1

thresholds_original = [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
#thresholds =          [10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65]
thresholds =          [15, 18, 20, 21, 23, 24, 25, 27, 29, 30, 32, 34]

# -- READ IMAGE AND MAKE PADDING --
img_original = read_image_and_pad(path, width, height, padding_blur)

# -- WEIGHTED MEAN FIRST ON ORIGINAL IMAGE --
#img = padd_img(img, width, height, padding)
img_blur = weightead_mean(img_original, width, height, padding_blur, kernel_blur, gaussian_kernel_size, gaussian_std_dev)
img_blur = padd_img(img_blur, width, height, padding_gradient)

# -- CALCULATE GRADIENT IMAGES --
gradient_magnitude, img_horizontal, img_vertical = gradient_compute(img_blur, kernel_name, width, height, padding_gradient)
gradient_magnitude_original, img_horizontal_original, img_vertical_original = gradient_compute(img_original, kernel_name, width, height, padding_gradient)
#gradient_magnitude = gradient_compute(img, "scharr", True)
#img_original = gradient_magnitude.copy()

# -- PERFORM EDGE DETECTION BASED ON SET THRESHOLD --
imgs_final_original = list()
for i in thresholds_original:
    imgs_final_original.append(threshold(gradient_magnitude_original, width, height, threshold=i))
#thresholds_original.reverse()

imgs_final = list()
for i in thresholds:
    imgs_final.append(threshold(gradient_magnitude, width, height, threshold=i))
#thresholds.reverse()

# -- PLOT ALL THE DATA --

fig, ax = plt.subplots(1, 2, figsize=(16, 9))
ax[0].imshow(img_original, cmap="gray")
ax[0].set_title("ORIGINAL + (padding)")
ax[1].imshow(img_blur, cmap="gray")
ax[1].set_title("BLURRED + (padding)")
plt.show()

fig, ax = plt.subplots(2, 3, figsize=(16, 9))
ax[0, 0].imshow(img_horizontal_original, cmap="gray")
ax[0, 1].imshow(img_vertical_original, cmap="gray")
ax[0, 2].imshow(gradient_magnitude_original, cmap="gray")
ax[0, 0].set_title("ORIGINAL HORIZONTAL")
ax[0, 1].set_title("ORIGINAL VERTICAL")
ax[0, 2].set_title("ORIGINAL COMBINED")
ax[1, 0].imshow(img_horizontal, cmap="gray")
ax[1, 1].imshow(img_vertical, cmap="gray")
ax[1, 2].imshow(gradient_magnitude, cmap="gray")
ax[1, 0].set_title("BLURRED HORIZONTAL")
ax[1, 1].set_title("BLURRED VERTICAL")
ax[1, 2].set_title("BLURRED COMBINED")
plt.show()

# -- PLOT THE HISTOGRAM --
bar_height = [x for x in range(256)]
histogram_original = histogram(gradient_magnitude_original, width, height)
histogram = histogram(gradient_magnitude, width, height)
fig, ax = plt.subplots(1, 1, figsize=(16, 9))
ax.bar(bar_height, histogram, color='r', alpha=0.8)
ax.bar(bar_height, histogram_original, color='b', alpha=0.6)
plt.show()

x = 0
for edges in [imgs_final_original, imgs_final]:
    fig, ax = plt.subplots(4, 6, figsize=(16, 9))
    if x == 0:
        fig.suptitle("ORIGINAL")
    else:
        fig.suptitle("BLURRED")
    for i in range(4):
        for j in range(6):
            if(i+j)%2 == 1:
                ax[i, j].set_visible(False)
                continue
            if x == 0:
                ax[i, j].set_title("Threshold: " + str(thresholds_original.pop()))
            else:
                ax[i, j].set_title("Threshold: " + str(thresholds.pop()))
            ax[i, j].imshow(edges.pop(), cmap="gray")
    plt.show()
    x+=1
