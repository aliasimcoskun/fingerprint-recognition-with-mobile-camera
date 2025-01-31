import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
from PIL import Image, ExifTags

# Function to load an image from a given path and convert it to grayscale
def load_image(image_path):
    image = Image.open(image_path)
    
    # Check for EXIF data and rotate if needed
    try:
        for orientation in ExifTags.TAGS.keys():
            if ExifTags.TAGS[orientation] == 'Orientation':
                break
        exif = dict(image._getexif().items())

        if exif[orientation] == 3:
            image = image.rotate(180, expand=True)
        elif exif[orientation] == 6:
            image = image.rotate(270, expand=True)
        elif exif[orientation] == 8:
            image = image.rotate(90, expand=True)
    except (AttributeError, KeyError, IndexError):
        # No EXIF data, or other error occurred
        pass

    image = image.convert('L')  # Convert image to grayscale
    image = np.array(image)
    if image is None:
        raise ValueError("Image could not be loaded. Is the file path correct?")
    return image

# Function to apply a bilateral filter to an image for noise reduction
def apply_bilateral_filter(image, d=11, sigma_color=75, sigma_space=75):
    return cv2.bilateralFilter(image, d, sigma_color, sigma_space)

# Function to apply adaptive thresholding to an image to convert it to a binary image
def apply_adaptive_threshold(image):
    blockSize = 31  # Size of the neighbourhood area
    C = 2  # Constant subtracted from the mean or weighted mean
    binary_image = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY, blockSize, C)
    return binary_image

# Function to extract the center region of an image
def extract_center(image):
    height, width = image.shape[:2]  # Get the dimensions of the image
    width_per_part = width // 3  # Calculate the width of the central part
    height_per_part = height // 3  # Calculate the height of the central part
    # Determine the coordinates of the central rectangle
    x1, y1 = width_per_part, height_per_part  # Top left corner of the center rectangle
    x2, y2 = x1 + width_per_part, y1 + height_per_part  # Bottom right corner
    center_rectangle = image[y1:y2, x1:x2]  # Extract the center rectangle

    return center_rectangle

# Function to apply morphological operations (closing and opening) to an image
def apply_morphological_operations(image, kernel_size=(3, 3), iterations=1):
    # Create a rectangular kernel
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size) 
    # Apply closing (dilation followed by erosion)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel, iterations=iterations) 
    # Apply opening (erosion followed by dilation)
    opening = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel, iterations=iterations) 

    return opening

# Sample image path
img_path = 'images/7.jpg'  
# Load and preprocess the fingerprint image
fingerprint = load_image(img_path)
blurred_fingerprint = apply_bilateral_filter(fingerprint)
binary_fingerprint = apply_adaptive_threshold(blurred_fingerprint)
centered = extract_center(binary_fingerprint)
processed_image = apply_morphological_operations(centered, kernel_size=(3, 3), iterations=1)

# Uncomment below lines to visualize the original, centered, and processed images
# plt.figure(figsize=(20, 16))
# plt.subplot(1, 3, 1)
# plt.imshow(fingerprint, cmap='gray')
# plt.title('Original Grayscale Fingerprint')
# plt.subplot(1, 3, 2)
# plt.imshow(centered, cmap='gray')
# plt.title('Center of the Fingerprint')
# plt.subplot(1, 3, 3)
# plt.imshow(processed_image, cmap='gray')
# plt.title('Morphological Operations (Closing and Opening)')
# plt.show()

# Function to match two fingerprint images and display the results
def match_fingerprints(image_path1, image_path2):
    # Load the fingerprint images
    fingerprint_image1 = load_image(image_path1)
    fingerprint_image2 = load_image(image_path2)

    # Preprocess the fingerprint images
    blurred_fingerprint1 = apply_bilateral_filter(fingerprint_image1)
    blurred_fingerprint2 = apply_bilateral_filter(fingerprint_image2)
    binary_fingerprint1 = apply_adaptive_threshold(blurred_fingerprint1)
    binary_fingerprint2 = apply_adaptive_threshold(blurred_fingerprint2)

    # Extract the center region of the preprocessed images
    centered1 = extract_center(binary_fingerprint1)
    centered2 = extract_center(binary_fingerprint2)
    
    # Initialize SIFT feature detector
    sift = cv2.SIFT_create()
    # Detect keypoints and compute descriptors for both images
    keypoints_1, descriptors_1 = sift.detectAndCompute(centered1, None)
    keypoints_2, descriptors_2 = sift.detectAndCompute(centered2, None)

    # Match descriptors between the two images using FLANN based matcher
    matches = cv2.FlannBasedMatcher({'algorithm':1, 'trees':10}, {}).knnMatch(descriptors_1, descriptors_2, k=2)
    match_points = []

    # Apply ratio test to find good matches
    for p, q in matches:
        if p.distance < 0.7 * q.distance:
            match_points.append(p)

    # Calculate the matching score based on number of good matches and total keypoints
    keypoints = min(len(keypoints_1), len(keypoints_2))
    score = len(match_points) / keypoints * 100 # not used in this example

    # Print matching results
    print("File name:   " + image_path1 + "   and   " + image_path2)
    print("Number of matches: " + str(len(match_points)))

    # Draw and display the matches between the two images
    result = cv2.drawMatches(centered1, keypoints_1, centered2, keypoints_2, match_points, None)
    result = cv2.resize(result, None, fx=4, fy=4)

    # Resize the result image for display
    scale_percent = 15  # Percentage of the new size
    width = int(result.shape[1] * scale_percent / 100)
    height = int(result.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(result, dim, interpolation=cv2.INTER_AREA)

    # Uncomment below lines to display the resized result image
    # cv2.imshow("Results", resized)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Uncomment below lines to match fingerprints between two images
# match_fingerprints('images/3.jpg', 'images/7.jpg')
# match_fingerprints('images/a1.jpg', 'images/a2.jpg')

# You need to create 'images' folder and put all the fingerprint images in it
# List all .jpg files in 'images' folder and match fingerprints
image_files = glob.glob('images/*.jpg')
for i in range(0, len(image_files), 1):
    if i+1 < len(image_files):
        match_fingerprints(image_files[i], image_files[len(image_files)-1])
