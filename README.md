# 📌 Fingerprint Recognition with Mobile Camera

## 📜 Project Description
This project develops a fingerprint recognition system that processes images captured with standard mobile phone cameras. Unlike traditional fingerprint scanners, this approach makes identification more accessible using everyday devices. The system applies advanced image processing techniques to filter, enhance, and analyze fingerprint patterns, ultimately matching them against stored references.

## 🎯 Key Features
- Processes fingerprints from regular smartphone camera images
- Applies multiple image enhancement techniques to improve fingerprint visibility
- Extracts key fingerprint features for reliable matching
- Compares fingerprints using feature-based matching algorithms
- Calculates similarity scores between fingerprint samples

## 🛠 Technologies Used
- **Python 3.x**: Main programming language
- **OpenCV**: Advanced image processing capabilities
- **NumPy**: Efficient numerical operations on image arrays
- **Matplotlib**: Visualization of processing stages and results
- **PIL (Pillow)**: Image loading with EXIF orientation handling

## 📂 Directory Structure
- `200104004067.py`: Core implementation file containing all processing and matching algorithms
- `examples/`: Sample fingerprint images (blurred for privacy)
- `README.md`: This documentation file
- `LICENSE`: MIT License documentation

## 📸 Capturing Fingerprint Images
For best results when capturing fingerprint images:
1. Use good lighting conditions (natural light works best)
2. Position your finger within the center rectangle of the camera frame
3. Hold the camera steady to avoid motion blur
4. Ensure your fingertip is clean and dry

<img src="examples/1.png" alt="Fingerprint Positioning Example" width="400"/>

*Example showing correct finger positioning within the center rectangle*

## 🔍 How It Works

### 1. Image Loading and Preprocessing
The system begins by loading the fingerprint image and preparing it for processing:
- Loads the image from the specified path
- Handles EXIF orientation data to ensure correct image alignment
- Converts the image to grayscale for more effective processing

```python
def load_image(image_path):
    # Loads image, handles orientation, converts to grayscale
```

<img src="examples/2.png" alt="Grayscale Conversion" width="400"/>

*Example of an image after loading and grayscale conversion*

### 2. Image Enhancement
Multiple filtering techniques are applied to enhance fingerprint ridge patterns:
- **Bilateral Filtering:** Reduces noise while preserving ridge edges

```python
def apply_bilateral_filter(image, d, sigma_color, sigma_space):
    # Applies edge-preserving noise reduction
```

- **Adaptive Thresholding:** Converts the image to binary (black and white) to highlight ridges

```python
def apply_adaptive_threshold(image):
    # Creates binary image using local neighborhood information
```

- **Center Region Extraction:** Focuses on the most important part of the fingerprint

```python
def extract_center(image):
    # Extracts the central portion containing the core fingerprint pattern
```

<img src="examples/3.png" alt="Center Extraction" width="400"/>

*Example showing the extracted center region of a fingerprint*

- **Morphological Operations:** Enhances ridge definition and removes artifacts

```python
def apply_morphological_operations(image, kernel_size, iterations):
    # Applies opening and closing operations to refine ridge patterns
```

<img src="examples/4.png" alt="Before Morphological Operations" width="400"/>

*Before applying morphological operations*

<img src="examples/5.png" alt="After Morphological Operations" width="400"/>

*After applying morphological operations - note the improved ridge definition*

### 3. Fingerprint Matching
The system uses feature-based matching to compare fingerprints:

- Feature Extraction: Uses SIFT (Scale-Invariant Feature Transform) to identify distinctive points
- Feature Matching: Employs FLANN (Fast Library for Approximate Nearest Neighbors) to efficiently match features
- Ratio Test: Filters matches based on distance ratios to ensure quality
- Similarity Scoring: Calculates a match percentage based on the number of matching features

```python
def match_fingerprints(image_path1, image_path2):
    # Processes two fingerprint images and determines if they match
```

## 🏗 Setup and Installation

### Prerequisites
- Python 3.6 or newer
- Basic understanding of Python and command line operations

### Installation Steps
1. Clone this repository:
```bash
git clone https://github.com/yourusername/fingerprint-recognition-with-mobile-camera.git
cd fingerprint-recognition-with-mobile-camera
```

2. Install required libraries:
```bash
pip install opencv-python numpy matplotlib pillow
```

3. Prepare your fingerprint images:
- Create an images folder in the project directory
- Add your fingerprint images in JPG format

### Running the Application
Run the main script to process and match fingerprints
```bash
python 200104004067.py
```

To customize processing or matching, modify the script parameters or uncomment visualization sections.

## 📋 Usage Examples

### Basic Fingerprint Processing

```python
# Load and process a single fingerprint image
fingerprint = load_image('images/sample.jpg')
blurred_fingerprint = apply_bilateral_filter(fingerprint)
binary_fingerprint = apply_adaptive_threshold(blurred_fingerprint)
centered = extract_center(binary_fingerprint)
processed_image = apply_morphological_operations(centered)
```

### Fingerprint Matching

```python
# Compare two fingerprints
match_fingerprints('images/finger1.jpg', 'images/finger2.jpg')
```

## ⚠️ Security and Data Privacy

- Fingerprint data is sensitive biometric information
- This project doesn't store or transmit fingerprint data to external services
- The example images in this repository are blurred to protect privacy
- For production applications, implement proper encryption and secure storage

## 📜 License

This project is licensed under the MIT License. For details, please refer to the LICENSE file.