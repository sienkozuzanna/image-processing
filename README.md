# Image-processing

## Overview
This project implements image processing algorithms from scratch using Python. The application is built with the Streamlit library, providing an interactive graphical user interface (GUI) for users to apply various image processing techniques in real-time. The solution makes use of several essential Python libraries, including NumPy, Pillow, Matplotlib, and Pandas, to handle image operations, visualize data, and perform mathematical computations.

## Features
The solution is implemented as a desktop application with an easy-to-understand interface that allows users to perform single or multiple operations on a loaded image. The effects of the transformations are visible in real-time, allowing users to compare before-and-after results. The application supports loading images in various formats, such as JPEG, PNG, and JPG, and allows users to save the processed images on their disk.

### Supported Image Operations

1. **Grayscale Conversion**  
   This method converts an image to grayscale. The conversion is done by computing the average value of the 3 color channels (R, G, and B), or by using more advanced methods that account for the human eye's sensitivity to different colors.  
   Additionally, a Decomposition Method is implemented, where the grayscale image is created using the minimum or maximum value from the color channels independently.

2. **Brightness Adjustment**  
   An operation that increases or decreases the brightness of an image. Adding a positive constant to each pixel value brightens the image, while subtracting a constant darkens it.

3. **Contrast Adjustment**  
   This method modifies the difference between the lightest and darkest pixels in the image using a specific formula.

4. **Negative**  
   A transformation that inverts the color values of all pixels in the image.

5. **Binarization**  
   Converts the image into black and white (binary) based on a threshold value. Each pixel is assigned either a white (255) or black (0) value depending on whether its grayscale value exceeds the threshold. The user can manually adjust the threshold.

6. **Smoothing Filters**  
   Includes filters like the **Mean Filter**, **Median Filter**, and **Gaussian Filter**, all of which blur the image to reduce noise or detail.

7. **Edge Detection**  
   Implements methods like **Sobel Filter** and **Roberts Cross** to detect edges in an image, which are essential for object detection or image analysis.

8. **Resize**  
   The image can be resized, either by scaling or cropping, to meet specific size requirements.

9. **Projection Graphs**  
    Vertical and horizontal projection graphs are available to visualize the distribution of pixel intensities across the image.

10. **Histograms**  
    The application generates histograms to visualize the distribution of pixel intensity values in an image. The histograms represent the frequency of occurrence of each pixel intensity level.

### Image Compression
The project also includes methods for image compression using **Singular Value Decomposition (SVD)**. This allows for representing an image in a more compact form while minimizing quality loss.

### Quality Metrics for Compression
- **Mean Squared Error (MSE)**: A metric to measure the average squared difference between the original and compressed images.
- **Peak Signal-to-Noise Ratio (PSNR)**: A metric used to evaluate the quality of the compressed image by comparing the maximum possible signal value to the noise introduced during compression.

### Performance Considerations
Since the image processing algorithms were implemented manually, the computational complexity of each operation is at least O(n²), which may result in noticeable delays when processing larger images or using large filter masks.

### Authors
- [Katarzyna Rogalska](https://github.com/katarzynarogalska)
- [Zuzanna Sieńko](https://github.com/sienkozuzanna)
