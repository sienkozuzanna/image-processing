from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import io
import math

# ----------------------------- Method to adjust brightness -----------------------------------------
def adjust_brightness(org_image, brightness=0):
    image = org_image.copy()
    width,height = image.size
    for h in range(height):
        for w in range(width):
            pixel = image.getpixel((w,h))
            if(len(pixel)==4):
                r,g,b,a = pixel #RGBA r-red, g-green, b-blue, a-alpha (opacity)
                new_pixel_val = (r+brightness, g+brightness, b+brightness, a)
            else:
                r,g,b = pixel
                new_pixel_val = (r+brightness, g+brightness, b+brightness)

            new_pixel_val = tuple(int(min(max(val,0), 255)) for val in new_pixel_val) #value of each channel can be 0-255
            image.putpixel((w,h), new_pixel_val)
    return image

#-----------------------------Different methods for gray conversion-------------------------------------

def convert_to_gray(org_image):
    image = org_image.copy()
    width, height = image.size
    for h in range(height):
        for w in range(width):
            pixel = image.getpixel((w,h))
            if(len(pixel)==4):
                r,g,b,a = pixel
                gray = 0.299*r + 0.587*g + 0.114*b  #values from literature
                gray = int(gray)
                new_pixel_val = (gray,gray,gray, a)
            else:
                r,g,b = pixel
                gray = 0.299*r + 0.587*g + 0.114*b
                gray = int(gray)
                new_pixel_val = (gray,gray,gray)

            image.putpixel((w,h), new_pixel_val)
    return image

def convert_to_gray_naive(org_image):
    image = org_image.copy()
    width, height = image.size
    for h in range(height):
        for w in range(width):
            pixel = image.getpixel((w,h))
            if(len(pixel)==4):
                r,g,b,a = pixel
                gray = (r+g+b)/3
                gray = int(gray)
                new_pixel_val = (gray,gray,gray, a)
            else:
                r,g,b = pixel
                gray = (r+g+b)/3
                gray = int(gray)
                new_pixel_val = (gray,gray,gray)

            image.putpixel((w,h), new_pixel_val)
    return image

def convert_to_gray_decomp(org_image):
    image = org_image.copy()
    width, height = image.size
    for h in range(height):
        for w in range(width):
            pixel = image.getpixel((w,h))
            if(len(pixel)==4):
                r,g,b,a = pixel
                gray = min(r,g,b)
                gray = int(gray)
                new_pixel_val = (gray,gray,gray, a)
            else:
                r,g,b = pixel
                gray = min(r,g,b)
                gray = int(gray)
                new_pixel_val = (gray,gray,gray)

            image.putpixel((w,h), new_pixel_val)
    return image

#-----------------------------------Method for adjusting contrast-----------------------------------
def adjust_contrast(org_image, contrast_factor):
    #contrast_factor > 1, contrast increases
    #contrast_factor < 1, contrast decreases
    #contrast_factor = 1 image unchanged
    image = org_image.copy()
    width ,height = image.size
    for h in range(height):
        for w in range(width):
            pixel = image.getpixel((w,h))
            if(len(pixel)==4):
                r,g,b,a = pixel
                r = int((r - 128) * contrast_factor + 128)
                g = int((g - 128) * contrast_factor + 128)
                b = int((b - 128) * contrast_factor + 128)

                #limiting pixel values to the range 0-255
                r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
                new_pixel_val = (r, g, b, a)
            else:
                r,g,b = pixel
                r = int((r - 128) * contrast_factor + 128)
                g = int((g - 128) * contrast_factor + 128)
                b = int((b - 128) * contrast_factor + 128)
                r, g, b = max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))
                new_pixel_val = (r, g, b)

            image.putpixel((w,h), new_pixel_val)
    return image

# ------------------------------- Method for getting a negative ------------------------------
def negative_image(org_image):
    image = org_image.copy()
    width, height = image.size
    for h in range(height):
        for w in range(width):
            pixel = image.getpixel((w, h))
            if (len(pixel) == 4):
                r, g, b, a = pixel
                image.putpixel((w, h), (255 - r, 255 - g, 255 - b, a))
            else:
                r, g, b = pixel
                image.putpixel((w, h), (255 - r, 255 - g, 255 - b))
    return image

# --------------------------Method for binarization-----------------------------------------
def binarization(org_image, threshold):
    #first convert to gray
    image=convert_to_gray(org_image.copy())
    width, height = image.size
    for h in range(height):
        for w in range(width):
            pixel = image.getpixel((w, h))
            new_pixel_value = 255 if pixel[0] > threshold else 0 #pixel values: (gray, gray, gray, .)
            if len(pixel) == 4:
                image.putpixel((w, h), (new_pixel_value, new_pixel_value, new_pixel_value, pixel[3]))
            else:
                image.putpixel((w, h), (new_pixel_value, new_pixel_value, new_pixel_value))
    return image

#------------------------------ Filter methods -------------------------------------------
def padding_edges(image: Image.Image, middle: int):
    """Dodaje padding do obrazu, kopiując najbliższe piksele na krawędziach."""
    np_image = np.array(image)
    padded_array = np.pad(np_image, ((middle, middle), (middle, middle), (0, 0)), mode='edge')
    return Image.fromarray(padded_array)

def average_filter(org_image, mask):
    mask_size = mask
    middle = mask_size//2

    padded_image = padding_edges(org_image, middle)
    width, height = padded_image.size
    pixels = padded_image.load()

    new_image = padded_image.copy()
    new_pixels = new_image.load()

    for x in range(middle, width-1):
        for y in range(middle, height-1):
            sum_r = 0
            sum_g = 0
            sum_b =0
            count = 0
            for i in range(-middle, middle+1):
                for j in range(-middle, middle+1):
                    if 0 <= x + i < width and 0 <= y + j < height:
                        r,g,b = pixels[x+i, y+j]
                        sum_r += r
                        sum_g+=g
                        sum_b += b
                        count+=1
            avg_r = sum_r//count
            avg_g = sum_g//count
            avg_b = sum_b//count

            new_pixels[x,y] = (avg_r, avg_g, avg_b)
    
    cropped_image = new_image.crop((middle, middle, width - middle, height - middle))
    return cropped_image

def median_filter(org_image, mask):
    mask_size = mask
    middle = mask_size//2

    padded_image = padding_edges(org_image, middle)
    width, height = padded_image.size
    pixels = padded_image.load()

    new_image = padded_image.copy()
    new_pixels = new_image.load()

    for x in range(middle, width-1):
        for y in range(middle, height-1):
            r_vals=[]
            g_vals=[]
            b_vals=[]
            for i in range(-middle, middle+1):
                for j in range(-middle, middle+1):
                    if 0 <= x + i < width and 0 <= y + j < height:
                        r,g,b = pixels[x+i, y+j]
                        r_vals.append(r)
                        g_vals.append(g)
                        b_vals.append(b)
            med_r = np.median(r_vals)
            med_g = np.median(g_vals)
            med_b = np.median(b_vals)

            new_pixels[x,y] = (int(med_r), int(med_g), int(med_b))

    cropped_image = new_image.crop((middle, middle, width - middle, height - middle))
    return cropped_image

def gaussian_filter(org_image, sigma):
    org_width, org_height = org_image.size
    if(min(org_width, org_height)<9 and min(org_width, org_height)>=3):
        mask_size=3
    else:
        mask_size = 9
    middle = mask_size//2
    padded_image = padding_edges(org_image, middle)
    width, height = padded_image.size
    pixels = padded_image.load()
    new_image = padded_image.copy()
    new_pixels = new_image.load()
    
    def create_kernel(size, sigma=sigma):
        kernel = np.fromfunction(
        lambda x, y: (1/ (2 * np.pi * sigma ** 2)) * 
                     np.exp(- ((x - (size // 2))**2 + (y - (size // 2))**2) / (2 * sigma ** 2)),
        (size, size))
        return kernel / np.sum(kernel)
    
    kernel = create_kernel(mask_size)
    middle = mask_size//2
    for x in range(middle, width-middle):
        for y in range(middle, height-middle):
            sum_r = sum_g = sum_b = 0
            count =0
            for i in range(-middle, middle+1):
                for j in range(-middle, middle+1):
                    if 0 <= x + i < width and 0 <= y + j < height:
                        r,g,b = pixels[x+i, y+j]
                        weight = kernel[i + middle, j + middle]
                        sum_r += r * weight
                        sum_g+=g * weight
                        sum_b += b * weight
                        count+=1
            new_pixels[x,y] = (int(sum_r), int(sum_g), int(sum_b))
    cropped_image = new_image.crop((middle, middle, width - middle, height - middle))
    return cropped_image


def sharpen_filter(org_image, middle_el_weight):
    mask_size =3
    middle = mask_size//2
    padded_image = padding_edges(org_image, middle)
    width, height = padded_image.size
    pixels = padded_image.load()
    new_image = padded_image.copy()
    new_pixels = new_image.load()

    def create_kernel(middle,size=3):
        kernel = [[0 for _ in range(size)] for _ in range(size)]
        kernel[size // 2][size // 2] = middle # Centralny element ma dużą wagę

        # Ustalanie pozostałych elementów na -1
        for i in range(size):
            for j in range(size):
                if i != size // 2 or j != size // 2:
                    kernel[i][j] = -1
        return kernel
    
    mask = create_kernel(middle_el_weight)
    
    for x in range(middle, width-middle):
        for y in range(middle, height-middle):
            r_total, g_total, b_total = 0, 0, 0
            for i in range(-middle, middle+1):
                for j in range(-middle, middle+1):
                    if 0 <= x + i < width and 0 <= y + j < height:
                        r,g,b = pixels[x+i, y+j]
                        weight = mask[i + middle][j + middle]
                        r_total += r * weight
                        g_total+=g * weight
                        b_total += b * weight
  
            new_pixels[x,y] = (min(max(r_total, 0), 255), min(max(g_total, 0), 255), min(max(b_total, 0), 255))
    cropped_image = new_image.crop((middle, middle, width - middle, height - middle))
    return cropped_image

def custom_filter(org_image, w11,w12,w13,w21,w22,w23,w31,w32,w33):
    mask_size =3
    middle = mask_size//2
    padded_image = padding_edges(org_image, middle)
    width, height = padded_image.size
    pixels = padded_image.load()
    new_image = padded_image.copy()
    new_pixels = new_image.load()

    def create_kernel(w11,w12,w13,w21,w22,w23,w31,w32,w33):
        kernel = [[w11,w12,w13],[w21,w22,w23],[w31,w32,w33]]
        return kernel
    
    mask = create_kernel(w11,w12,w13,w21,w22,w23,w31,w32,w33)
    
    for x in range(middle, width-middle):
        for y in range(middle, height-middle):
            r_total, g_total, b_total = 0, 0, 0
            for i in range(-middle, middle+1):
                for j in range(-middle, middle+1):
                    if 0 <= x + i < width and 0 <= y + j < height:
                        r,g,b = pixels[x+i, y+j]
                        weight = mask[i + middle][j + middle]
                        r_total += r * weight
                        g_total+=g * weight
                        b_total += b * weight
            r_total = int(r_total)
            g_total=int(g_total)
            b_total = int(b_total)
  
            new_pixels[x,y] = (min(max(r_total, 0), 255), min(max(g_total, 0), 255), min(max(b_total, 0), 255))
    cropped_image = new_image.crop((middle, middle, width - middle, height - middle))
    return cropped_image

#--------------------------- Edge detection methods---------------------------------------

def roberts_cross(org_image):

    Gx = [[1,0],[0,-1]]
    Gy = [[0,1],[-1,0]]

    image = org_image.copy()
    width, height = image.size
    new_image = Image.new("RGB", (width, height))
    new_pixels = new_image.load()

    for i in range(0, width-1):
        for j in range(0, height-1):
            current_submatrix_r = np.array([[image.getpixel((i,j))[0], image.getpixel((i+1,j))[0]],[image.getpixel((i,j+1))[0],image.getpixel((i+1,j+1))[0]]])
            current_submatrix_g = np.array([[image.getpixel((i,j))[1], image.getpixel((i+1,j))[1]],[image.getpixel((i,j+1))[1],image.getpixel((i+1,j+1))[1]]])
            current_submatrix_b = np.array([[image.getpixel((i,j))[2], image.getpixel((i+1,j))[2]],[image.getpixel((i,j+1))[2],image.getpixel((i+1,j+1))[2]]])
            def calculate_result(current_submatrix):
                x = np.sum(np.multiply(Gx, current_submatrix))
                y = np.sum(np.multiply(Gy, current_submatrix))
                magnitude = np.sqrt(x**2 + y**2)
                return magnitude

            new_pixels[i,j] = (int(calculate_result(current_submatrix_r)), int(calculate_result(current_submatrix_g)), int(calculate_result(current_submatrix_b)))
    return new_image

def sobel_filter(org_image):
    Gx = [[1,0,-1],[2,0,-2],[1,0,-1]]
    Gy = [[1,2,1],[0,0,0],[-1,-2,-1]]
    image = org_image.copy()
    width, height = image.size
    new_image = Image.new("RGB", (width, height))
    new_pixels = new_image.load()

    for i in range(1, width-1):
        for j in range(1, height-1):
            current_submatrix_r = np.array([
                                [image.getpixel((i-1,j-1))[0],image.getpixel((i-1,j))[0], image.getpixel((i-1,j+1))[0]],
                                [image.getpixel((i,j-1))[0],image.getpixel((i,j))[0],image.getpixel((i,j+1))[0]],
                                [image.getpixel((i+1,j-1))[0], image.getpixel((i+1,j))[0], image.getpixel((i+1,j+1))[0]]])
            current_submatrix_g = np.array([
                                [image.getpixel((i-1,j-1))[1],image.getpixel((i-1,j))[1], image.getpixel((i-1,j+1))[1]],
                                [image.getpixel((i,j-1))[1],image.getpixel((i,j))[1],image.getpixel((i,j+1))[1]],
                                [image.getpixel((i+1,j-1))[1], image.getpixel((i+1,j))[1], image.getpixel((i+1,j+1))[1]]])
            current_submatrix_b = np.array([
                                [image.getpixel((i-1,j-1))[2],image.getpixel((i-1,j))[2], image.getpixel((i-1,j+1))[2]],
                                [image.getpixel((i,j-1))[2],image.getpixel((i,j))[2],image.getpixel((i,j+1))[2]],
                                [image.getpixel((i+1,j-1))[2], image.getpixel((i+1,j))[2], image.getpixel((i+1,j+1))[2]]])
            def calculate_result(current_submatrix):
                x = np.sum(np.multiply(Gx, current_submatrix))
                y = np.sum(np.multiply(Gy, current_submatrix))
                magnitude = np.sqrt(x**2 + y**2)
                return magnitude

            new_pixels[i,j] = (int(calculate_result(current_submatrix_r)), int(calculate_result(current_submatrix_g)), int(calculate_result(current_submatrix_b)))
    return new_image


def DCT_matrix(N=8): #N=8 for image compression
    #creating NxN DTC matrix
    T=np.zeros((N,N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                T[i, j] = 1 / np.sqrt(N)
            else:
                T[i, j] = np.sqrt(2 / N) * np.cos(((2 * j + 1) * i * np.pi) / (2 * N))
    return T

def DCT(org_image):
    image_np = np.array(org_image)
    w, h, channels = image_np.shape
    if channels == 4: channels = 3

    T_w = DCT_matrix(8)
    T_h = DCT_matrix(8)
    M = image_np - 128

    quantized_matrix = np.zeros_like(M)
    quantization_matrix = np.array([
        [16, 11, 10, 16, 24, 40, 51, 61],
        [12, 12, 14, 19, 26, 58, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99]
    ])

    for c in range(channels):
        for i in range(0, w, 8):
            for j in range(0, h, 8):
                block = M[i:i + 8, j:j + 8, c]
                if block.shape == (8, 8):
                    D = np.dot(np.dot(T_w, block), T_h.T)
                    quantized_block = np.round(D / quantization_matrix)
                    quantized_matrix[i:i + 8, j:j + 8, c] = quantized_block

    quantized_matrix = np.clip(quantized_matrix + 128, 0, 255).astype(np.uint8)
    return quantized_matrix


def IDCT_matrix(N):
    T_inv = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i == 0:
                T_inv[i, j] = 1 / np.sqrt(N)
            else:
                T_inv[i, j] = np.sqrt(2 / N) * np.cos(np.pi / N * (2 * j + 1) * i / 2)
    return T_inv.T


def IDCT(org_image):
    image_np = np.array(org_image)
    w, h, channels = image_np.shape
    if channels == 4: channels = 3

    T_w = DCT_matrix(8)
    T_h = DCT_matrix(8)
    M = image_np - 128

    idct_matrix = np.zeros_like(M)

    for c in range(channels):
        for i in range(0, w, 8):
            for j in range(0, h, 8):
                block = M[i:i + 8, j:j + 8, c]
                if block.shape == (8, 8):
                    D = np.dot(np.dot(T_w.T, block), T_h)
                    idct_matrix[i:i + 8, j:j + 8, c] = D

    idct_matrix = np.clip(idct_matrix + 128, 0, 255).astype(np.uint8)
    return idct_matrix


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import io
    import streamlit as st
    import math
    image = Image.open("example_photo.jpeg")
    #image.show()
    blurred = sharpen_filter(image, 11)
    blurred.show()



