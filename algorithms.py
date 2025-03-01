from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st

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

def average_filter(org_image, mask):
    mask_size = mask
    middle = mask_size//2
    image = org_image.copy()
    width, height = image.size
    pixels = image.load()

    new_image = Image.new("RGB", (width, height))
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
    return new_image

def gaussian_filter(org_image, mask_size, sigma):
    image = org_image.copy()

    width,height = image.size
    pixels = image.load()

    new_image = Image.new("RGB", (width, height))
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
    return new_image

def sharpen_filter(org_image, middle):
    image = org_image.copy()
    width, height = image.size
    pixels = image.load()
    new_image = Image.new("RGB", (width, height))
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
    
    mask = create_kernel(middle)
    middle = 3//2
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
    return new_image

    
def histogram(org_image, gray_scale):
    image = org_image.copy()

    def compute_histogram(image, gray_scale=False):
        hist_r, hist_g, hist_b, histogram_gray = [0] * 256, [0] * 256, [0] * 256, [0] * 256
        width, height = image.size

        for h in range(height):
            for w in range(width):
                pixel = image.getpixel((w, h))

                if gray_scale:
                    gray_value = pixel[0]
                    histogram_gray[gray_value] += 1
                else:
                    r, g, b = pixel[0], pixel[1], pixel[2]
                    hist_r[r] += 1
                    hist_g[g] += 1
                    hist_b[b] += 1

        if gray_scale:
            return histogram_gray
        else:
            return hist_r, hist_g, hist_b

    if gray_scale:
        histogram = compute_histogram(image, gray_scale=True)

        plt.figure(figsize=(10, 6))
        plt.hist(range(256), bins=256, weights=histogram, color="black", alpha=0.7, edgecolor="gray")
        plt.title("Grayscale Histogram", fontsize=16, fontweight='bold')
        plt.xlabel("Brightness Level", fontsize=14)
        plt.ylabel("Number of Pixels", fontsize=14)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)

    else:
        hist_r, hist_g, hist_b = compute_histogram(image)

        plt.figure(figsize=(10, 6))
        plt.bar(range(256), hist_r, color="red", alpha=0.7, width=1.0, label="Red", zorder=3)
        plt.bar(range(256), hist_g, color="green", alpha=0.7, width=1.0, label="Green", zorder=2)
        plt.bar(range(256), hist_b, color="blue", alpha=0.7, width=1.0, label="Blue", zorder=1)

        plt.title("RGB Histogram", fontsize=16, fontweight='bold')
        plt.xlabel("Brightness Level", fontsize=14)
        plt.ylabel("Number of Pixels", fontsize=14)
        plt.legend(loc='upper right', fontsize=12)
        plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha=0.7)
        plt.tight_layout()
        st.pyplot(plt)

if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    from matplotlib.pyplot import plt
    import streamlit as st
    image = Image.open("example_photo.jpeg")
    image.show()
    '''
    bright_image = adjust_brightness(image,100.6)
    bright_image.show()

    gray_image = convert_to_gray(image)
    gray_image.show()
    
    contrast_image = adjust_contrast(image, 1.6)
    contrast_image.show()

    negative_image = negative_image(image)
    negative_image.show()
    '''

    # image_binarization=binarization(image, 90)
    # image_binarization.show()
    blurred = average_filter(image, 3)
    gaussian = sharpen_filter(blurred, 11)
    gaussian.show()



