from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import io

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

def projections(org_image, grayscale):
    image_np = np.array(org_image)

    if(grayscale):
        r = image_np[:,:,0]
        r_sum_ver = np.sum(r, axis=0)
        r_sum_hor = np.sum(r, axis=1)
        plt.figure(figsize=(10, 6))
        plt.subplot(2, 1, 1)
        plt.plot( r_sum_hor, color='gray')
        plt.title('Horizontal projection')
        plt.subplot(2, 1, 2)
        plt.plot(r_sum_ver, color='gray')
        plt.subplots_adjust(hspace=0.4)  # Dostosowanie odległości między wykresami
        plt.title('Vertical projection')
        st.pyplot(plt)
        return

    # Rozdzielenie na kanały RGB
    r = image_np[:, :, 0]
    g = image_np[:, :, 1]
    b = image_np[:, :, 2]

    #suma w kolumnach
    r_sum_ver = np.sum(r, axis=0)
    g_sum_ver = np.sum(g, axis=0)
    b_sum_ver = np.sum(b, axis=0)

    # w wierszach
    r_sum_hor = np.sum(r, axis=1)
    g_sum_hor = np.sum(g, axis=1)
    b_sum_hor = np.sum(b, axis=1)

    plt.figure(figsize=(10, 6))

    plt.subplot(2, 1, 1)
    plt.plot( r_sum_hor, color='red', label='Red Channel')
    plt.plot( g_sum_hor, color='green', label='Green Channel')
    plt.plot( b_sum_hor, color='blue', label='Blue Channel')
    plt.title('Horizontal projection')

    # Projekcja pionowa
    plt.subplot(2, 1, 2)
    plt.plot(r_sum_ver, color='red', label='Red Channel')
    plt.plot(g_sum_ver, color='green', label='Green Channel')
    plt.plot(b_sum_ver, color='blue', label='Blue Channel')
    plt.title('Vertical projection')

    plt.subplots_adjust(hspace=0.4)  # Dostosowanie odległości między wykresami
    st.pyplot(plt)

#image compression using sigular value decomposition
def compression_svd(org_image, k):
    image_np = np.array(org_image)
    r, g, b= image_np[:,:,0], image_np[:,:,1], image_np[:,:,2]

    def svd_compress(channel, k):
        U, S, Vt = np.linalg.svd(channel, full_matrices=False)
        S_k = np.zeros((k, k))
        np.fill_diagonal(S_k, S[:k]) #first k singular values
        #print(U[:,:k].shape, S_k.shape, Vt[:k, :].shape)
        compressed_channel = np.dot(U[:, :k], np.dot(S_k, Vt[:k, :]))
        return compressed_channel

    svd_compressed_r, svd_compressed_g, svd_compressed_b=svd_compress(r,k), svd_compress(g,k), svd_compress(b,k)
    compressed_image = np.zeros((r.shape[0], r.shape[1], 3))
    compressed_image[:, :, 0] = svd_compressed_r
    compressed_image[:, :, 1] = svd_compressed_g
    compressed_image[:, :, 2] = svd_compressed_b

    compressed_image = np.clip(compressed_image, 0, 255)
    compressed_image = compressed_image.astype(np.uint8)
    return compressed_image

def get_no_singular_values(org_image):
    image_np = np.array(org_image)
    r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    S_r, S_g, S_b= np.linalg.svd(r, full_matrices=False)[1], np.linalg.svd(g, full_matrices=False)[1], np.linalg.svd(b, full_matrices=False)[1]
    k_max_r, k_max_g, k_max_b = np.count_nonzero(S_r), np.count_nonzero(S_g), np.count_nonzero(S_b)
    return k_max_r, k_max_g, k_max_b

def get_image_bytes(image, format='PNG'):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return len(img_byte_arr.getvalue())


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import io
    import streamlit as st
    image = Image.open("example_photo.jpeg")
    #image.show()
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
    #projections(image, True)



