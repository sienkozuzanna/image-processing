from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from algorithms import *

#--------------------------------------Resizing------------------------------------------
def resize_width(org_image, width_mult):
    image = org_image.copy()
    old_width, old_height = image.size
    new_width = int(old_width*width_mult)
    new_image = Image.new("RGB", (new_width, old_height))
    for x in range(new_width):
        x_old = int(x/width_mult)
        for y in range(old_height):
            new_image.putpixel((x, y), image.getpixel((x_old, y)))
    return new_image

def resize_height(org_image, height_mult):
    image = org_image.copy()
    old_width, old_height = image.size
    new_height = int(old_height*height_mult)
    new_image = Image.new("RGB", (old_width, new_height))
    for x in range(old_width):
        for y in range(new_height):
            y_old = int(y/height_mult)
            new_image.putpixel((x, y), image.getpixel((x, y_old)))
    return new_image

def resize_whole(org_image, mult):
    image = org_image.copy()
    old_width, old_height = image.size
    new_height = int(old_height*mult)
    new_width = int(old_width* mult)
    new_image = Image.new("RGB", (new_width, new_height))
    for x in range(new_width):
        x_old = int(x/mult)
        for y in range(new_height):
            y_old = int(y/mult)
            new_image.putpixel((x, y), image.getpixel((x_old, y_old)))
    return new_image
#-----------------------------------------Plot methods------------------------------------------

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


def projections(org_image):
    
    black_white_img = binarization(org_image, threshold=128)
    image_np = np.array(black_white_img)

    # all channels the same 
    r = image_np[:,:,0]
    r_sum_hor = np.sum(r, axis=0)
    r_sum_ver = np.sum(r, axis=1)

    fig, axes = plt.subplots(2, 1, figsize=(6, 8))
    axes[0].bar(range(len(r_sum_hor)) ,r_sum_hor, color='darkgray')
    axes[0].set_title('Horizontal projection')
    axes[1].barh(range(len(r_sum_ver)), r_sum_ver, color='gray') 
    axes[1].set_title('Vertical projection') 
    axes[1].set_position([0.1, 0.1, 0.8, 0.3])
    st.pyplot(plt)

# ------------------- compression functions ----------------------------------------------
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
    compressed_image[:, :, 0], compressed_image[:, :, 1], compressed_image[:, :, 2]= svd_compressed_r, svd_compressed_g, svd_compressed_b
    compressed_image = np.clip(compressed_image, 0, 255)
    compressed_image = compressed_image.astype(np.uint8)

    return compressed_image

def get_no_singular_values(org_image):
    image_np = np.array(org_image)
    r, g, b = image_np[:, :, 0], image_np[:, :, 1], image_np[:, :, 2]
    S_r, S_g, S_b= np.linalg.svd(r, full_matrices=False)[1], np.linalg.svd(g, full_matrices=False)[1], np.linalg.svd(b, full_matrices=False)[1]
    k_max_r, k_max_g, k_max_b = np.count_nonzero(S_r), np.count_nonzero(S_g), np.count_nonzero(S_b)
    return k_max_r, k_max_g, k_max_b

def get_image_bytes(image, format='JPEG'):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format=format)
    return len(img_byte_arr.getvalue())

def compression_mse(org_image, compressed_image):
    org_image = np.array(org_image)
    compressed_image = np.array(compressed_image)
    if org_image.shape[2] != compressed_image.shape[2]:
        #RGBA to RGB
        if org_image.shape[2] == 4:
            org_image = org_image[:, :, :3]
        elif compressed_image.shape[2] == 4:
            compressed_image = compressed_image[:, :, :3]
    n=float(org_image.shape[0]*org_image.shape[1]*org_image.shape[2])
    error=np.sum((org_image.astype("float")-compressed_image.astype("float"))**2)
    return error/n

def compression_psnr(org_image, compressed_image):
    max_pixel=np.max(org_image)
    mse=compression_mse(org_image, compressed_image)
    if mse == 0: float('inf')
    return 20 * np.log10(max_pixel / np.sqrt(mse))

def visualize_compression_errors(org_image, k_values):
    mse_val, psnr_val=[], []
    for k in k_values:
        compressed_image = compression_svd(org_image, k)
        mse_val.append(compression_mse(org_image, compressed_image))
        psnr_val.append(compression_psnr(org_image, compressed_image))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    # mse plot
    ax1.plot(k_values, mse_val, marker='o', color='r', label="MSE")
    ax1.set_title("MSE vs k-value")
    ax1.set_xlabel("k (number of SVD components)")
    ax1.set_ylabel("MSE")
    ax1.grid(True)
    ax1.legend()

    #psnr plot
    ax2.plot(k_values, psnr_val, marker='o', color='b', label="PSNR")
    ax2.set_title("PSNR vs k-value")
    ax2.set_xlabel("k (number of SVD components)")
    ax2.set_ylabel("PSNR (dB)")
    ax2.grid(True)
    ax2.legend()

    plt.subplots_adjust(hspace=0.2, wspace=0.2)
    return fig

if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    import io
    import streamlit as st
    import math
    image = Image.open("example_photo.jpeg")
    #image.show()
    resized = resize_whole(image, 0.5)
    resized.show()

    

   


    