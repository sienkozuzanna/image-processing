from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np

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

def median_filter(org_image, mask):
    mask_size = mask
    middle = mask_size//2
    image = org_image.copy()
    width, height = image.size
    pixels = image.load()

    new_image = Image.new("RGB", (width, height))
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
    return new_image

if __name__ == "__main__":
    import numpy as np
    from PIL import Image
    
    import streamlit as st
    image = Image.open("example_photo.jpeg")
    #image.show()
    gray  = median_filter(image,9)
    gray.show()



    