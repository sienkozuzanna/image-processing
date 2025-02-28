from PIL import Image

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


if __name__ == "__main__":
    import numpy as np
    from PIL import Image
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

    image_binarization=binarization(image, 90)
    image_binarization.show()




