
import numpy as np
from PIL import Image 
def adjust_brightness(org_image, brightness=0):
    
    image = org_image.copy()
    height,width = image.size
    for h in range(height):
        for w in range(width):
            pixel = image.getpixel((h,w))
            if(len(pixel)==4):
                r,g,b,a = pixel
                new_pixel_val = (r+brightness, g+brightness, b+brightness, a)
            else:
                r,g,b = pixel
                new_pixel_val = (r+brightness, g+brightness, b+brightness)

            new_pixel_val = tuple(int(min(max(val,0), 255)) for val in new_pixel_val) #value of each channel can be 0-255
            image.putpixel((h,w), new_pixel_val)
    return image

def convert_to_gray(org_image):


    image = org_image.copy()
    height,width = image.size
    for h in range(height):
        for w in range(width):
            pixel = image.getpixel((h,w))
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

            image.putpixel((h,w), new_pixel_val)
    return image


if __name__ == "__main__":
    import numpy as np
    from PIL import Image 
    image = Image.open("C:\\Users\\rogal\\SEM6\\Biometria\\Proj1\\kolorowy.jpg")
    bright_image = adjust_brightness(image,100.6)

    bright_image.show()

    gray_image = convert_to_gray(image)
    gray_image.show()

    image.show()



