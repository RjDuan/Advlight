"""
This file aims for simulating the light by given prameters, including:
alpha: Initilized luminous intensity
beta: attenuation paramter, describe the illuminous changes with distance
wavelength (w): color of the light
light type (t): point light, tube light and area light
location (x,y,w): the position of the light sourcel

TODO:
1. Add argument parser
"""

from scipy import ndimage
import numpy as np   
import matplotlib.pyplot as plt
import math
from PIL import Image
import os
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--target_label",  dest='target_label', nargs='?', type = int,
                    help="The target label for target attack", default=920)
parser.add_argument("--batch_size",  dest='batch_size', nargs='?', type = int,
                    help="Batch size.", default=16)
parser.add_argument("--start_img_path",             dest='start_img_path',              nargs='?',
                    help='Path to start image', default='./elements/light_10.jpg')
parser.add_argument("--result_dir",             dest='result_dir',              nargs='?',
                    help='Path to save the results', default='./robust_features')
args = parser.parse_args() 

def simple_add(base_img, light_pattern, alpha = 1.0):
        base_img = base_img.astype(np.float32)
        light_pattern = light_pattern.astype(np.float32)
        resized_light_pattern = cv2.resize(light_pattern, (base_img.shape[1], base_img.shape[0]))
        c = cv2.addWeighted(base_img, 1.0 , resized_light_pattern, alpha , 0)
        return c
    
def gaussian_add(base_img, light_pattern, eps = 128 ):
        base_img = base_img.astype(np.float32)
        resized_light_pattern = cv2.resize(light_pattern, (base_img.shape[1], base_img.shape[0]))
        mu, sigma = 0, 1.0 # mean and standard deviation
        s = np.random.normal(mu, sigma, base_img.shape)
        gaussian_matric =np.clip(s * eps *  (resized_light_pattern/255.0), -1 * eps, eps)
        print(np.amax(gaussian_matric))
        c = base_img  + gaussian_matric
        return c 
    
def wavelength_to_rgb(wavelength, gamma=0.8):
    """
    Description:
    Given a wavelength in the range of (380nm, 750nm), visible light range.
    a tuple of intergers for (R,G,B) is returned. 
    The integers are scaled to the range (0, 1).
    
    Based on code: http://www.noah.org/wiki/Wavelength_to_RGB_in_Python

    Parameters:
        Wavelength: the given wavelength range in (380, 750) 
    Returns:
        (R,G,B): color range in (0,1)
    """
    wavelength = float(wavelength)
    if wavelength >= 380 and wavelength <= 440:
        attenuation = 0.3 + 0.7 * (wavelength - 380) / (440 - 380)
        R = ((-(wavelength - 440) / (440 - 380)) * attenuation) ** gamma
        G = 0.0
        B = (1.0 * attenuation) ** gamma
    elif wavelength >= 440 and wavelength <= 490:
        R = 0.0
        G = ((wavelength - 440) / (490 - 440)) ** gamma
        B = 1.0
    elif wavelength >= 490 and wavelength <= 510:
        R = 0.0
        G = 1.0
        B = (-(wavelength - 510) / (510 - 490)) ** gamma
    elif wavelength >= 510 and wavelength <= 580:
        R = ((wavelength - 510) / (580 - 510)) ** gamma
        G = 1.0
        B = 0.0
    elif wavelength >= 580 and wavelength <= 645:
        R = 1.0
        G = (-(wavelength - 645) / (645 - 580)) ** gamma
        B = 0.0
    elif wavelength >= 645 and wavelength <= 750:
        attenuation = 0.3 + 0.7 * (750 - wavelength) / (750 - 645)
        R = (1.0 * attenuation) ** gamma
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return (R, G, B)   

def area_light_generation(direction, alpha, beta, wavelength, w = 150, h=150):
    """
    Generating area light with given parameters
    Args:
        direction (s): Denote the direction where the light illuminates, "left", "right", "top", "bottom".
        alpha (int): An integer (0,1] denotes the illumination intensity.
        beta (int): Annuatation factor.
        wavelength (interger): An interger (380, 750) denotes the wavelength of the light.
        w (int, optional): Width. Defaults to 400.
        h (int, optional): Height. Defaults to 400.

    Returns:
        area_light: an numpy array with shape (w,h,3)
    """
    area_light = np.zeros((w,h,3))
    
    full_light_end_x = int(math.sqrt(beta) + 0.5)
    light_end_x = int(math.sqrt(beta * 100) + 0.5)
    
    # Get the color by given wavelength
    c = wavelength_to_rgb(wavelength)
    for x in range(w):
        if x < full_light_end_x:
            attenuation = 1.0
        else:
            attenuation = beta /(x*x)
            print("Current x: {} with attenuation: {}".format(x, attenuation))
        area_light[:,x,0] = c[0] * alpha * attenuation
        area_light[:,x,1] = c[1] * alpha * attenuation
        area_light[:,x,2] = c[2] * alpha * attenuation
    if direction == "top":
        area_light = ndimage.rotate(area_light, 90, reshape =False)
    elif direction == "right":
        area_light = ndimage.rotate(area_light, 180, reshape = False)
    elif direction == "bottom":
        area_light = ndimage.rotate(area_light, 270, reshape = False)
    return area_light


def tube_light_generation_by_func(k, b, alpha, beta, wavelength, w = 400, h = 400):
    """Description:
    This functio generates a tube light (light beam) with given paratmers, in which,
    k and b represent the function y = k*x + b 
    # TODO:
    Test k, b range
    Args:
        k (int): y = k*x + b
        b (int): y = k*x + b 
        alpha (int): An integer (0,1] denotes the illumination intensity.
        beta (int): Annuatation factor. depends on the annuatation function, current beta/distance^2
        wavelength (interger): An interger (380, 750) denotes the wavelength of the light.
        w (int, optional): Width. Defaults to 400.
        h (int, optional): Height. Defaults to 400.

    Returns:
        tube light:  an numpy array with shape (w,h,3)
    """
    
    tube_light = np.zeros((w,h,3))
    full_light_end_y = int(math.sqrt(beta) + 0.5)
    light_end_y = int(math.sqrt(beta * 20) + 0.5)
    
    c = wavelength_to_rgb(wavelength)
    
    for x in range(w):
        for y in range(h):
            distance = abs(k*x - y + b) / math.sqrt(1 + k*k)
            if distance < 0:
                print(distance)
            if distance <= full_light_end_y:
                tube_light[y,x,0] = c[0] * alpha
                tube_light[y,x,1] = c[1] * alpha
                tube_light[y,x,2] = c[2] * alpha
            elif distance> full_light_end_y and distance <= light_end_y:
                attenuation = beta/(distance * distance)
                tube_light[y,x,0] = c[0] * alpha * attenuation
                tube_light[y,x,1] = c[1] * alpha * attenuation
                tube_light[y,x,2] = c[2] * alpha * attenuation              
    
    return tube_light
def tube_light_generation(angle, alpha, beta, wavelength, w = 400, h = 400):
    """
    Generating tube light with given parameters.
    Args:
        st (dictionary): A dictionary includes the position of the light: 
                        "start_point": (x,y), 
                        "tube_width" :t_b
                        "rotation_angle": r_a
        alpha (int): An integer (0,1] denotes the illumination intensity.
        beta (int): Annuatation factor.
        wavelength (interger): An interger (380, 750) denotes the wavelength of the light.
        w (int, optional): Width. Defaults to 400.
        h (int, optional): Height. Defaults to 400.

    Returns:
        tube_light: an numpy array with shape (w,h,3)
    TODO:
    1. Check full_light_y, light_end_y with h/w
    2. A more reasonable roation
    """
    tube_light = np.zeros((w,h,3))
    
    full_light_end_y = int(math.sqrt(beta) + 0.5)
    light_end_y = int(math.sqrt(beta * 10) + 0.5)
    c = wavelength_to_rgb(wavelength)
    # c = (1.0,1.0,1.0)
    # For calculation
    total_distance_y = light_end_y + full_light_end_y
    mid_diff = h//2 - (light_end_y + full_light_end_y//2)
    print(mid_diff)
    # full light
    for y in range(light_end_y, total_distance_y+1):
        tube_light[y,:,0] = c[0] * alpha
        tube_light[y,:,1] = c[1] * alpha
        tube_light[y,:,2] = c[2] * alpha
    
    # Total width of the light beam
    total_width = light_end_y * 2 + full_light_end_y
    for y in range(light_end_y+1):
        distance = total_distance_y - y
        attenuation = beta/(distance * distance)
        # print("Current y: {}, and actual distance {} with attenuation {}".format(y, distance,attenuation))
        # cur_y = mid_diff + y
        tube_light[y,:,0] = c[0] * alpha * attenuation
        tube_light[y,:,1] = c[1] * alpha * attenuation
        tube_light[y,:,2] = c[2] * alpha * attenuation
        # Also simulate the other side
        tube_light[total_width - y,:,0] = c[0] * alpha * attenuation
        tube_light[total_width - y,:,1] = c[1] * alpha * attenuation
        tube_light[total_width - y,:,2] = c[2] * alpha * attenuation
        
        
    tube_light = ndimage.rotate(tube_light, angle, reshape =False)
    
    return tube_light

def point_light_generation(st, alpha, beta, wavelength, w = 400, h = 400):
    """
    Generating point light with given parameters.
    Args:
        st (dictionary): A dictionary includes the position of the light: 
                        "start_point": (x,y)
                        "radium": r, denote the radium of the point light
        alpha (int): An integer (0,1] denotes the illumination intensity.
        beta (int): Annuatation factor.
        wavelength (interger): An interger (380, 750) denotes the wavelength of the light.
        w (int, optional): Width. Defaults to 400.
        h (int, optional): Height. Defaults to 400.

    Returns:
        point_light: an numpy array with shape (w,h,3)
    """
    point_light = np.zeros((w,h,3))
    return point_light
if __name__ == "__main__":
    h = 40
    w = 370
    color_bar = np.zeros((h, w, 3))

    for w in range(380,70):
        c = wavelength_to_rgb(w)
        # Position start to save color infor
        x = w - 380
        # print("Color:", c[0]* 255, c[1]*255, c[2]*255)
        color_bar[:,x,0] = c[0]
        color_bar[:,x,1] = c[1]
        color_bar[:,x,2] = c[2]
    # plt.imshow(color_bar)
    # plt.show()
    
    # area_light = area_light_generation("left", alpha = 1.0, beta = 1000, wavelength= 750)
    # plt.imshow(area_light)
    # plt.show()
    light_color_list = [0,70,140,200,240,300,1000]
    # for light_color in light_color_list:
    #     w_length = light_color + 380
    #     area_light = area_light_generation("left", alpha = 1.0, beta = 1000, wavelength= w_length)
    #     area_light_name = "alpha_1.0_beta_400_wavelength_{}.png".format(w_length)
    #     area_light_path = os.path.join('./area_light', area_light_name)
    #     area_light_img = Image.fromarray((area_light*255.0).astype(np.uint8))
    #     area_light_img.save(area_light_path)
    #     plt.imshow(area_light)
    #     plt.show()
    for light_color in range(380,830,50):
        w_length = light_color 
        radians = math.radians(160)
        k = round(math.tan(radians), 2)
        # tube_light = tube_light_generation(135, alpha = 1.0, beta=600, wavelength=w_length)
        # area_light = area_light_generation("left", alpha = 1.0, beta = 800, wavelength= w_length)
        tube_light = tube_light_generation_by_func(k, 350, alpha = 1.0, beta=9, wavelength=520)
        # tube_light_name = "alpha_1.0_beta_400_wavelength_{}.png".format(w_length)
        tube_light_path = os.path.join('./tube_light', 'approach.jpg')
        tube_light_img = Image.fromarray((tube_light*255.0* 0.8).astype(np.uint8))
        tube_light_img.save(tube_light_path)
        plt.imshow(tube_light)
        plt.show()
        img_path_list = os.listdir('./for_show')
        for img_name in img_path_list:
            img_path = os.path.join('./for_show', img_name)
            img = np.asarray(Image.open(img_path).convert('RGB'), dtype = np.float32)
            img_with_light = simple_add(img, tube_light * 255.0, 0.8)
            img_with_light = np.clip(img_with_light, 0.0, 255.0)
            save_light_img = Image.fromarray((img_with_light).astype(np.uint8))
            save_light_img.save('./for_show/green.png')
            plt.imshow(img_with_light/255.0)
            plt.show()
        
    