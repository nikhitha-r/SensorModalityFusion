import copy

import numpy as np
import cv2
import sys
import random
from random import randint
from random import sample
import math
AUG_BRIGHTNESS = "brighten"
AUG_BLUR = "blur"
AUG_SNOW = "snow"


def brighten(image_in):

    """Applies changes to brightness in the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    print(image_in.shape)
    #image_in = np.asarray([image_in], dtype=np.uint8)
    # Generates value between 0.0 and 2.0
    coeff = 2* np.random.uniform(0.6,1.5)
    print(image_in.shape)
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)
    #image_HLS = np.array(image_HLS, dtype = np.float64)
    # Scale pixel values up or down for channel 1(Lightness)
    #image_HLS[:,:,1] = image_HLS[:,:,1]*coeff
    h, s, v = cv2.split(image_HLS)
    value = 40
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    image_HLS[:,:,2] = image_HLS[:,:,2] + 40
    """
    if(coeff>1):
        # Sets all values above 255 to 255
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255
    else:
        image_HLS[:,:,1][image_HLS[:,:,1]<0]=0
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    """
    # Conversion to RGB
    image_BGR = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
    #image_RGB = cv2.cvtColor(final_hsv,cv2.COLOR_BGR2RGB)
    return image_BGR



def blur(image, fog_coeff=-1):
    if(fog_coeff!=-1):
        if(fog_coeff<0.0 or fog_coeff>1.0):
            raise Exception(err_fog_coeff)
    if(type(image) is list):
        image_RGB=[]
        image_list=image
        imshape = image[0].shape

        for img in image_list:
            if fog_coeff==-1:
                fog_coeff_t=np.random.uniform(0.3,1)
            else:
                fog_coeff_t=fog_coeff
            hw=int(imshape[1]//3*fog_coeff_t)
            haze_list= generate_random_blur_coordinates(imshape,hw)
            for haze_points in haze_list:
                img= add_blur(img, haze_points[0],haze_points[1], hw,fog_coeff_t) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
            img = cv2.blur(img ,(hw//10,hw//10))
            image_RGB.append(img)
    else:
        imshape = image.shape
        if fog_coeff==-1:
            fog_coeff_t=np.random.uniform(0.1,0.4)
        else:
            fog_coeff_t=fog_coeff
        hw=int(imshape[1]//3*fog_coeff_t)
        haze_list= generate_random_blur_coordinates(imshape,hw)
        for haze_points in haze_list:
            image= add_blur(image, haze_points[0],haze_points[1], hw,fog_coeff_t)
        image = cv2.blur(image ,(hw//10,hw//10))
        image_RGB = image

    return image_RGB
def generate_random_blur_coordinates(imshape,hw):
    blur_points=[]
    midx= imshape[1]//2-2*hw
    midy= imshape[0]//2-hw
    index=1
    while(midx>-hw or midy>-hw):
        for i in range(hw//10*index):
            x= np.random.randint(midx,imshape[1]-midx-hw)
            y= np.random.randint(midy,imshape[0]-midy-hw)
            blur_points.append((x,y))
        midx-=3*hw*imshape[1]//sum(imshape)
        midy-=3*hw*imshape[0]//sum(imshape)
        index+=1
    return blur_points

err_fog_coeff="Fog coeff can only be between 0 and 1"
def add_blur(image, x,y,hw,fog_coeff):
    overlay= image.copy()
    output= image.copy()
    alpha= 0.08*fog_coeff
    rad= hw//2
    point=(x+hw//2, y+hw//2)
    cv2.circle(overlay,point, int(rad), (255,255,255), -1)
    cv2.addWeighted(overlay, alpha, output, 1 -alpha ,0, output)
    return output

def snow_process(image,snow_coeff):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_BGR2HLS) ## Conversion to HLS
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    brightness_coefficient = 2.5 
    imshape = image.shape
    snow_point=snow_coeff ## increase this for more snow
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient ## scale pixel values up for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2BGR) ## Conversion to RGB
    return image_RGB

def snow(image, snow_coeff=-1):
    if(snow_coeff!=-1):
        if(snow_coeff<0.0 or snow_coeff>1.0):
            raise Exception(err_snow_coeff)
    else:
        snow_coeff=np.random.uniform(0,1)
    snow_coeff*=255/2
    snow_coeff+=255/3
    if(type(image) is list):
        image_RGB=[]
        image_list=image
        for img in image_list:
            output= snow_process(img,snow_coeff)
            image_RGB.append(output) 
    else:
        output= snow_process(image,snow_coeff)
        image_RGB=output
    
    return image_RGB

def generate_shadow_coordinates(imshape, no_of_shadows, rectangular_roi, shadow_dimension):
    vertices_list=[]
    x1=rectangular_roi[0]
    y1=rectangular_roi[1]
    x2=rectangular_roi[2]
    y2=rectangular_roi[3]
    for index in range(no_of_shadows):
        vertex=[]
        for dimensions in range(shadow_dimension): ## Dimensionality of the shadow polygon
            vertex.append((np.random.randint(x1, x2),np.random.randint(y1, y2)))
        vertices = np.array([vertex], dtype=np.int32) ## single shadow vertices 
        vertices_list.append(vertices)
    return vertices_list ## List of shadow vertices

def shadow_process(image,no_of_shadows,x1,y1,x2,y2, shadow_dimension):
    image_HLS = cv2.cvtColor(image,cv2.COLOR_BGR2HLS) ## Conversion to HLS
    mask = np.zeros_like(image) 
    imshape = image.shape
    vertices_list= generate_shadow_coordinates(imshape, no_of_shadows,(x1,y1,x2,y2), shadow_dimension) #3 getting list of shadow vertices
    for vertices in vertices_list: 
        cv2.fillPoly(mask, vertices, 255) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
    image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5   ## if red channel is hot, image's "Lightness" channel's brightness is lowered 
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2BGR) ## Conversion to RGB
    return image_RGB

def shadow(image,no_of_shadows=1,rectangular_roi=(-1,-1,-1,-1), shadow_dimension=5):## ROI:(top-left x1,y1, bottom-right x2,y2), shadow_dimension=no. of sides of polygon generated
    if not(type(no_of_shadows) is int and no_of_shadows>=1 and no_of_shadows<=10):
        raise Exception(err_shadow_count)
    if not(type(shadow_dimension) is int and shadow_dimension>=3 and shadow_dimension<=10):
        raise Exception(err_shadow_dimension)
    if type(rectangular_roi) is tuple and len(rectangular_roi)==4:
        x1=rectangular_roi[0]
        y1=rectangular_roi[1]
        x2=rectangular_roi[2]
        y2=rectangular_roi[3]
    else:
        raise Exception(err_invalid_rectangular_roi)
    if rectangular_roi==(-1,-1,-1,-1):
        x1=0
        
        if(isinstance(image, np.ndarray)):
            y1=image.shape[0]//2
            x2=image.shape[1]
            y2=image.shape[0]
        else:
            y1=image[0].shape[0]//2
            x2=image[0].shape[1]
            y2=image[0].shape[0]

    elif x1==-1 or y1==-1 or x2==-1 or y2==-1 or x2<=x1 or y2<=y1:
        raise Exception(err_invalid_rectangular_roi)
    if(type(image) is list):
        image_RGB=[]
        image_list=image
        for img in image_list:
            output=shadow_process(img,no_of_shadows,x1,y1,x2,y2, shadow_dimension)
            image_RGB.append(output)
    else:
        output=shadow_process(image,no_of_shadows,x1,y1,x2,y2, shadow_dimension)
        image_RGB = output
    return image_RGB

def change_light(image, coeff):
    image_HLS = image
    image_HLS = cv2.cvtColor(image,cv2.COLOR_BGR2HSV) ## Conversion to HLS
    """
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    image_HLS[:,:,1] = image_HLS[:,:,1]*coeff ## scale pixel values up or down for channel 1(Lightness)
    if(coeff>1):
        image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 ##Sets all values above 255 to 255
    else:
        image_HLS[:,:,1][image_HLS[:,:,1]<0]=0
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    """
    
    """
    lim = 255 - value
    v[v > lim] = 255
    v[v <= lim] -= value

    final_hsv = cv2.merge((h, s, v))
    """
    image_HLS[:,:,2] = image_HLS[:,:,2]*coeff
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HSV2BGR) ## Conversion to RGB
    #image_RGB = image_HLS
    return image_RGB 

def darken(image, darkness_coeff=-1): ##function to darken the image
    if(darkness_coeff!=-1):
        if(darkness_coeff<0.0 or darkness_coeff>1.0):
            raise Exception(err_darkness_coeff) 

    if(type(image) is list):
        image_RGB=[]
        image_list=image
        for img in image_list:
            if(darkness_coeff==-1):
                darkness_coeff_t=1- random.uniform(0,0.6)
            else:
                darkness_coeff_t=1- darkness_coeff            
            image_RGB.append(change_light(img,darkness_coeff_t))
    else:
        if(darkness_coeff==-1):
             darkness_coeff_t=1- random.uniform(0,0.8)
        else:
            darkness_coeff_t=1- darkness_coeff  
        image_RGB= change_light(image,darkness_coeff_t)
    return image_RGB

def flare_source(image, point,radius,src_color):
    overlay= image.copy()
    output= image.copy()
    num_times=radius//10
    alpha= np.linspace(0.0,1,num= num_times)
    rad= np.linspace(1,radius, num=num_times)
    for i in range(num_times):
        cv2.circle(overlay,point, int(rad[i]), src_color, -1)
        alp=alpha[num_times-i-1]*alpha[num_times-i-1]*alpha[num_times-i-1]
        cv2.addWeighted(overlay, alp, output, 1 -alp ,0, output)
    return output

def add_sun_flare_line(flare_center,angle,imshape):
    x=[]
    y=[]
    i=0
    for rand_x in range(0,imshape[1],10):
        rand_y= math.tan(angle)*(rand_x-flare_center[0])+flare_center[1]
        x.append(rand_x)
        y.append(2*flare_center[1]-rand_y)
    return x,y

def add_sun_process(image, no_of_flare_circles,flare_center,src_radius,x,y,src_color):
    overlay= image.copy()
    output= image.copy()
    imshape=image.shape
    for i in range(no_of_flare_circles):
        alpha=random.uniform(0.05,0.2)
        r=random.randint(0, len(x)-1)
        rad=random.randint(1, imshape[0]//100-2)
        cv2.circle(overlay,(int(x[r]),int(y[r])), rad*rad*rad, (random.randint(max(src_color[0]-50,0), src_color[0]),random.randint(max(src_color[1]-50,0), src_color[1]),random.randint(max(src_color[2]-50,0), src_color[2])), -1)
        cv2.addWeighted(overlay, alpha, output, 1 - alpha,0, output)                      
    output= flare_source(output,(int(flare_center[0]),int(flare_center[1])),src_radius,src_color)
    return output

def sun_flare(image,flare_center=-1, angle=-1, no_of_flare_circles=8,src_radius=400, src_color=(255,255,255)):
    if(angle!=-1):
        angle=angle%(2*math.pi)
    if not(no_of_flare_circles>=0 and no_of_flare_circles<=20):
        raise Exception(err_flare_circle_count)
    if(type(image) is list):
        image_RGB=[]
        image_list=image
        imshape=image_list[0].shape
        for img in image_list: 
            if(angle==-1):
                angle_t=random.uniform(0,2*math.pi)
                if angle_t==math.pi/2:
                    angle_t=0
            else:
                angle_t=angle
            if flare_center==-1:   
                flare_center_t=(random.randint(0,imshape[1]),random.randint(0,imshape[0]//2))
            else:
                flare_center_t=flare_center
            x,y= add_sun_flare_line(flare_center_t,angle_t,imshape)
            output= add_sun_process(img, no_of_flare_circles,flare_center_t,src_radius,x,y,src_color)
            image_RGB.append(output)
    else:
        imshape=image.shape
        if(angle==-1):
            angle_t=random.uniform(0,2*math.pi)
            if angle_t==math.pi/2:
                angle_t=0
        else:
            angle_t=angle
        if flare_center==-1:
            flare_center_t=(random.randint(0,imshape[1]),random.randint(0,imshape[0]//2))
        else:
            flare_center_t=flare_center
        x,y= add_sun_flare_line(flare_center_t,angle_t,imshape)
        output= add_sun_process(image, no_of_flare_circles,flare_center_t,src_radius,x,y,src_color)
        image_RGB = output
    return image_RGB



def fog(image, fog_coeff=-1):

    if(fog_coeff!=-1):
        if(fog_coeff<0.0 or fog_coeff>1.0):
            raise Exception(err_fog_coeff)
    if(type(image) is list):
        image_RGB=[]
        image_list=image
        imshape = image[0].shape

        for img in image_list:
            if fog_coeff==-1:
                fog_coeff_t=random.uniform(0.2,0.5)
            else:
                fog_coeff_t=fog_coeff
            
            hw=int(imshape[1]//3*fog_coeff_t)
            haze_list= generate_random_blur_coordinates(imshape,hw)
            for haze_points in haze_list: 
                img= add_blur(img, haze_points[0],haze_points[1], hw,fog_coeff_t) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
            img = cv2.blur(img ,(hw//10,hw//10))
            image_RGB.append(img) 
    else:
        imshape = image.shape
        if fog_coeff==-1:
            fog_coeff_t=random.uniform(0.2,0.5)
        else:
            fog_coeff_t=fog_coeff
        print(fog_coeff_t)
        hw=int(imshape[1]//3*fog_coeff_t)
        haze_list= generate_random_blur_coordinates(imshape,hw)
        for haze_points in haze_list: 
            image= add_blur(image, haze_points[0],haze_points[1], hw,fog_coeff_t) 
        image = cv2.blur(image ,(hw//10,hw//10))
        image_RGB = image

    return image_RGB

def apply_motion_blur(image,count):
    image_t=image.copy()
    imshape=image_t.shape
    size=15
    kernel_motion_blur = np.zeros((size, size))
    kernel_motion_blur[int((size-1)/2), :] = np.ones(size)
    kernel_motion_blur = kernel_motion_blur / size
    i= imshape[1]*3//4 - 10*count
    try:
        while(i<=imshape[1]):
            image_t[:,i:,:] = cv2.filter2D(image_t[:,i:,:], -1, kernel_motion_blur)
            image_t[:,:imshape[1]-i,:] = cv2.filter2D(image_t[:,:imshape[1]-i,:], -1, kernel_motion_blur)
            i+=imshape[1]//25-count
            count+=1
    except:
        image_t = blur(image)
    
    image_RGB=image_t
    return image_RGB

def speed(image, speed_coeff=-1):
    if(speed_coeff !=-1):
        if(speed_coeff<0.0 or speed_coeff>1.0):
            raise Exception(err_speed_coeff)
    if(type(image) is list):
        image_RGB=[]
        image_list=image
        for img in image_list:
            if(speed_coeff==-1):
                count_t=int(15*random.uniform(0,1))
            else:
                count_t=int(15*speed_coeff)
            img=apply_motion_blur(img,count_t)
            image_RGB.append(img)
    else:
        if(speed_coeff==-1):
            count_t=int(15*random.uniform(0,1))
        else:
            count_t=int(15*speed_coeff)
        print(count_t)
        image_RGB= apply_motion_blur(image,count_t)


    return image_RGB

def rain_process(image,slant,drop_length,drop_color,drop_width,rain_drops):
    imshape = image.shape  
    image_t= image.copy()
    for rain_drop in rain_drops:
        cv2.line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    image= cv2.blur(image_t,(7,7)) ## rainy view are blurry
    brightness_coefficient = 0.7 ## rainy days are usually shady 
    image_HLS = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    image_HLS[:,:,1] = image_HLS[:,:,1]*brightness_coefficient ## scale pixel values down for channel 1(Lightness)
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HSV2BGR)

    return image_RGB


##rain_type='drizzle','heavy','torrential'
def rain(image,slant=-1,drop_length=15,drop_width=1,drop_color=(200,200,200),rain_type='None'): ## (200,200,200) a shade of gray
    slant_extreme=slant

    if(type(image) is list):
        image_RGB=[]
        image_list=image
        imshape = image[0].shape
        if slant_extreme==-1:
            slant= np.random.randint(-10,10) ##generate random slant if no slant value is given
        rain_drops,drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
        for img in image_list:
            output= rain_process(img,slant_extreme,drop_length,drop_color,drop_width,rain_drops)
            image_RGB.append(output)
    else:
        imshape = image.shape
        if slant_extreme==-1:
            slant= np.random.randint(-10,10) ##generate random slant if no slant value is given
        rain_drops,drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
        output= rain_process(image,slant_extreme,drop_length,drop_color,drop_width,rain_drops)
        image_RGB=output

    return image_RGB

def generate_random_lines(imshape,slant,drop_length,rain_type):
    drops=[]
    area=imshape[0]*imshape[1]
    no_of_drops=area//600

    if rain_type.lower()=='drizzle':
        no_of_drops=area//770
        drop_length=10
    elif rain_type.lower()=='heavy':
        drop_length=30
    elif rain_type.lower()=='torrential':
        no_of_drops=area//500
        drop_length=60

    for i in range(no_of_drops): ## If You want heavy rain, try increasing this
        if slant<0:
            x= np.random.randint(slant,imshape[1])
        else:
            x= np.random.randint(0,imshape[1]-slant)
        y= np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops,drop_length

aug_list = ["brighten", "blur", "snow", "darken", "fog", "shadow", "sun_flare", "rain"]
image_dir = "/storage/remote/atcremers61/w0017/KITTI/PointRCNN/KITTI/object/training/image_2/"
aug_dir = "/usr/prakt/w0017/Augmented_Data/images_aug/"

with open("../split/val.txt", "r") as valFile:
    num_set = sum(1 for line in valFile)
    aug_num = int(0.4 * num_set)
    print("testing", num_set, aug_num)
    #lines = valFile.read().splitlines()
with open("../split/val.txt", "r") as valFile:
    with open("../split/aug.txt", "w") as augFile:
        num_list = []
        for i in valFile:
            num_list.append(i)
        image_id = "7480"

        
        for i in range(0, aug_num):
            lineSelect = random.randint(0, num_set-1)
            print(lineSelect)
            random_line = num_list[lineSelect].strip()
            augFile.write(random_line)
            augFile.write("\n")
            random_aug = random.choice(aug_list)
            print(random_line, random_aug)
            print(image_dir + random_line + ".png")
            orig_image = cv2.imread(image_dir + random_line.strip() + ".png", 1)
           
            image_mod = eval(random_aug)(orig_image)

            image_id = int(image_id) + 1

            cv2.imwrite(aug_dir + "00" + str(image_id) + ".png", image_mod)


            
        



