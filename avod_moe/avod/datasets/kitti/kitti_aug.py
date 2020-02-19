import copy

import numpy as np
import random
import cv2
AUG_FLIPPING = 'flipping'
AUG_PCA_JITTER = 'pca_jitter'


def flip_image(image):
    """Flips an image horizontally
    """
    flipped_image = np.fliplr(image)
    return flipped_image


def flip_points(points):
    """Flips a list of points (N, 3)
    """
    flipped_points = np.copy(points)
    flipped_points[:, 0] = -points[:, 0]
    return flipped_points


def flip_point_cloud(point_cloud):
    """Flips a point cloud (3, N)
    """
    flipped_point_cloud = np.copy(point_cloud)
    flipped_point_cloud[0] = -point_cloud[0]
    return flipped_point_cloud


def flip_label_in_3d_only(obj_label):
    """Flips only the 3D position of an object label. The 2D bounding box is
    not flipped to save time since it is not used.

    Args:
        obj_label: ObjectLabel

    Returns:
        A flipped object
    """

    flipped_label = copy.deepcopy(obj_label)

    # Flip the rotation
    if obj_label.ry >= 0:
        flipped_label.ry = np.pi - obj_label.ry
    else:
        flipped_label.ry = -np.pi - obj_label.ry

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_t = (-flipped_label.t[0], flipped_label.t[1], flipped_label.t[2])
    flipped_label.t = flipped_t

    return flipped_label


def flip_boxes_3d(boxes_3d, flip_ry=True):
    """Flips boxes_3d

    Args:
        boxes_3d: List of boxes in box_3d format
        flip_ry bool: (optional) if False, rotation is not flipped to save on
            computation (useful for flipping anchors)

    Returns:
        flipped_boxes_3d: Flipped boxes in box_3d format
    """

    flipped_boxes_3d = np.copy(boxes_3d)

    if flip_ry:
        # Flip the rotation
        above_zero = boxes_3d[:, 6] >= 0
        below_zero = np.logical_not(above_zero)
        flipped_boxes_3d[above_zero, 6] = np.pi - boxes_3d[above_zero, 6]
        flipped_boxes_3d[below_zero, 6] = -np.pi - boxes_3d[below_zero, 6]

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_boxes_3d[:, 0] = -boxes_3d[:, 0]

    return flipped_boxes_3d


def flip_ground_plane(ground_plane):
    """Flips the ground plane by negating the x coefficient
        (ax + by + cz + d = 0)

    Args:
        ground_plane: ground plane coefficients

    Returns:
        Flipped ground plane coefficients
    """
    flipped_ground_plane = np.copy(ground_plane)
    flipped_ground_plane[0] = -ground_plane[0]
    return flipped_ground_plane


def flip_stereo_calib_p2(calib_p2, image_shape):
    """Flips the stereo calibration matrix to correct the projection back to
    image space. Flipping the image can be seen as a movement of both the
    camera plane, and the camera itself. To account for this, the instrinsic
    matrix x0 value is flipped with respect to the image width, and the
    extrinsic matrix t1 value is negated.

    Args:
        calib_p2: 3 x 4 stereo camera calibration matrix
        image_shape: (h, w) image shape

    Returns:
        'Flipped' calibration p2 matrix with shape (3, 4)
    """
    flipped_p2 = np.copy(calib_p2)
    flipped_p2[0, 2] = image_shape[1] - calib_p2[0, 2]
    flipped_p2[0, 3] = -calib_p2[0, 3]

    return flipped_p2


def compute_pca(image_set):
    """Calculates and returns PCA of a set of images

    Args:
        image_set: List of images read with cv2.imread in np.uint8 format

    Returns:
        PCA for the set of images
    """

    # Check for valid input
    assert(image_set[0].dtype == np.uint8)

    # Reshape data into single array
    reshaped_data = np.concatenate([image
                                    for pixels in image_set for image in
                                    pixels])

    # Convert to float and normalize the data between [0, 1]
    reshaped_data = (reshaped_data / 255.0).astype(np.float32)

    # Calculate covariance, eigenvalues, and eigenvectors
    # np.cov calculates covariance around the mean, so no need to shift the
    # data
    covariance = np.cov(reshaped_data.T)
    e_vals, e_vecs = np.linalg.eigh(covariance)

    # svd can also be used instead
    # U, S, V = np.linalg.svd(mean_data)

    pca = np.sqrt(e_vals) * e_vecs

    return pca


def add_pca_jitter(img_data, pca):
    """Adds a multiple of the principle components,
    with magnitude from a Gaussian distribution with mean 0 and stdev 0.1


    Args:
        img_data: Original image in read with cv2.imread in np.uint8 format
        pca: PCA calculated with compute_PCA for the image set

    Returns:
        Image with added noise
    """

    # Check for valid input
    assert (img_data.dtype == np.uint8)

    # Make a copy of the image data
    new_img_data = np.copy(img_data).astype(np.float32) / 255.0

    # Calculate noise by multiplying pca with magnitude,
    # then sum horizontally since eigenvectors are in columns
    magnitude = np.random.randn(3) * 0.1
    noise = (pca * magnitude).sum(axis=1)

    # Add the noise to the image, and clip to valid range [0, 1]
    new_img_data = new_img_data + noise
    np.clip(new_img_data, 0.0, 1.0, out=new_img_data)

    # Change back to np.uint8
    new_img_data = (new_img_data * 255).astype(np.uint8)

    return new_img_data


def apply_pca_jitter(image_in):
    """Applies PCA jitter or random noise to a single image

    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    image_in = np.asarray([image_in], dtype=np.uint8)

    pca = compute_pca(image_in)
    image_out = add_pca_jitter(image_in, pca)

    return image_out

def brighten(image_in):

    """Applies changes to brightness in the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    print(image_in.shape)
    # Generates value between 0.0 and 2.0
    coeff = 2* np.random.uniform(0.6,1.5)
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image_in, cv2.COLOR_BGR2HSV)
    # Scale pixel values up or down for channel 1(Lightness)
    h, s, v = cv2.split(image_HLS)
    value = 40
    lim = 255 - value
    v[v > lim] = 255
    # Increase the values of the pixels
    v[v <= lim] += value

    final_hsv = cv2.merge((h, s, v))
    image_HLS[:,:,2] = image_HLS[:,:,2] + 40
    
    # Conversion to RGB
    image_BGR = cv2.cvtColor(final_hsv,cv2.COLOR_HSV2BGR)
    return image_BGR

def blur(image, fog_coeff=-1):
    """Applies blurness to the image
    Args:
        image_in: Image to modify
        fog_coeff : Fog coefficient to apply

    Returns:
        Modified image
    """
    if(fog_coeff!=-1):
        if(fog_coeff<0.0 or fog_coeff>1.0):
            raise Exception(err_fog_coeff)
    if(type(image) is list):
        image_RGB=[]
        image_list=image
        imshape = image[0].shape
        # If the input is an image list, go through each image.
        for img in image_list:
            if fog_coeff==-1:
                fog_coeff_t=np.random.uniform(0.3,1)
            else:
                fog_coeff_t=fog_coeff
            hw = int(imshape[1]//3*fog_coeff_t)
            haze_list = generate_random_blur_coordinates(imshape,hw)
            for haze_points in haze_list:
                img = add_blur(img, haze_points[0],haze_points[1], hw,fog_coeff_t) ## adding all shadow polygons on empty mask, single 255 denotes only red channel
            img = cv2.blur(img ,(hw//10,hw//10))
            image_RGB.append(img)
    else:
        imshape = image.shape
        # Generate a random fog coeff
        if fog_coeff == -1:
            fog_coeff_t = np.random.uniform(0.1,0.4)
        else:
            fog_coeff_t=fog_coeff
        hw = int(imshape[1]//3*fog_coeff_t)
        haze_list = generate_random_blur_coordinates(imshape,hw)
        for haze_points in haze_list:
            # Add blur to the image given the points and the fog coeff
            image = add_blur(image, haze_points[0],haze_points[1], hw, fog_coeff_t)
        image = cv2.blur(image ,(hw//10,hw//10))
        image_RGB = image

    return image_RGB

def generate_random_blur_coordinates(imshape, hw):
    """Applies random coordinates for applying blurness
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    blur_points=[]
    midx= imshape[1]//2-2*hw
    midy= imshape[0]//2-hw
    index=1
    while(midx>-hw or midy>-hw):
        for i in range(hw//10*index):
            # Get random values for x and y coordinates.
            x = np.random.randint(midx,imshape[1]-midx-hw)
            y = np.random.randint(midy,imshape[0]-midy-hw)
            blur_points.append((x,y))
        midx-=3*hw*imshape[1]//sum(imshape)
        midy-=3*hw*imshape[0]//sum(imshape)
        index+=1
    return blur_points

err_fog_coeff="Fog coeff can only be between 0 and 1"
def add_blur(image, x, y, hw, fog_coeff):
    """Applies blur to the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    # Keep copies of the image
    overlay  = image.copy()
    output= image.copy()
    alpha = 0.08*fog_coeff
    rad = hw//2
    point = (x+hw//2, y+hw//2)
    # Apply circles to the image given points
    cv2.circle(overlay, point, int(rad), (255,255,255), -1)
    cv2.addWeighted(overlay, alpha, output, 1 -alpha ,0, output)
    return output

def snow_process(image,snow_coeff):
    """Define the snow process to be applied
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image,cv2.COLOR_BGR2HLS) 
    image_HLS = np.array(image_HLS, dtype = np.float64) 
    brightness_coefficient = 2.5 
    imshape = image.shape
    # increase this for more snow
    snow_point = snow_coeff 
    # scale pixel values up for channel 1(Lightness)
    image_HLS[:,:,1][image_HLS[:,:,1]<snow_point] = image_HLS[:,:,1][image_HLS[:,:,1]<snow_point]*brightness_coefficient 
    # Sets all values above 255 to 255
    image_HLS[:,:,1][image_HLS[:,:,1]>255]  = 255 
    image_HLS = np.array(image_HLS, dtype = np.uint8)
    # Conversion to RGB
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2BGR) 
    return image_RGB

def snow(image, snow_coeff=-1):
    """Add snow to the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    # Generate random snow coeff
    if(snow_coeff!=-1):
        if(snow_coeff<0.0 or snow_coeff>1.0):
            raise Exception(err_snow_coeff)
    else:
        snow_coeff=np.random.uniform(0,1)
    snow_coeff*=255/2
    snow_coeff+=255/3
    if(type(image) is list):
        image_RGB = []
        image_list = image
        for img in image_list:
            output = snow_process(img,snow_coeff)
            image_RGB.append(output) 
    else:
        # Apply the snow process
        output = snow_process(image,snow_coeff)
        image_RGB = output
    
    return image_RGB

def generate_shadow_coordinates(imshape, no_of_shadows, rectangular_roi, shadow_dimension):
    """Generate shadow effects in the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    vertices_list=[]
    x1=rectangular_roi[0]
    y1=rectangular_roi[1]
    x2=rectangular_roi[2]
    y2=rectangular_roi[3]
    for index in range(no_of_shadows):
        vertex=[]
        # Dimensionality of the shadow polygon
        for dimensions in range(shadow_dimension): 
            vertex.append((np.random.randint(x1, x2),np.random.randint(y1, y2)))
        vertices = np.array([vertex], dtype=np.int32) 
        # single shadow vertices 
        vertices_list.append(vertices)
    # List of shadow vertices
    return vertices_list 

def shadow_process(image, no_of_shadows, x1, y1, x2, y2, shadow_dimension):
    """Add shadow to the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image,cv2.COLOR_BGR2HLS) 
    mask = np.zeros_like(image) 
    imshape = image.shape
    # getting list of shadow vertices
    vertices_list= generate_shadow_coordinates(imshape, no_of_shadows,(x1,y1,x2,y2), shadow_dimension) 
    for vertices in vertices_list: 
        # Adding all shadow polygons on empty mask, single 255 denotes only red channel
        cv2.fillPoly(mask, vertices, 255) 
    # if red channel is hot, image's "Lightness" channel's brightness is lowered 
    image_HLS[:,:,1][mask[:,:,0]==255] = image_HLS[:,:,1][mask[:,:,0]==255]*0.5   
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HLS2BGR) ## Conversion to RGB
    return image_RGB

def shadow(image,no_of_shadows=1,rectangular_roi=(-1,-1,-1,-1), shadow_dimension=5):## ROI:(top-left x1,y1, bottom-right x2,y2), shadow_dimension=no. of sides of polygon generated
    """Add shadow to the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
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
    """Function to change the light in the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    image_HLS = image
    # Conversion to HLS
    image_HLS = cv2.cvtColor(image,cv2.COLOR_BGR2HSV) 
    
    image_HLS[:,:,2] = image_HLS[:,:,2]*coeff
    # Conversion to RGB
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HSV2BGR) 
    return image_RGB 

def darken(image, darkness_coeff=-1): 
    """Function to darken the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
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
        # Generate random darkness coeff
        if(darkness_coeff==-1):
             darkness_coeff_t=1- random.uniform(0,0.8)
        else:
            darkness_coeff_t=1- darkness_coeff 
        # Change the light in the image according to the darkness coeff     
        image_RGB= change_light(image,darkness_coeff_t)
    return image_RGB

def flare_source(image, point, radius, src_color):
    """Function to addthe flare source
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
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
    """Function to add sun flare line
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    x=[]
    y=[]
    i=0
    for rand_x in range(0,imshape[1],10):
        rand_y= math.tan(angle)*(rand_x-flare_center[0])+flare_center[1]
        x.append(rand_x)
        y.append(2*flare_center[1]-rand_y)
    return x,y

def add_sun_process(image, no_of_flare_circles, flare_center, src_radius, x, y, src_color):
    """Function to add sun flare
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
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
    """main function to add sun flare to the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
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
    """Function to add fog to the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    if(fog_coeff!=-1):
        if(fog_coeff<0.0 or fog_coeff>1.0):
            raise Exception(err_fog_coeff)
    if(type(image) is list):
        image_RGB = []
        image_list = image
        imshape = image[0].shape

        for img in image_list:
            if fog_coeff == -1:
                fog_coeff_t = random.uniform(0.2,0.5)
            else:
                fog_coeff_t = fog_coeff
            
            hw=int(imshape[1]//3*fog_coeff_t)
            haze_list= generate_random_blur_coordinates(imshape,hw)
            for haze_points in haze_list: 
                # adding all shadow polygons on empty mask, single 255 denotes only red channel
                img = add_blur(img, haze_points[0],haze_points[1], hw,fog_coeff_t) 
            img = cv2.blur(img ,(hw//10,hw//10))
            image_RGB.append(img) 
    else:
        imshape = image.shape
        if fog_coeff == -1:
            fog_coeff_t = random.uniform(0.2,0.5)
        else:
            fog_coeff_t = fog_coeff
        hw = int(imshape[1]//3*fog_coeff_t)
        haze_list = generate_random_blur_coordinates(imshape,hw)
        for haze_points in haze_list: 
            image = add_blur(image, haze_points[0],haze_points[1], hw,fog_coeff_t) 
        image = cv2.blur(image ,(hw//10,hw//10))
        image_RGB = image

    return image_RGB

def rain_process(image,slant,drop_length,drop_color,drop_width,rain_drops):
    """Function to add rain to the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    imshape = image.shape  
    image_t= image.copy()
    for rain_drop in rain_drops:
        cv2.line(image_t,(rain_drop[0],rain_drop[1]),(rain_drop[0]+slant,rain_drop[1]+drop_length),drop_color,drop_width)
    # rainy view are blurry
    image = cv2.blur(image_t,(7,7))
    # rainy days are usually shady  
    brightness_coefficient = 0.7 
    image_HLS = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    # scale pixel values down for channel 1(Lightness)
    image_HLS[:,:,1] = image_HLS[:,:,1] * brightness_coefficient 
    image_RGB = cv2.cvtColor(image_HLS,cv2.COLOR_HSV2BGR)

    return image_RGB


##rain_type='drizzle','heavy','torrential'
def rain(image,slant=-1,drop_length=15,drop_width=1,drop_color=(200,200,200),rain_type='None'): ## (200,200,200) a shade of gray
    """main function to add rain to the image
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    slant_extreme = slant

    if(type(image) is list):
        image_RGB=[]
        image_list=image
        imshape = image[0].shape
        if slant_extreme==-1:
            slant= np.random.randint(-10,10) 
        #generate random slant if no slant value is given
        rain_drops,drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
        for img in image_list:
            output= rain_process(img,slant_extreme,drop_length,drop_color,drop_width,rain_drops)
            image_RGB.append(output)
    else:
        imshape = image.shape
        if slant_extreme==-1:
            slant= np.random.randint(-10,10) 
        # generate random slant if no slant value is given
        rain_drops,drop_length= generate_random_lines(imshape,slant,drop_length,rain_type)
        output= rain_process(image,slant_extreme,drop_length,drop_color,drop_width,rain_drops)
        image_RGB=output

    return image_RGB

def generate_random_lines(imshape,slant,drop_length,rain_type):

    """Function to generate lines ti imitate rain 
    Args:
        image_in: Image to modify

    Returns:
        Modified image
    """
    drops = []
    area = imshape[0]*imshape[1]
    # Increase number of drops for heavy rain 
    no_of_drops = area//600

    if rain_type.lower() == 'drizzle':
        no_of_drops = area//770
        drop_length = 10
    elif rain_type.lower() == 'heavy':
        drop_length = 30
    elif rain_type.lower() == 'torrential':
        no_of_drops = area//500
        drop_length = 60

    for i in range(no_of_drops): 
        if slant<0:
            x = np.random.randint(slant,imshape[1])
        else:
            x = np.random.randint(0,imshape[1]-slant)
        y = np.random.randint(0,imshape[0]-drop_length)
        drops.append((x,y))
    return drops,drop_length
