import cv2
import glob
import numpy as np

#################### X-Y CONVENTIONS #########################
# 0,0  X  > > > > >
#
#  Y
#
#  v  This is the image. Y increases downwards, X increases rightwards
#  v  Please return bounding boxes as ((xmin, ymin), (xmax, ymax))
#  v
#  v
#  v
###############################################################
def image_print(img):
    """
    Helper function to print out images, for debugging. Pass them in as a list.
    Press any key to continue.
    """
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def erosion_filter(box_size = 3, iterations = 1):
    erosion_kernel = np.ones((box_size, box_size), np.uint8)
    def erosion_func(input_image):
        return cv2.erode(input_image, erosion_kernel, iterations = iterations)
    
    return erosion_func

def dilation_filter(box_size = 3, iterations = 1):
    dilation_kernel = np.ones((box_size, box_size), np.uint8)
    def dilation_func(input_image):
        return cv2.dilate(input_image, dilation_kernel, iterations = iterations)
    
    return dilation_func

def create_filter_cascade(filter_list):
    def filter_cascade(image):
        for filt in filter_list:
            image = filt(image)

        return image
    
    return filter_cascade

def cd_color_segmentation(img, hsv_ranges, margins = (5,5)):
    """
    Implement standard color detection using color segmentation algorithm
    Input:
        img: np.3darray; the input image BGR.
    Return:
        bbox: ((x1, y1), (x2, y2)); the bounding box of the most prominent area of color, unit in px
            (x1, y1) is the top left of the bbox and (x2, y2) is the bottom right of the bbox
        bbox_size: float; the size of the bbox, unit in px

        None if no bbox is found
    """
    ### Program ###
    hsv_input_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    
    final_color_mask = np.zeros(hsv_input_img.shape[:2], dtype="uint8")
    for hsv_range in hsv_ranges:
        low_hsv, high_hsv = np.array([hsv_convert_to_cv2(hsv) for hsv in hsv_range])
        color_mask = cv2.inRange(hsv_input_img, low_hsv, high_hsv)
        final_color_mask = cv2.bitwise_or(final_color_mask, color_mask)

    filt_cascade = create_filter_cascade([
        dilation_filter(box_size = 5, iterations = 1),
        erosion_filter(box_size = 5, iterations = 1),
        dilation_filter(box_size = 5, iterations = 2)
    ])
    filtered_mask = filt_cascade(final_color_mask)
 
    contours, _ = cv2.findContours(filtered_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    contoured_mask = filtered_mask.copy()
    cv2.drawContours(contoured_mask, contours, -1, color = (0,0,255), thickness = 1)

    x_margin = margins[0]
    y_margin = margins[1]

    img_height, img_width, _= img.shape
    bounding_box = None
    box_size = None
    if len(contours) != 0:
        biggest_ctr = max(contours, key = cv2.contourArea)
        bx, by, bw, bh = cv2.boundingRect(biggest_ctr)

        left, right = max(bx - x_margin, 0), min(bx + bw + x_margin, img_width-1)
        top, bottom = max(by - y_margin, 0), min(by+bh+y_margin, img_height-1)
        
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5))
        region_of_interest_mask = filtered_mask[top:bottom, left:right].copy()
        
        region_of_interest_mask = cv2.morphologyEx(region_of_interest_mask, cv2.MORPH_CLOSE, kernel)

        contours_in_roi, _ = cv2.findContours(region_of_interest_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        
        for ctr in contours_in_roi:
            for point in ctr:
                point[0] = (point[0][0] + left, point[0][1] + top)

        x,y,w,h = cv2.boundingRect(np.vstack(contours_in_roi))
        bounding_box = ((x,y), (x+w, y+h))
        box_size = w * h

    return bounding_box, box_size

def find_most_prominent_color(img, color_dict, img_to_draw = None, draw_colors = None):
    """
    For each color in the given color_dict, the image is segmented into pixels belonging to that color and
    pixels that don't. Then, a bounding box is fitted to the biggest area of that color. Finally, whichever
    color has the biggest bounding box is returned.

    Args:
        img (2D Array): The input image
        color_dict (dictionary): 
            Keys (string): a color (eg. red, yellow, blue)
            Values (list of lists): the hsv_ranges corresponding to that color
        img_to_draw (2D Array): An image to draw the bounding boxes on if desired
        draw_colors (dictionary):
            Keys (string): the same colors as the color_dict
            Values (1D Array): The (b,g,r) color to draw the bounding box of said color
    
    Returns:
        most_prominent_color: string; self-explanatory

        None if none of the colors return a bounding box in the image
    """

    most_prominent_color = None
    max_bbox_size = 0
    for color, hsv_ranges in color_dict.items():
        color_bbox, bbox_size = cd_color_segmentation(img, hsv_ranges)

        if color_bbox is not None:
            if bbox_size > max_bbox_size:
                most_prominent_color = color
                max_bbox_size = bbox_size

            if img_to_draw is not None:
                cv2.rectangle(img_to_draw, color_bbox[0], color_bbox[1], draw_colors[color], 2)
    
    return most_prominent_color


def hsv_convert_to_cv2(hsv_input):
    hue, sat, val = hsv_input
    return (hue / 2, sat * 2.55, val * 2.55)

if __name__ == "__main__":
    red_hsv_range_1 = [[0, 70, 75], [10, 90, 100]]
    red_hsv_range_2 = [[350, 70, 75], [360, 90, 100]]

    green_hsv_range = [[140, 40, 75],[180, 100, 100]]

    yellow_hsv_range = [[48, 40, 70],[60, 90, 100]]

    color_dict = {
        "red": [red_hsv_range_1, red_hsv_range_2],
        "green": [green_hsv_range],
        "yellow": [yellow_hsv_range]
    }

    colors_to_draw = {
        "red": (0,0,255),
        "green": (0,255,0),
        "yellow": (0,255,255)
    }
    
    path = "traffic_light_imgs/*jpg"
    correctly_identified = 0
    img_files = glob.glob(path)
    for file in img_files:
        img = cv2.imread(file)
        color_ret = find_most_prominent_color(img, color_dict)
        if color_ret in file:
            print(f"CORRECT: {file} identified as {color_ret}")
            correctly_identified +=1
        else:
            print(f"FALSE: {file} identified as {color_ret}")

    print(f"{correctly_identified} / {len(img_files)} traffic lights correctly identified!")
    pass