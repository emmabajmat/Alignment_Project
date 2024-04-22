import cv2
import numpy as np
import scipy.signal
import easyocr as ocr
from skimage.transform import radon


def get_str_from_img( image: np.ndarray) -> str:
    """Get the string from the provided image using easyocr

    Args:
        image (np.ndarray): Image that contains the string we want to find

    Returns:
        str: String from the image
    """

    reader = ocr.Reader(['en'], gpu=True)
    # We use detail = 0 to just get the text, we dont care for the other info
    text_str = reader.readtext(image, detail = 0)
    return text_str

def get_scale_bar_size( image: np.ndarray) -> int:
    """As the scalebar in the SEM images is dynamic get the size of the line with the help of cv2.

    Args:
        image (np.ndarray): Image of the info bar at the bottom of the SEM image

    Returns:
        int: Length of the scale bar in pixels
    """

    scale_img = image[0:125, 200:1250]

    # Standard settings from multiple tutorials.
    edges = cv2.Canny(scale_img, 50, 150, apertureSize = 3)
    lines = cv2.HoughLinesP(
                edges,              # Input edge image
                1,                  # Distance resolution in pixels
                np.pi/180,          # Angle resolution in radians
                threshold=100,      # Min number of votes for valid line
                minLineLength=5,    # Min allowed length of line
                maxLineGap=10       # Max allowed gap between line for joining them
    )

    # So why is there this random -6 you may ask, and that would be a good question.
    # The scale bar in the SEM image has two sort of 'ticks' at the end who mark the
    # beginning and the end of the line. The pixels of these ticks do get counted in
    # the line length estimation and we don't want that. It was found that they are 
    # both 3 pixels wide, so 2 x 3 is... tada! Magic number -6!
    return int(lines[0][0][2] - lines[0][0][0]) - 6


def get_scale( image:np.ndarray, return_all = False, return_pm = False) -> float | tuple | tuple:
    """Get the scale from a given image in pixels per micrometer.

    Args:
        image (np.ndarray): Complete SEM image before cropping or other processing.
        return_all (bool, optional): Choice to return all calculated values. Defaults to False.
        return_pm (bool, optional): Choise to return only scale and the error. Defaults to False.

    Returns:
        float | tuple | tuple: Depending on given preference return the scale, the scale and the error
        or the scale string from the image, the integer from that string and the scale. Defaults to float.
    """
    
    infobar = image[image.shape[0] - 95:image.shape[0], 0:image.shape[1]]

    scale_img = infobar[0:infobar.shape[0], 0:95]
    scale_str = get_str_from_img(scale_img)[0]
    scale_int = int(scale_str)

    scale = scale_int / get_scale_bar_size(infobar)
    pm = (scale_int / (get_scale_bar_size(infobar) - 1)) - scale

    if return_all:
        return scale_str, scale_int, scale
    elif return_pm:
        return scale, pm

    return scale

def crop_image(image:np.ndarray, top_margin:int, bottom_margin:int, left_margin:int, right_margin:int) -> np.ndarray:
    """Crop a given image according to given margins

    Args:
        image (np.ndarray): The image you want to crop
        top_margin (int): How many pixel you want to scrap from the top
        bottom_margin (int): How many pixel you want to scrap from the bottom
        left_margin (int): How many pixel you want to scrap from the left
        right_margin (int): How many pixel you want to scrap from the right

    Returns:
        np.ndarray: The cropped image
    """
    width, height= image.shape
    return image[top_margin:height-bottom_margin, left_margin:width-right_margin]

def get_angle( image:np.ndarray) -> int:
    """Get the angle at which the image is projected using the radon function.

    Args:
        image (np.ndarray): Image you want to get the most prevalent angle from.

    Returns:
        int: Array of indices into the array.
    """
    sinogram = radon(image)

    # In short what happens:
    # Thanks to the radon function we find the most prevalent angle
    # And now we want to get that angle from the radon which is given as an image.
    rms_vals = np.array([np.sqrt(np.mean(np.abs(line) ** 2)) for line in sinogram.transpose()])
    angle = np.argmax(rms_vals)
    return -angle



def normalize_1D_array( array: np.ndarray) -> np.ndarray:
    """Normalize a 1D array

    Args:
        array (np.ndarray): Array you want to normalize.

    Returns:
        np.ndarray: Normalized array
    """
    normalize_values = (array - min(array)) / (max(array) - min(array))
    return normalize_values

def get_intensity_profile( image:np.ndarray, row:int, invert_profile:bool = True) -> np.ndarray:
    """Function to extract the intensity data from a given row

    Args:
        image (np.ndarray): The image you want to analyse.
        row (int): The row you want the profile of.
        invert_profile (bool, optional): Invert profile in case your colors were inverted. Defaults to True.

    Returns:
        np.ndarray: _description_
    """
    if invert_profile:
        return 1-normalize_1D_array(image[row, :])
    else: 
        return normalize_1D_array(image[row, :])
    
def get_intensity_profile_dy( intensity_profile:np.ndarray, scale_factor:int = 5) -> np.ndarray:
    """Get the derivative of the intensity profile.

    Args:
        intensity_profile (np.ndarray): The intensity profile you want the derivative of.
        scale_factor (int, optional): Scale factor to make analysis of the derivative a bit easier. Defaults to 5.

    Returns:
        np.ndarray: Derivative of the intensity profile.
    """
    return np.diff(intensity_profile * scale_factor)

def get_mean_intensity_profile( image:np.ndarray, starting_row: int = 0, ending_row: int = 20) -> np.ndarray:
    """Get the mean intensity profile over a given range of rows.

    Args:
        image (np.ndarray): The image you want to analyse.
        starting_row (int, optional): What is the first row of your range. Defaults to 0.
        ending_row (int, optional): What is the last row of your range. Defaults to 20.

    Returns:
        np.ndarray: The mean intensity profile over a given range
    """
    row_profile_list = np.empty(ending_row - starting_row, dtype=object)
    for row in range(starting_row, ending_row):
        row_profile = get_intensity_profile(image, row)
        row_profile_list[row - starting_row] = row_profile
    mean_data = np.mean(row_profile_list)
    return mean_data

def filter_data( data:np.ndarray, filter_order:int = 3, cutoff_freq:float = 0.1) -> np.ndarray:
    """Simple Buterworth filter the make analysis of data easier when needed.
    # https://stackoverflow.com/questions/35588782/how-to-average-a-signal-to-remove-noise-with-python

    Args:
        data (np.ndarray): The data that you want to filter
        filter_order (int, optional): The order of the filter. Defaults to 3.
        cutoff_freq (float, optional): The critical frequency or frequencies. Defaults to 0.1.

    Returns:
        np.ndarray: Filtered data.
    """
    B, A = scipy.signal.butter(filter_order, cutoff_freq, output='ba')
    return scipy.signal.filtfilt(B, A, data)

def get_leading_edge_positions( intensity_profile_derivative:np.ndarray, treshold:float = 0.1) -> np.ndarray:
    """Find the leading edges by looking up the peaks in the derivative profile. 
    Only peaks above the given treshold are picked up.

    Args:
        intensity_profile_derivative (np.ndarray): Derivative of an intensity profile.
        treshold (float, optional): Treshold above which a value need to be to be picked up by the function. Defaults to 0.1.

    Returns:
        np.ndarray: The leading edge positions in pixel space
    """
    peaks, _ = scipy.signal.find_peaks(intensity_profile_derivative, height=treshold)
    return peaks

def get_trailing_edge_positions( intensity_profile_derivative:np.ndarray, treshold:float = 0.1) -> np.ndarray:
    """Find the trailing edges by looking up the peaks in the inverted derivative profile. 
    Only peaks above the given treshold are picked up.

    Args:
        intensity_profile_derivative (np.ndarray): Derivative of an intensity profile.
        treshold (float, optional): Treshold above which a value need to be to be picked up by the function. Defaults to 0.1.

    Returns:
        np.ndarray: The trailing edge positions in pixel space
    """
    peaks, _ = scipy.signal.find_peaks(-intensity_profile_derivative, height=treshold)
    return peaks

def get_translation( intensity_profile:np.ndarray, scale:float) -> tuple:
    """Get the translation from a given intensity profile.

    Args:
        intensity_profile (np.ndarray): The intensity profile that holds the translation.
        scale (float): The scale determined from the image

    Returns:
        tuple: (A list of translation values, the std of this translation)
    """
    dy = filter_data(get_intensity_profile_dy(intensity_profile))
    
    # Get the positions of the leading and trailing edge of a line
    le = get_leading_edge_positions(dy)
    te = get_trailing_edge_positions(dy)
    
    # Assuming your first line is the first layer you'll always want
    # an uneven amount of lines to do the following calculations. More info
    # on this is in the report
    n = int(len(le)/2)-1
    
    # Calculate the translation for every second line
    T = [(((le[(i + 1) * 2] - te[i * 2 + 1]) - (le[i * 2 + 1] - te[i * 2])) * scale)/2 for i in range(n)]
    std = np.std(T)
    
    return T, std

#TODO: Fix the rotation function as it is not working as intended at this moment.
def get_rotation( image:np.ndarray, scale:float) -> tuple:
    y, _ = image.shape

    translations_top = get_translation(get_mean_intensity_profile(image, 10, 11), scale)
    translations_bottom = get_translation(get_mean_intensity_profile(image, y-1, y), scale)
    
    angle = []
    for i in range(min(len(translations_bottom[0]), len(translations_top[0]))):
        angle.append(np.rad2deg((np.arctan2(translations_top[0][i] - translations_bottom[0][i], (y * scale)))))
    
    angle = np.abs(angle)/2
    
    R = np.mean(angle)
    std = np.std(angle)

    return R, std

def get_line_width_data(image:np.ndarray, mean_intensity_profile:np.ndarray, leading_edges:np.ndarray, trailing_edges:np.ndarray, scale:float, debug:bool = False) -> tuple:
    """Get the line width data for every line in a given image.

    Args:
        image (np.ndarray): Image containing the lines.
        mean_intensity_profile (np.ndarray): The mean intensity profile.
        leading_edges (np.ndarray): The leading edges of the mean_intensity_profile
        trailing_edges (np.ndarray): The trailing edges of the mean_intensity_profile
        scale (float): The scale determined from the image
        debug (bool, optional): Debug the data for quick reference. Defaults to False.

    Returns:
        tuple: (Line widths, std of line widths)
    """
    # Some functions get very sad if you don't pad the data a bit, so give it some space
    margin = 10
    
    line_widths = []
    line_stds = []
    # Every line starts with a peak and end with a valley in the dy spectrum so loop through them
    for peak, valley in zip(leading_edges, trailing_edges):
        # Get the linewidth by substracting the position of the valley (trailing edge)
        # by the position of the peak (leading edge)
        line_width = (valley - peak) * scale
        line_widths.append(line_width)
        
        img_slice = image[:, peak - margin:valley + margin]
        data_slice = mean_intensity_profile[peak - margin:valley + margin]
        y, _ = img_slice.shape
        
        # Calculate the LW error according to the formula in the report
        # It is a bit rough tho due to time, so this could be way better
        LW_differences = np.array([
            np.sqrt(
                sum(np.maximum(data_slice - filter_data(get_intensity_profile(img_slice, row), 1, 0.1)**2, 0))) * scale 
                for row in range(y)
        ])
    
        std = np.std(LW_differences)
        line_stds.append(std)
        
        if debug:
            print(f'Line width: ({line_width:.2f} ± {std:.2f}) µm')
        
    return line_widths, line_stds

def get_single_line_width( image:np.ndarray, mean_intensity_profile:np.ndarray, leading_edges:np.ndarray, trailing_edges:np.ndarray, scale:float, lineIdx:int) -> tuple:
    """Get the line width data for a single line in a given image selected by index.

    Args:
        image (np.ndarray): Image containing the lines.
        mean_intensity_profile (np.ndarray): The mean intensity profile.
        leading_edges (np.ndarray): The leading edges of the mean_intensity_profile
        trailing_edges (np.ndarray): The trailing edges of the mean_intensity_profile
        scale (float): The scale determined from the image
        lineIdx (int): Index of the desired line.

    Returns:
        tuple: (Line width, std of line width)
    """
    lw, std = get_line_width_data(image, mean_intensity_profile, leading_edges, trailing_edges, scale)
    return lw[lineIdx], std[lineIdx]

