import skimage
from glob import glob
import json
import os
import numpy as np
from skimage.draw import polygon
from skimage.measure import regionprops, perimeter
from skimage.feature import graycomatrix, graycoprops
from multiprocessing import Pool

def features_extraction(json_dict, original_image):
    """
    Feature extraction for each cell of an image using the inference results from HoVer-UNet
    Adding new color and morphological-based features to the json files
    """
    centroids = json_dict['centroid']
    contours = json_dict['contour']
    contours = [np.array(contour) for contour in contours]
    
    total_features = []
    for contour, centroid in zip(contours, centroids):
        #Creation of binary roi for shape features, and rgb roi for color based features
        min_row, min_col = np.min(contour, axis=0)
        max_row, max_col = np.max(contour, axis=0)
        h, w = original_image.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)
        c, r = contour[:,0], contour[:,1]
        rr, cc = polygon(r, c)
        mask[rr, cc] = 1
        rgb_roi = original_image[min_col:max_col, min_row:max_row]
        cropped_mask = mask[min_col:max_col, min_row:max_row]
        
        #Shape-based feature extraction
        props = regionprops(cropped_mask)[0]
        eccentricity = props.eccentricity
        solidity = props.solidity
        area = props.area
        orientation = props.orientation
        cell_perimeter = perimeter(cropped_mask, neighborhood=8)
        major_axis_length = props.major_axis_length
        minor_axis_length = props.minor_axis_length
        
        #Color-based feature extraction
        gray = skimage.color.rgb2gray(rgb_roi)
        masked_img = (gray * cropped_mask * 255).astype(np.uint8)
        glcm = graycomatrix(masked_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        contrast = graycoprops(glcm, 'contrast')[0, 0]
        
        features = [eccentricity, solidity, area, orientation, cell_perimeter, contrast, major_axis_length, minor_axis_length]
        total_features.append(features)
        
    total_features = np.array(total_features)
        
    return total_features


def add_features_to_json(path):
    img_name = os.path.splitext(os.path.basename(path))[0] + '.png'
    img_path = f'images/{img_name}'
    img = skimage.io.imread(img_path)
    with open(path) as file:
        json_dict = json.load(file)
        
    features = features_extraction(json_dict, img)
    
    del json_dict['contour']
    
    json_dict['eccentricity'] = features[:,0].tolist()
    json_dict['solidity'] = features[:,1].tolist()
    json_dict['area'] = features[:,2].tolist()
    json_dict['orientation'] = features[:,3].tolist()
    json_dict['cell_perimeter'] = features[:,4].tolist()
    json_dict['contrast'] = features[:,5].tolist()
    json_dict['major_axis_length'] = features[:,6].tolist()
    json_dict['minor_axis_length'] = features[:,7].tolist()
    
    dst = 'json_with_features/' + os.path.basename(path)
    with open(dst, 'w') as file:
        json.dump(json_dict, file)
    
    
if __name__ == '__main__': 
    paths = glob('json/*.json')
    cpu_cores = os.cpu_count()
    print(f'Maximum number of processes: {cpu_cores}')
    print('Adding new features to the json files...')
    with Pool(cpu_cores) as p:
        p.map(add_features_to_json, paths)
