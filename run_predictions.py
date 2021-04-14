import os
import numpy as np
import json
from PIL import Image
from numba import *

# import sys
# import numpy
# numpy.set_printoptions(threshold=sys.maxsize)

@njit
def rgb_to_hsv(r, g, b):
    r0 = r / 255.0
    g0 = g / 255.0
    b0 = b / 255.0

    c_max = max(r0, g0, b0)
    c_min = min(r0, g0, b0)
    delta = c_max - c_min
    
    if delta == 0:
        h = 0
    elif c_max == r0:
        h = (60 * ((g0 - b0) / delta + 6)) % 360
    elif c_max == g0:
        h = (60 * ((b0 - r0) / delta + 2)) % 360
    elif c_max == b0:
        h = (60 * ((r0 - g0) / delta + 4)) % 360

    if c_max == 0:
        s = 0
    else:
        s = (delta / c_max) * 100

    v = c_max * 100

    return h, s, v

@njit
def find_red_black(I):
    '''
    Takes a numpy array <I> and returns two lists <red_coords> and <black_coords>,
    and a 2D array of shape np.shape(I) with 
        black coordinates having entry 0,
        red coordinates having entry 1, and 
        all other colors -1. 
    <red_coords> contains all coordinates in I which are approx. red, and 
    <black_coords> contains all coordinates in I which are approx. black.
    '''
    # Find the dimensions of the image I and set threshold
    (n_rows, n_cols, n_channels) = I.shape
    new_img = np.zeros((n_rows, n_cols)) 
    red_coords = set()
    black_coords = set()

    for row in range(n_rows):
        for col in range(n_cols):
            r, g, b = I[row, col, :]
            h, s, v = rgb_to_hsv(r, g, b)

            if (h < 20 or h > 330) and v > 60 and s > 60:
                red_coords.add((row, col))
                new_img[row, col] = 1
            elif v < 35:
                black_coords.add((row, col))
                new_img[row, col] = 0
            else:
                new_img[row, col] = -1

    return red_coords, black_coords, new_img

@njit
def normalize(v):
    #norm = np.linalg.norm(v)
    norm = 0
    n_rows, n_cols = v.shape
    for i in range(n_rows):
        for j in range(n_cols):
            norm += v[i][j] ** 2

    norm = np.sqrt(norm)

    if norm != 0:
        return v / norm
    else:
        return v


def near_red(r, c, new_img, radius):
    '''
    Check if this coordinate is near red or not.
    '''
    n_rows, n_cols = new_img.shape

    min_r = max(0, r - radius)
    max_r = min(r + radius, n_rows)
    min_c = max(0, c - radius)
    max_c = min(c + radius, n_cols) 

    return 1 in new_img[min_r:max_r, min_c:max_c]


def compute_convolution(I, T, stride=None, padding=True):
    '''
    This function takes an image <I> and a template <T> (both numpy arrays) 
    and returns a heatmap where each grid represents the output produced by 
    convolution at each location. You can add optional parameters (e.g. stride, 
    window_size, padding) to create additional functionality. 
    '''
    n_rows,n_cols,n_channels = I.shape

    '''
    BEGIN YOUR CODE
    '''

    # Find red and black pixels to reduce search time
    red_coords, black_coords, new_img = find_red_black(I)

    heatmap = np.zeros((n_rows, n_cols))
    t_rows, t_cols, t_channels = T.shape


    if padding:
        # Adding padding to ensure heatmap is same size as I
        bottom_padding = int((t_rows - 1) / 2)
        top_padding = (t_rows - 1) - bottom_padding
        left_padding = int((t_cols - 1) / 2)
        right_padding = (t_cols - 1) - left_padding

        I_new = np.zeros((n_rows + t_rows - 1, n_cols + t_cols - 1, n_channels))
        I_new[left_padding:left_padding + n_rows, top_padding:top_padding + n_cols, :] = np.copy(I)

        n_rows += t_rows - 1
        n_cols += t_cols - 1

    for row in range(int(0.7 * (n_rows - t_rows + 1))):
        for col in range(n_cols - t_cols + 1):
            if near_red(row, col, new_img, int(t_rows / 3)):
                channel_prods = []
                for ch in range(n_channels):
                    panel = I_new[row:row + t_rows, col:col + t_cols, ch]
                    panel_vec = normalize(panel).flatten()
                    temp_vec = normalize(T[:, :, ch]).flatten()
                    channel_prods.append(np.dot(panel_vec, temp_vec))

                p_sum = 0
                for prod in channel_prods:
                    p_sum += prod
                
                heatmap[row][col] = p_sum / n_channels
            else:
                continue
    '''
    END YOUR CODE
    '''

    return heatmap

@njit
def label_grid(grid):
    '''
    Helper function which finds clusters of 1s in a 0-1 grid. Returns a list
    of lists which contain tuples of (row, col) locations. Each list in the list
    corresponds to a cluster of points.
    '''
    n_rows, n_cols = grid.shape
    clusters = []
    visited = set()
    queue = []

    for row in range(n_rows):
        for col in range(n_cols):
            curr_cluster = []
            if (row, col) not in visited and grid[row][col] == 1:
                curr_cluster.append((row, col))
                visited.add((row, col))
                queue.append((row, col))

            while queue:
                last_elem = queue.pop(0)
                for i in [-1, 1]:
                    for j in [-1, 1]:
                        ne_row = row + i
                        ne_col = col + j
                        if (ne_row, ne_col) not in visited and grid[ne_row][ne_col] == 1:
                            curr_cluster.append((ne_row, ne_col))
                            visited.add((ne_row, ne_col))
                            queue.append((ne_row, ne_col))

            if curr_cluster:
                clusters.append(curr_cluster)

    return clusters


def predict_boxes(heatmap, T):
    '''
    This function takes heatmap and returns the bounding boxes and associated
    confidence scores.
    '''

    output = []

    '''
    BEGIN YOUR CODE
    '''
    
    '''
    As an example, here's code that generates between 1 and 5 random boxes
    of fixed size and returns the results in the proper format.
    '''

    n_rows, n_cols = np.shape(heatmap)
    threshold = 0.9
    t_rows, t_cols, t_channels = np.shape(T)
    bottom_padding = int((t_rows - 1) / 2)
    top_padding = (t_rows - 1) - bottom_padding
    left_padding = int((t_cols - 1) / 2)
    right_padding = (t_cols - 1) - left_padding

    # A 0-1 array, where 1's appear when the product of the bounding box with
    # top left corner with the template is high, otherwise 0
    grid = np.zeros((n_rows, n_cols))
    prods = np.zeros((n_rows, n_cols))

    for row in range(n_rows):
        for col in range(n_cols):
            prod = heatmap[row, col]
            if prod > threshold:
                grid[row, col] = 1
                prods[row, col] = prod

    # Using the grid, find clusters. Average all the points in a cluster to
    # get the bounding box for that cluster.
    clusters = label_grid(grid)

    for cluster in clusters:
        
        max_r, max_c = max(cluster, key = lambda pt : prods[pt[0], pt[1]])
        max_prod = prods[max_r, max_c]

        # Add the bounding box
        tl_row = max_r - top_padding
        tl_col = max_c - left_padding
        br_row = tl_row + t_rows
        br_col = tl_col + t_cols 

        if tl_row > 0 and tl_col > 0:
            output.append([tl_row, tl_col, br_row, br_col, max_prod])

    '''
    END YOUR CODE
    '''

    return output


def delete_duplicates(bounding_boxes, l_rows=10, l_cols=10):
    # Clean up bounding boxes by removing duplicates
    visited = set()

    bounding_boxes = sorted(bounding_boxes, key = lambda box : box[4], reverse=True)
    new_boxes = []

    for box in bounding_boxes:
        tl_row, tl_col, br_row, br_col, prod = box

        if (tl_row, tl_col) not in visited:
            new_boxes.append(box)

            for i in range(tl_row - l_rows * 2, br_row + l_rows * 2):
                for j in range(tl_col - l_cols * 2, br_col + l_cols * 2):
                    visited.add((i, j))

    return new_boxes

def detect_red_light_mf(I, name):
    '''
    This function takes a numpy array <I> and returns a list <output>.
    The length of <output> is the number of bounding boxes predicted for <I>. 
    Each entry of <output> is a list <[row_TL,col_TL,row_BR,col_BR,score]>. 
    The first four entries are four integers specifying a bounding box 
    (the row and column index of the top left corner and the row and column 
    index of the bottom right corner).
    <score> is a confidence score ranging from 0 to 1. 

    Note that PIL loads images in RGB order, so:
    I[:,:,0] is the red channel
    I[:,:,1] is the green channel
    I[:,:,2] is the blue channel
    '''

    '''
    BEGIN YOUR CODE
    '''
    output = []

    lights = []
    # imgs = []

    def save_light_img(img, coords):
        c1, c2, c3, c4 = coords
        # imgs.append(img.crop((c1, c2, c3, c4)))
        picture = np.array(img, dtype=float)
        lights.append(picture[c2:c4, c1:c3, :])

    # Use an example traffic light from the first image
    im0 = Image.open('./data/RedLights2011_Medium/RL-001.jpg')
    save_light_img(im0, [316, 154, 323, 161])
    
    #formatted = (lights[0]).astype('uint8')
    #img = Image.fromarray(formatted)
    #img.show()
    '''
    im1 = Image.open('./data/RedLights2011_Medium/RL-012.jpg')
    save_light_img(im1, [299, 37, 318, 55])

    im2 = Image.open('./data/RedLights2011_Medium/RL-021.jpg')
    save_light_img(im2, [282, 148, 289, 156])

    im3 = Image.open('./data/RedLights2011_Medium/RL-139.jpg')
    save_light_img(im3, [338, 172, 348, 182])

    im4 = Image.open('./data/RedLights2011_Medium/RL-051.jpg')
    save_light_img(im4, [316, 218, 325, 226])
    '''

    for i in range(len(lights)):
        T = lights[i]
        # img = imgs[i]
        # img.save('./data/template_images/temp_{}.jpg'.format(i))
        heatmap = compute_convolution(I, T)

        # For visualization purposes only: save the heatmap
        # img = Image.fromarray(np.uint8(heatmap * 255) , 'L')
        # img.save('./data/heatmap_images/heatmap_' + name, 'JPEG')

        output.extend(predict_boxes(heatmap, T))

    # Clean up any remaining duplicate boxes across sizes
    output = delete_duplicates(output, T.shape[0], T.shape[1])

    '''
    END YOUR CODE
    '''

    for i in range(len(output)):
        assert len(output[i]) == 5
        assert (output[i][4] >= 0.0) and (output[i][4] <= 1.0)

    return output

# Note that you are not allowed to use test data for training.
# set the path to the downloaded data:
data_path = './data/RedLights2011_Medium'

# load splits: 
split_path = './data/hw02_splits'
file_names_train = np.load(os.path.join(split_path,'file_names_train.npy'))
file_names_test = np.load(os.path.join(split_path,'file_names_test.npy'))

# set a path for saving predictions:
preds_path = './data/hw02_preds'
os.makedirs(preds_path, exist_ok=True) # create directory if needed

# Set this parameter to True when you're done with algorithm development:
done_tweaking = True

'''
Make predictions on the training set.
'''

preds_train = {}
for i in range(len(file_names_train)):

    # read image using PIL:
    I = Image.open(os.path.join(data_path,file_names_train[i]))

    # convert to numpy array:
    I = np.asarray(I)

    print("{}/{}".format(i, len(file_names_train)))     
    print(file_names_train[i])
    preds_train[file_names_train[i]] = detect_red_light_mf(I, file_names_train[i])

# save preds (overwrites any previous predictions!)
with open(os.path.join(preds_path,'preds_train.json'),'w') as f:
    json.dump(preds_train,f)


if done_tweaking:
    '''
    Make predictions on the test set. 
    '''
    preds_test = {}
    for i in range(len(file_names_test)):

        # read image using PIL:
        I = Image.open(os.path.join(data_path,file_names_test[i]))

        # convert to numpy array:
        I = np.asarray(I)

        preds_test[file_names_test[i]] = detect_red_light_mf(I, file_names_test[i])

    # save preds (overwrites any previous predictions!)
    with open(os.path.join(preds_path,'preds_test.json'),'w') as f:
        json.dump(preds_test,f)
