from pylab import *
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook
import random
import time
from scipy.misc import imread
from scipy.misc import imresize
from scipy.misc import imsave
import matplotlib.image as mpimg
import os
from scipy.ndimage import filters
import urllib
import re

GENDER_MAP = {}

MALE_ACT = list(set(["{} {}".format(a.split()[0], a.split()[1])
    for a in open("subset_actors.txt").readlines()]))
MALE_ACT.sort()

FEMALE_ACT = list(set(["{} {}".format(a.split()[0], a.split()[1])
    for a in open("subset_actresses.txt").readlines()]))
FEMALE_ACT.sort()

ALL_ACT = MALE_ACT + FEMALE_ACT

SUBSET_ACT = ['Gerard Butler', 'Daniel Radcliffe', 'Michael Vartan',
              'Lorraine Bracco', 'Peri Gilpin', 'Angie Harmon']

ALL_MINUS_SUBSET_ACT = [act for act in ALL_ACT if act not in SUBSET_ACT]

TRAIN_SIZE = 100
VALID_SIZE = 10
TEST_SIZE = 10

def rgb2gray(rgb):
    '''Author: Michael Guerzhoy
    Return the grayscale version of the RGB image rgb as a 2D numpy array
    whose range is 0..1
    Arguments:
    rgb -- an RGB image, represented as a numpy array of size n x m x 3. The
    range of the values is 0..255
    '''
    if len(rgb.shape) == 2:
        return rgb
    r, g, b = rgb[:,:,0], rgb[:,:,1], rgb[:,:,2]
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    return gray/255

def timeout(func, args=(), kwargs={}, timeout_duration=1, default=None):
    '''From:
    http://code.activestate.com/recipes/473878-timeout-function-using-threading/'''
    import threading
    class InterruptableThread(threading.Thread):
        def __init__(self):
            threading.Thread.__init__(self)
            self.result = None

        def run(self):
            try:
                self.result = func(*args, **kwargs)
            except:
                self.result = default

    it = InterruptableThread()
    it.start()
    it.join(timeout_duration)
    if it.isAlive():
        return False
    else:
        return it.result

def process_image(uncropped, cropped, dims):
    '''
    Using dims, crop and grayscale the image, stored at uncropped. 
    Then, save it in a folder containing cropped images.
    '''
    image = imread(uncropped)
    gray_image = rgb2gray(image)
    x1, y1 = int(dims[0]), int(dims[1])
    x2, y2 = int(dims[2]), int(dims[3])
    cropped_gray_image = imresize(gray_image[y1:y2, x1:x2], (32, 32))
    imsave(cropped, cropped_gray_image)

def split_to_lower(s):
    '''
    Return the last name of s in lowercase letters.
    '''
    return s.split()[1].lower()

def load_data(folder, class_type, data_set):
    '''
    Load all of the photos located in the given folder
    into a numpy array with labels appended to each datapoint.
    class_type is one of 
        -"gender"
        -"identity"
    data_set is one of
        -"full"
        -"subset"
    '''
    data = np.empty([1, 1025])
    gender = folder.split('/')[1]
    files = [f for f in os.listdir(folder) if re.search('\d', f)]
    if data_set == "full":
        actors = map(split_to_lower, ALL_MINUS_SUBSET_ACT)
    elif data_set == "subset":
        actors = map(split_to_lower, SUBSET_ACT)
    for filename in files:
        d = re.search("\d", filename)
        identity = filename[:d.start()]
        if identity not in actors:
            continue
        if class_type == "gender":
            target = gender
        elif class_type == "identity":
            target = identity
        try:
            img = imread(folder + filename).reshape(1, 1024)
            img = np.append(img, target).reshape(-1, 1025)
            data = np.append(data, img, axis=0)
        except ValueError:
            print "{} ill-formatted. Deleting.".format(filename)
            os.remove(folder + filename)
    return data[1:]

def train_valid_test_split(data):
    '''
    Return a training set, a validation set, a test set,
    and all of their respective targets. All are obtained from
    the input data.
    '''
    data = data[data[:,-1].argsort()]
    prev = 0

    N, M = data.shape
    targets, target_indices = np.unique(data[:, -1], return_index=True)
    target_indices = np.append(target_indices, N)
    num_targets = targets.size

    train_in = np.empty([TRAIN_SIZE * num_targets, M-1], dtype=data.dtype)
    train_targ = np.empty([TRAIN_SIZE * num_targets, 1], dtype=data.dtype)

    valid_in = np.empty([VALID_SIZE * num_targets, M-1], dtype=data.dtype)
    valid_targ = np.empty([VALID_SIZE * num_targets, 1], dtype=data.dtype)

    test_in, test_targ = get_test_data(data, targets, target_indices)

    for i in xrange(num_targets):
        train_range = xrange(TRAIN_SIZE * i, TRAIN_SIZE * (i + 1))
        valid_range = xrange(VALID_SIZE * i, VALID_SIZE * (i + 1))
        
        test_start = target_indices[i+1] - TEST_SIZE
        no_test = data[prev:test_start]
        np.random.shuffle(no_test)

        t_in = no_test[:TRAIN_SIZE, :M-1]
        t_targ = no_test[:TRAIN_SIZE, M-1].reshape(-1, 1)
        train_in[train_range] = t_in
        train_targ[train_range] = t_targ

        v_in = no_test[TRAIN_SIZE:TRAIN_SIZE+VALID_SIZE, :M-1]
        v_targ = no_test[TRAIN_SIZE:TRAIN_SIZE+VALID_SIZE, M-1].reshape(-1, 1)
        valid_in[valid_range] = v_in
        valid_targ[valid_range] = v_targ

        prev = target_indices[i+1] + 1

    return (train_in.astype(np.float), train_targ, 
            valid_in.astype(np.float), valid_targ, 
              test_in.astype(np.float), test_targ)

def get_test_data(data, targets, target_indices):
    '''
    Returning a test set and corresponding targets.
    Distinct from producing training and validation sets
    because test data is selected deterministically.
    '''
    prev = 0
    N, M = data.shape
    num_targets = targets.size

    test_in = np.empty([TEST_SIZE * num_targets, M-1], dtype=data.dtype)
    test_targ = np.empty([TEST_SIZE * num_targets, 1], dtype=data.dtype)

    for i in xrange(num_targets):
        test_range = xrange(TEST_SIZE * i, TEST_SIZE * (i+1))
        test_start = target_indices[i+1] - TEST_SIZE
        test_end = target_indices[i+1]
        curr_test = data[test_start:test_end, :M-1]
        test_in[test_range] = curr_test
        test_targ[test_range] = np.tile(targets[i], TEST_SIZE).reshape(-1, 1)

        prev = target_indices[i+1] + 1

    return test_in, test_targ

def download_data():
    '''
    Download and save all data from the internet.
    '''
    testfile = urllib.URLopener()
    for a in MALE_ACT:
        name = a.split()[1].lower()
        i = 0
        for line in open("subset_actors.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                crop_dims = line.split()[5].split(',')
                uncropped_filename = "uncropped/male/"+filename
                cropped_filename = "cropped/male/"+filename
                timeout(testfile.retrieve, (line.split()[4], uncropped_filename), {}, 15)

                if not os.path.isfile("uncropped/male/"+filename):
                    continue
                try:
                    process_image(uncropped_filename, cropped_filename, crop_dims)
                    i += 1
                    print filename
                    os.remove(uncropped_filename)
                except:
                    print "Error occured : {}".format(filename)

    for a in FEMALE_ACT:
        name = a.split()[1].lower()
        i = 0
        for line in open("subset_actresses.txt"):
            if a in line:
                filename = name+str(i)+'.'+line.split()[4].split('.')[-1]
                crop_dims = line.split()[5].split(',')
                uncropped_filename = "uncropped/female/"+filename
                cropped_filename = "cropped/female/"+filename
                timeout(testfile.retrieve, (line.split()[4], uncropped_filename), {}, 15)

                if not os.path.isfile("uncropped/female/"+filename):
                    continue
                try:
                    process_image(uncropped_filename, cropped_filename, crop_dims)
                    i += 1
                    print filename
                    os.remove(uncropped_filename)
                except:
                    print "Error occured : {}".format(filename)

if __name__ == "__main__":
    if not os.path.exists("cropped/"):
        os.makedirs("cropped/")
        if not os.path.exists("cropped/male/"):
            print "Creating directory ./cropped/male/"
            os.makedirs("cropped/male")
        if not os.path.exists("cropped/female/"):
            print "Creating directory ./cropped/female/"
            os.makedirs("cropped/female")

    if not os.path.exists("uncropped/"):
        os.makedirs("uncropped/")
        if not os.path.exists("uncropped/male/"):
            print "Creating directory ./uncropped/male/"
            os.makedirs("uncropped/male/")
        if not os.path.exists("uncropped/female"):
            print "Creating directory ./uncropped/female/"
            os.makedirs("uncropped/female")

    download_data()
