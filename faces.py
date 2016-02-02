import sys
import numpy as np
import matplotlib.pyplot as plt
from knn import *
from get_data import *

K_VALUES = [k for k in range(1, 26)]

'''
This program may be run from the command line in the following ways:
    1) python faces.py
       This defaults to classifying based on the given subset of actors
       defined in get_data.py, with classes being the actors identities.
    2) python faces.py -g
       Using the -g flag will run the classification task on the given
       subset of actors defined in get_data.py, with binary classes of
       male and female genders.
    3) python faces.py -fg
       Using the -fg flag will run the classification task on
       the entire dataset modulo the subset defined in get_data.py,
       with binary classes of male and female genders.
'''

def run_knn(train_in, train_targ, valid_in, valid_targ, test_in, test_targ):
    '''
    Run k-NN, where k ranges over K_VALUES and plot performance.
    '''
    train_rates = []
    valid_rates = []
    test_rates = []
    for k in K_VALUES:
        print "Running {}-NN.".format(k)
        train_prediction = knn(train_in, train_targ, train_in, k)
        valid_prediction = knn(train_in, train_targ, valid_in, k)
        test_prediction = knn(train_in, train_targ, test_in, k)

        train_rates += [classification_rate(train_prediction, train_targ)]
        valid_rates += [classification_rate(valid_prediction, valid_targ)]
        test_rates += [classification_rate(test_prediction, test_targ)]
        
        print "TRAINING CLASSIFICATION RATE = {}".format(train_rates[k-1])
        print "VALIDATION CLASSIFICATION RATE = {}".format(valid_rates[k-1])
        print "TEST CLASSIFICATION RATE = {}".format(test_rates[k-1])

    best_k = K_VALUES[np.argmax(valid_rates)]
    print "MOST ACCURATE MODEL : {}-NN".format(best_k)
    test_prediction = knn(train_in, train_targ, test_in, best_k)
    test_classification = classification_rate(test_prediction, test_targ)
    print "TEST CLASSIFICATION RATE = {}".format(test_classification)
    
    plt.title("Classification of Actors using k-NN")
    plt.xlabel("k")
    plt.axis([np.min(K_VALUES), np.max(K_VALUES), 0, 100])
    plt.grid(True)

    plt.ylabel("Training Classification Rate")
    plt.plot(K_VALUES, train_rates, 'go')
    plt.show()

    plt.ylabel("Validation Classification Rate")
    plt.plot(K_VALUES, valid_rates, 'ro')
    plt.show()

    plt.ylabel("Test Classification Rate")
    plt.plot(K_VALUES, test_rates, 'bo')
    plt.show()

if __name__ == "__main__":
    flags = sys.argv[1:]

    if flags in [['-fg'],['-gf']]:
        classification = "gender"
        dataset = "full"
    elif flags == ['-g']:
        classification = "gender"
        dataset = "subset"
    elif not flags:
        classification = "identity"
        dataset = "subset"
    else:
        print "USAGE : python main.py [-fg|-g] [FILE]"
        sys.exit(-1)

    print "Loading data."
    male_data = load_data("cropped/male/", classification, dataset)
    female_data = load_data("cropped/female/", classification, dataset)
    data = np.vstack((male_data, female_data))

    print "Splitting into training, validation, and test sets."
    (train_in, train_targ,
      valid_in, valid_targ,
        test_in, test_targ) = train_valid_test_split(data)

    run_knn(train_in, train_targ, valid_in, valid_targ, test_in, test_targ)

    sys.exit(1)
