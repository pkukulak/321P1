import numpy as np

def knn(train_in, train_targ, test_in, k):
    '''
    Apply k-Nearest Neighbors to train_in,
    returning a prediction vector for test_in.
    '''
    n_test, _ = test_in.shape
    test_prediction = np.empty([n_test, 1], dtype=train_targ.dtype)

    for i in xrange(n_test):
        distances = euclidean_distance(train_in, test_in[i]).reshape(-1, 1)
        nearest = np.argsort(distances, axis=0)[:k]
        majority_winner = mode(train_targ[nearest])
        test_prediction[i] = majority_winner
    return test_prediction

def classification_rate(prediction, target):
    '''
    Calculate the classification rate of the prediction
    vector given the true target vector.
    '''
    return np.mean(prediction == target) * 100.0

def mode(x):
    '''
    Find the most frequently occuring value in the array x.
    '''
    values, indices = np.unique(x, return_inverse=True)
    counts = np.bincount(indices)
    majority = counts.argmax()
    return values[majority]

def euclidean_distance(x, y):
    '''
    Return the L2-distance between x, y.
    '''
    c = np.apply_along_axis(np.sum, 1, (x - y)**2)
    return np.sqrt(c)
