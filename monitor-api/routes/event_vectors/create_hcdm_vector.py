from collections import Counter, defaultdict
from sklearn.externals import joblib
from multiprocessing import Pool
import glob, pdb, sys, os
from scipy import stats
import mysql.connector
from tqdm import tqdm
from data import data
import numpy as np
import scipy as sp


NORMALIZER_PATH = '/home/monitor_api/routes/event_vectors/normalizer.save'


def mode(ndarray,axis=0):

    if ndarray.size == 1: 
        return (ndarray[0], 1)

    elif ndarray.size == 0: 
        raise Exception('Attempted to find mode on an empty array!')

    try: 
        axis = [i for i in range(ndarray.ndim)][axis]

    except IndexError: 
        raise Exception('Axis %i out of range for array with %i dimension(s)' % (axis,ndarray.ndim))

    srt = np.sort(ndarray, axis=axis)
    dif = np.diff(srt, axis=axis)

    shape = [i for i in dif.shape]
    shape[axis] += 2

    indices = np.indices(shape)[axis]
    index = tuple([slice(None) if i != axis else slice(1,-1) for i in range(dif.ndim)])
    indices[index][dif == 0] = 0
    indices.sort(axis=axis)

    bins = np.diff(indices, axis=axis)
    location = np.argmax(bins, axis=axis)
    mesh = np.indices(bins.shape)

    index = tuple([slice(None) if i != axis else 0 for i in range(dif.ndim)])
    index = [mesh[i][index].ravel() if i != axis else location.ravel() for i in range(bins.ndim)]
    counts = bins[tuple(index)].reshape(location.shape)

    index[axis] = indices[tuple(index)]
    modals = srt[tuple(index)].reshape(location.shape)
    
    return (modals, counts)


class data_feature_extractor():

    def __init__(self, X):
        self.X = np.asarray(X) # X
        self.functions = [ 'getF_2019_Pandey_single', 'getF_2019_Pandey_split' ]


    def getF_2019_Pandey(self, X):

        ''' 'min', 'median', 'mean', 'max', 'std', 'skewness', 'kurtosis', 'entropy', 'slope' '''
        N, D = X.shape
        # dim = 10
        dim = 9 
        F = np.zeros([N, dim])

        F[:, 0] = np.min(X, 1)
        F[:, 1] = np.median(X, 1)
        F[:, 2] = np.mean(X, 1)
        F[:, 3] = np.max(X, 1)
        F[:, 4] = np.std(X, 1)
        F[:, 5] = sp.stats.skew(X, 1)
        F[:, 6] = sp.stats.kurtosis(X, 1)
        
        # digitize the data for the calculation of entropy if it only contains less than 100 discreate values
        XX = np.zeros(X.shape)
        bins = 100

        for i in range(X.shape[0]):
            if len(np.unique(X[i,:])) < bins:
                XX[i,:] = X[i,:]
            else:
                XX[i,:] = np.digitize(X[i,:], np.linspace(min(X[i,:]), max(X[i,:]), num=bins))        

        F[:, 7] = sp.stats.entropy(XX.T)
        F[:, 8] = mode(X,1)[0]
        # F[:, 9] = np.polyfit(X[0], X[1], 1)[0]

        # names = [ 'min', 'median', 'mean', 'max', 'std', 'skewness', 'kurtosis', 'entropy', 'mode', 'slope' ]
        names = [ 'min', 'median', 'mean', 'max', 'std', 'skewness', 'kurtosis', 'entropy', 'mode' ]

        # check illegal features nan/inf
        F[np.isnan(F)] = 0
        F[np.isinf(F)] = 0

        return F


    def getF_2019_Pandey_single(self):

        return self.getF_2019_Pandey(self.X)


    def getF_2019_Pandey_split(self, division):

        split = np.array_split(self.X, division, axis=1)

        transformed = []
        for item in split:
            transformed.append(self.getF_2019_Pandey(item))

        vector = np.hstack(transformed)
        return vector


def main():

    event, stream, start, end = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    num_coeff, cols = data['num_coeff'], ['EVENT'] + data['cols']

    db = mysql.connector.connect(
        host="monitor_db",
        user="root",
        database="monitor_db",
        passwd="monitor_db"
    )

    query = "SELECT timestamp, value FROM stream_data WHERE name='%s' AND TIMESTAMP >= '%s' AND TIMESTAMP <= '%s' ORDER BY TIMESTAMP DESC" % (stream, start, end)
    cursor = db.cursor()
    cursor.execute(query)

    x, y = [], []
    for pair in cursor: 
        x.append(pair[0])
        y.append(pair[1])

    x, y = np.asarray(x), np.asarray(y)
    raw = np.transpose(np.asarray([x, y]))
    timeseries_helper = data_feature_extractor(np.transpose(raw))

    groups = [
        timeseries_helper.getF_2019_Pandey_single()[1],
        timeseries_helper.getF_2019_Pandey_split(5)[1]
    ]

    normalizer = joblib.load(NORMALIZER_PATH)
    vector = [value for features in groups for value in features]
    vector = normalizer.transform([vector])[0]
    vector = np.asarray([event, stream] + [start, end] + ['%.10f' % item for item in vector])
    for item in vector: print('%s,' % item)
    sys.stdout.flush()


if __name__ == '__main__':
    main()
