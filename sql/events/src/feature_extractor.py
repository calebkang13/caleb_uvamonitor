from sklearn.preprocessing import Normalizer
from collections import Counter, defaultdict
from sklearn.externals import joblib
from multiprocessing import Pool
from scipy import stats
from tqdm import tqdm
import glob, pdb, os
import numpy as np
import scipy as sp


SPLIT_DATA = '/home/monitor_db/events/data/split/'
OUTPUT = '/home/monitor_db/events/data/vector/hcdm_vectors.csv'
VECTOR_SCRIPT = '/home/monitor_db/events/data/vector/hcdm_vector_table.sql'
EVENT_SCRIPT = '/home/monitor_db/events/data/vector/event_table.sql'
API_SCRIPT = '/home/monitor_db/events/src/data.py'
NORMALIZER_PATH = '/home/monitor_db/events/src/normalizer.save'


def create_subsets(data, subset_length, step_size):

    if data.shape[0] >= subset_length:

        data = data[int(data.shape[0] - np.floor(data.shape[0]/subset_length)*subset_length):]
        start_index, end_index = 0, subset_length

        subsets = []
        while end_index <= data.shape[0]:
            subsets.append(data[start_index:end_index])
            start_index += step_size
            end_index += step_size

        return subsets

    else:

        return None


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

    streams = [p for p in os.listdir(SPLIT_DATA) if not p.startswith('.')]

    ####################################
    ########## CREATE VECTORS ##########
    ####################################

    vectors, names = [], []
    for stream in tqdm(streams):

        filename = '%s%s' % (SPLIT_DATA, stream)
        streamname = stream.split('.csv')[0]
        
        data = open(filename).readlines()
        data = np.asarray([(float(line.strip().split(',')[0]), float(line.strip().split(',')[1])) for line in open(filename).readlines()])

        subsets = create_subsets(data, 500, 100)

        try:

            if subsets:

                for subset in subsets:

                    start, end = subset[:, 0][0], subset[:, 0][-1]
                    timeseries_helper = data_feature_extractor(np.transpose(subset))

                    groups = [
                        timeseries_helper.getF_2019_Pandey_single()[1],
                        timeseries_helper.getF_2019_Pandey_split(5)[1]
                    ]

                    vectors.append(np.asarray([value for features in groups for value in features]))
                    names.append([streamname, start, end])


        except Exception as e:

            print(e)


    vectors = np.asarray(vectors)
    normalizer = Normalizer().fit(vectors)
    joblib.dump(normalizer, NORMALIZER_PATH)
    vectors = normalizer.transform(vectors)

    with open(OUTPUT, 'w') as w:
    
        for i in range(len(vectors)):
            name, vector = names[i], vectors[i]
            streamname = name[0]

            w.write('%s,' % name[0])
            for value in name[1:]: w.write('%.10f,' % value)
            for value in vector[:-1]: w.write('%.10f,' % value)
            w.write('%.10f\n' % vector[-1])

    #######################################
    ########## CREATE SQL SCRIPT ##########
    #######################################

    filename = '%s%s' % (SPLIT_DATA, streams[0])
    data = open(filename).readlines()
    data = np.asarray([(float(line.strip().split(',')[0]), float(line.strip().split(',')[1])) for line in open(filename).readlines()])

    start, end = data[:, 0][0], data[:, 0][-1]
    timeseries_helper = data_feature_extractor(np.transpose(data))

    groups = [
        timeseries_helper.getF_2019_Pandey_single()[1],
        timeseries_helper.getF_2019_Pandey_split(5)[1]
    ]

    cols = [ 'STREAM', 'START', 'END' ]
    vector_length = len([value for features in groups for value in features])

    for i in range(vector_length): 
        cols += ['F%s' % (str(i + 1))]

    sqlstr = 'USE monitor_db;\n\n'
    sqlstr += 'DROP TABLE IF EXISTS hcdm_vectors_start;\n\n'
    sqlstr += 'CREATE TABLE hcdm_vectors_start (\n'
    sqlstr += '\t%s VARCHAR(100),\n' % cols[0]
    for col in cols[1:3]: sqlstr += '\t%s INTEGER,\n' % col
    for col in cols[3:-1]: sqlstr += '\t%s DOUBLE,\n' % col
    sqlstr += '\t%s DOUBLE\n' % cols[-1]
    sqlstr += ');\n\n';

    sqlstr += 'LOAD DATA LOCAL INFILE "/home/monitor_db/events/data/vector/hcdm_vectors.csv" INTO TABLE hcdm_vectors_start FIELDS TERMINATED BY ",";\n\n'

    sqlstr += 'DROP TABLE IF EXISTS hcdm_vectors;\n\n'

    sqlstr += 'CREATE TABLE hcdm_vectors (\n'
    sqlstr += '\tSELECT '
    for col in cols[:-1]: sqlstr += '%s, ' % col
    sqlstr += '%s\n' % cols[-1]
    sqlstr += '\tFROM hcdm_vectors_start as a, (SELECT DISTINCT name FROM stream_data) AS b \n\tWHERE a.stream = b.name\n);\n\n'

    sqlstr += 'DROP TABLE hcdm_vectors_start;'

    with open(VECTOR_SCRIPT, 'w') as w:
        w.write(sqlstr)

    sqlstr = 'USE monitor_db;\n\n'
    sqlstr += 'DROP TABLE IF EXISTS event;\n\n'
    sqlstr += 'CREATE TABLE event (\n'
    sqlstr += '\tevent VARCHAR(100),\n'
    sqlstr += '\tstream VARCHAR(100),\n'
    sqlstr += '\ttimeStart INTEGER,\n'
    sqlstr += '\ttimeEnd INTEGER,\n'
    for col in cols[3:-1]: sqlstr += '\t%s DOUBLE,\n' % col
    sqlstr += '\t%s DOUBLE\n' % cols[-1]
    sqlstr += ');\n\n';

    with open(EVENT_SCRIPT, 'w') as w:
        w.write(sqlstr)

    datastr = "data = {\n"
    datastr += "\t'num_coeff': 2,\n"
    datastr += "\t'cols': ["
    for col in cols[:-1]: datastr += "'%s', " % col
    datastr += "'%s' ]\n" % cols[-1]
    datastr += "}"

    with open(API_SCRIPT, 'w') as w:
        w.write(datastr)


if __name__ == '__main__':
    main()
