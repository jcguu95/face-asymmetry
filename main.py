# Usage :: Manually mount
#   `/media/pawel/disk12T/FaceAnalysis/fm_annot/img/` to `./data/mnt/`
# with the shell command
#
#   sshfs jinchengguu@129.49.109.97:/media/pawel/disk12T/ ./data/mnt
#
# and run the following code. The end result is asymmetry(1).

# global modules
import math
import pandas as pd
import numpy as np
import warnings
from compose import compose

import utils as u
import data

def locations (id):
    result = []
    for frame in data.frames():
        d = data.data(frame=frame, id=id)
        if d is not np.nan:
            (x,y) = (d['x0'],d['y0'])
            result.append((x,y))
    return set(result)

def location (id):
    locs = list(locations(id))
    if len(locs) == 1:
        return locs[0]
    elif len(locs) == 0:
        return np.nan
    else:
        warnings.warn("id:{id} has more than one location.".format(id=id))

NULL_IDS = []
NON_NULL_IDS = []
for id in data.ids():
    if (location(id)) is np.nan:
        NULL_IDS.append(id)
    else:
        NON_NULL_IDS.append(id)

# >>> NULL_IDS = [2, 17, 30, 126, 128, 194, 201, 202, 206, 207,
#             216, 217, 218, 233, 276, 307, 308, 315, 316, 317,
#             323, 325, 326, 373, 398, 476, 483, 492, 604, 605,
#             606, 607, 608, 610]
#
# >>> len(NULL_IDS) = 34

def x_mean ():
    result = 0
    for id in NON_NULL_IDS:
        loc = location(id)
        result += loc[0]/len(NON_NULL_IDS)
    return result

def y_mean ():
    result = 0
    for id in NON_NULL_IDS:
        loc = location(id)
        result += loc[1]/len(NON_NULL_IDS)
    return result

def mirror_candidates (id):
    '''Returns the id mirrored along the central x-axis. This
    function should square to identity. (Caveat: But it doesn't!)'''
    #raise Exception("Not correctly yet implemented.") # TODO To implement.
    (x, y) = location(id)
    mirror_point = (2*x_mean() - x, y)
    def sqr_dist_to_mirror_point (id):
       (a, b) = location(id)
       return (a - mirror_point[0])**2 + (b - mirror_point[1])**2
    tmp_IDS = list(NON_NULL_IDS)
    tmp_IDS.sort(key=sqr_dist_to_mirror_point)
    return tmp_IDS

def mirror (id):
    return mirror_candidates(id)[0]

def stability (id):
    '''Return the least n such that mirror^n(id)==mirror^(n+2)(id).'''
    result = 0
    point = id
    while point != mirror(mirror(point)):
        result += 1
        point = mirror(point)
    return result

# As the data points are not symmetrically distributed, the
# function mirror is not an idempotent. Therefore, we create an
# idempotent dictionary adhocally mirror_dict[].
#
## Initialization
MIRROR_DICT = {}
for id in NON_NULL_IDS:
    MIRROR_DICT[id] = ''

## Popularization.
for id in NON_NULL_IDS:
    if MIRROR_DICT[id] == '':
        cands = mirror_candidates(id)
        for k in range(len(NON_NULL_IDS)):
            if MIRROR_DICT[cands[k]] == '':
                MIRROR_DICT[id] = cands[k]
                MIRROR_DICT[cands[k]] = id
                break

LEFT_IDS = []
RIGHT_IDS = []
for id in NON_NULL_IDS:
    if location(id)[0] <= x_mean():
        LEFT_IDS.append(id)
        RIGHT_IDS.append(MIRROR_DICT[id])

assert(len(LEFT_IDS)==len(RIGHT_IDS))

for kk in range(len(LEFT_IDS)):
    # Make sure that the order in LEFT_IDS and RIGHT_IDS respect
    # MIRROR_DICT[_].
    assert(MIRROR_DICT[LEFT_IDS[kk]] == RIGHT_IDS[kk])

def dxdys_left (frame):
    result = []
    for id in LEFT_IDS:
        entry = data.data(frame=frame, id=id)
        if entry is np.nan:
            # If `id` is missing from `frame`, assume (dx,dy)=(0,0).
            result.append(0)
            result.append(0)
        else:
            result.append(entry['dx'])
            result.append(entry['dy'])
    return result

def dxdys_right (frame):
    result = []
    for id in RIGHT_IDS:
        entry = data.data(frame=frame, id=id)
        if entry is np.nan:
            # If `id` is missing from `frame`, assume (dx,dy)=(0,0).
            result.append(0)
            result.append(0)
        else:
            result.append(-entry['dx']) # flip dx to be compare with mirror point
            result.append(entry['dy'])
    return result

VECTOR_LEFT, VECTOR_RIGHT = [], []
for frame in data.frames():
    VECTOR_LEFT.append(dxdys_left(frame))
    VECTOR_RIGHT.append(dxdys_right(frame))

def asymmetry (N):
    import json
    print("Analyzing asymmetry for profile:\n.")
    print(json.dumps(data.CURRENT_PROFILE, indent=2, default=str))
    # Excellent tutorial for PCA in python:
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    from sklearn.decomposition import PCA
    pca = PCA(n_components=N)  # get N principle components
    #
    pca.fit(VECTOR_LEFT)       # ~130 vectors, each having ~600 coordinates.
    left_normalized_components = pca.components_ # get principle components
    left_eigenvalues = list(map(math.sqrt, pca.explained_variance_))
    left_components = list(map(u.scalar_mul, left_eigenvalues, left_normalized_components))
    #
    pca.fit(VECTOR_RIGHT)       # ~130 vectors, each having ~600 coordinates.
    right_normalized_components = pca.components_ # get principle components
    right_eigenvalues = list(map(math.sqrt, pca.explained_variance_))
    right_components = list(map(u.scalar_mul, right_eigenvalues, right_normalized_components))
    #
    angle_diffs = list(map(u.angle_diff, left_components, right_components))
    distances   = list(map(compose(u.distance, u.vect_minus), left_components, right_components))
    length_ratios = list(map((lambda x,y: x/y), left_eigenvalues, right_eigenvalues))
    df = pd.DataFrame([left_eigenvalues, right_eigenvalues, angle_diffs, distances, length_ratios], \
                      index = ['left eigenvalues', 'right eigenvalues', \
                               'angle difference', 'distance', 'length ratio']).transpose()
    print(df)
    result = 0
    for k in range(N):
        result += (left_eigenvalues[k]+right_eigenvalues[k])/2 * distances[k]
    result = result / (N * (left_eigenvalues[0]+right_eigenvalues[0])/2)
    print("Rate of asymmetry is %.5f." % result)
    return result

asymmetry(len(data.CURRENT_PROFILE['frames']))

# Note: There's no natural cut-off that dictates if an asymmetry
# value is too high or not. We need to compare among different
# data after all. When we compare, we have to make the number of
# frames the same. It is not clear in general how the asymmetry
# grows as the number of frames grow.
