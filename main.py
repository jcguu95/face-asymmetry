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
# local modules
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

data.CURRENT_PROFILE["null_ids"] = []
data.CURRENT_PROFILE["non_null_ids"] = []
for id in data.ids():
    if (location(id)) is np.nan:
        data.CURRENT_PROFILE["null_ids"].append(id)
    else:
        data.CURRENT_PROFILE["non_null_ids"].append(id)
# NULL_IDS = []
# NON_NULL_IDS = []
# for id in data.ids():
#     if (location(id)) is np.nan:
#         NULL_IDS.append(id)
#     else:
#         NON_NULL_IDS.append(id)

def x_mean ():
    result = 0
    #for id in NON_NULL_IDS:
    for id in data.CURRENT_PROFILE["non_null_ids"]:
        loc = location(id)
        #result += loc[0]/len(NON_NULL_IDS)
        result += loc[0]/len(data.CURRENT_PROFILE["non_null_ids"])
    return result

def y_mean ():
    result = 0
    #for id in NON_NULL_IDS:
    for id in data.CURRENT_PROFILE["non_null_ids"]:
        loc = location(id)
        #result += loc[1]/len(NON_NULL_IDS)
        result += loc[1]/len(data.CURRENT_PROFILE["non_null_ids"])
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
    #tmp_IDS = list(NON_NULL_IDS)
    tmp_IDS = list(data.CURRENT_PROFILE["non_null_ids"])
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
#MIRROR_DICT = {}
data.CURRENT_PROFILE["mirror_dict"] = {}
#for id in NON_NULL_IDS:
for id in data.CURRENT_PROFILE["non_null_ids"]:
    #MIRROR_DICT[id] = ''
    data.CURRENT_PROFILE["mirror_dict"][id] = ''

## Popularization.
#for id in NON_NULL_IDS:
for id in data.CURRENT_PROFILE["non_null_ids"]:
    #if MIRROR_DICT[id] == '':
    if data.CURRENT_PROFILE["mirror_dict"][id] == '':
        cands = mirror_candidates(id)
        #for k in range(len(NON_NULL_IDS)):
        for k in range(len(data.CURRENT_PROFILE["non_null_ids"])):
            # if MIRROR_DICT[cands[k]] == '':
            #     MIRROR_DICT[id] = cands[k]
            #     MIRROR_DICT[cands[k]] = id
            #     break
            if data.CURRENT_PROFILE["mirror_dict"][cands[k]] == '':
                data.CURRENT_PROFILE["mirror_dict"][id] = cands[k]
                data.CURRENT_PROFILE["mirror_dict"][cands[k]] = id
                break

#LEFT_IDS = []
#RIGHT_IDS = []
data.CURRENT_PROFILE['left_ids'] = []
data.CURRENT_PROFILE['right_ids'] = []
#for id in NON_NULL_IDS:
for id in data.CURRENT_PROFILE["non_null_ids"]:
    if location(id)[0] <= x_mean():
        #LEFT_IDS.append(id)
        data.CURRENT_PROFILE['left_ids'].append(id)
        #RIGHT_IDS.append(MIRROR_DICT[id])
        data.CURRENT_PROFILE['right_ids'].append(data.CURRENT_PROFILE["mirror_dict"][id])

#assert(len(LEFT_IDS)==len(RIGHT_IDS))
assert(len(data.CURRENT_PROFILE['left_ids']) == \
       len(data.CURRENT_PROFILE['right_ids']))

#for kk in range(len(LEFT_IDS)):
for kk in range(len(data.CURRENT_PROFILE['left_ids'])):
    # Make sure that the order in LEFT_IDS and RIGHT_IDS respect
    # MIRROR_DICT[_].
    #assert(MIRROR_DICT[LEFT_IDS[kk]] == RIGHT_IDS[kk])
    assert(data.CURRENT_PROFILE["mirror_dict"][data.CURRENT_PROFILE['left_ids'][kk]] == data.CURRENT_PROFILE['right_ids'][kk])

def dxdys_left (frame):
    result = []
    #for id in LEFT_IDS:
    for id in data.CURRENT_PROFILE['left_ids']:
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
    #for id in RIGHT_IDS:
    for id in data.CURRENT_PROFILE['right_ids']:
        entry = data.data(frame=frame, id=id)
        if entry is np.nan:
            # If `id` is missing from `frame`, assume (dx,dy)=(0,0).
            result.append(0)
            result.append(0)
        else:
            result.append(-entry['dx']) # flip dx to be compare with mirror point
            result.append(entry['dy'])
    return result

#VECTORS_LEFT, VECTORS_RIGHT = [], []
data.CURRENT_PROFILE['vectors_left'], data.CURRENT_PROFILE['vectors_right'] = [], []
for frame in data.frames():
    #VECTORS_LEFT.append(dxdys_left(frame))
    #VECTORS_RIGHT.append(dxdys_right(frame))
    data.CURRENT_PROFILE['vectors_left'].append(dxdys_left(frame))
    data.CURRENT_PROFILE['vectors_right'].append(dxdys_right(frame))

def asymmetry (vectors_left, vectors_right):
    assert len(vectors_left)==len(vectors_right)
    n_components = len(vectors_left)
    #
    import json
    print("Analyzing asymmetry for profile:\n.")
    print(json.dumps(data.CURRENT_PROFILE["name"], indent=2, default=str))
    print(json.dumps(data.CURRENT_PROFILE["doc"], indent=2, default=str))
    print(json.dumps(data.CURRENT_PROFILE["ids"], indent=2, default=str))
    print(json.dumps(data.CURRENT_PROFILE["frames"], indent=2, default=str))
    # Excellent tutorial for PCA in python:
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    from sklearn.decomposition import PCA
    pca = PCA(n_components=n_components)  # get n_components principle components
    #
    pca.fit(vectors_left)       # ~130 vectors, each having ~600 coordinates.
    left_normalized_components = pca.components_ # get principle components
    left_eigenvalues = list(map(math.sqrt, pca.explained_variance_))
    left_components = list(map(u.scalar_mul, left_eigenvalues, left_normalized_components))
    #
    pca.fit(vectors_right)       # ~130 vectors, each having ~600 coordinates.
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
    for k in range(n_components):
        result += (left_eigenvalues[k]+right_eigenvalues[k])/2 * distances[k]
    result = result / (n_components * (left_eigenvalues[0]+right_eigenvalues[0])/2)
    print("Rate of asymmetry is %.5f." % result)
    return result

asymmetry(data.CURRENT_PROFILE['vectors_left'], \
          data.CURRENT_PROFILE['vectors_right'])

# Note: There's no natural cut-off that dictates if an asymmetry
# value is too high or not. We need to compare among different
# data after all. When we compare, we have to make the number of
# frames the same. It is not clear in general how the asymmetry
# grows as the number of frames grow.
