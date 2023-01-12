import numpy as np
import math
import scipy
import stats.marietan_test

# Utils
def dxdy_frame_concat (dxdy, frame, ids, flip):
    result = []
    for id in ids:
        dx = dxdy[frame][id][0]
        dy = dxdy[frame][id][1]
        if flip: dx = -dx
        result += [dx,dy] # concat
    return result

def norm (x,y):
    return math.sqrt(x**2+y**2)

# Stage 1: raw_data => data (points, dxdy)
def stage_1 ():
    ## Raw data
    __pointi=np.load("./data/pointi.npy")
    # __pointf=np.load("pointf.npy")  # We do not use this file.
    __dest_x=np.load("./data/dest_x.npy")
    __dest_y=np.load("./data/dest_y.npy")
    ## Raw data cleansing
    frames=range(0,__pointi.shape[0])
    amount_of_points=__pointi.shape[1]//2
    __points_from_id={}
    __points_from_old_id={}
    __id=0
    for old_id in range(0,amount_of_points):
        if not np.isnan(__pointi[0][old_id]):
            x = __pointi[0][old_id]
            y = __pointi[0][old_id+amount_of_points]
            # We assume that each __pointi is constant over different frames.
            __points_from_id[__id] = ([x,y], old_id)
            __points_from_old_id[old_id] = ([x,y], __id)
            __id+=1
    def effective_old_ids ():
        return __points_from_old_id.keys()
    def effective_p (old_id):
        return old_id in effective_old_ids()
    # We assume that each __pointi is constant over different frames.
    def xy (id):
        return __points_from_id[id][0]
    def x (id):
        return xy(id)[0]
    def y (id):
        return xy(id)[1]
    def old_id_of (id):
        return __points_from_id[id][1]
    def old_ids ():
        return range(0,amount_of_points)
    def ids ():
        return __points_from_id.keys()
    __x_mean  = 0
    for id in ids():
        __x_mean += x(id)
    __x_mean = __x_mean / len(ids())
    __dxdy={}
    for frame in frames:
        __dxdy[frame]={}
        for id in ids():
            dx = __dest_x[frame][old_id_of(id)]
            dy = __dest_y[frame][old_id_of(id)]
            # If dx or dy is missing, treat it as 0.
            if np.isnan(dx): dx = 0
            if np.isnan(dy): dy = 0
            __dxdy[frame][id] = [dx,dy]
    # ids is the collection of ids. xy is the collection of point
    # locations: xy(id) is the location of the id. dxdy[id] gives
    # the vector increment assigned to id.
    data = [xy, __dxdy, ids(), frames, __x_mean]
    return data

# Stage 2: data => (left_points, right_points)
def stage_2 (xy, dxdy, ids, frames, x_mean):
    def x (id):
        return xy(id)[0]
    def y (id):
        return xy(id)[1]
    def cost (id_0, id_1):
        x_0_mirrored = 2*x_mean - x(id_0)
        return norm(x_0_mirrored - x(id_1),
                    y(id_0)      - y(id_1))
    cost_matrix = [[cost(id_0,id_1) for id_0 in ids] for id_1 in ids]
    match = scipy.optimize.linear_sum_assignment(cost_matrix)[1]
    # TODO Use Hungarian Algorithm to find pairs on the face.
    def compute_left_right_ids ():
        left, right = [], []
        for id in ids:
            if (id not in left) and (id not in right):
                if   x(id) < x(match[id]):
                    left_id  = id
                    right_id = match[id]
                elif x(id) >= x(match[id]):
                    left_id  = match[id]
                    right_id = id
                left.append(left_id)
                right.append(right_id)
        return (left, right)
    left_ids, right_ids = compute_left_right_ids()
    assert len(left_ids)==len(right_ids), "Something is wrong."
    return left_ids, right_ids

def stage_3 (left_ids, right_ids, dxdy, frames):
    L, R = [], []
    for frame in frames:
        L.append(dxdy_frame_concat(dxdy, frame,  left_ids, False))
        R.append(dxdy_frame_concat(dxdy, frame, right_ids, True))
    return (L, R)

# Computations
[_xy, _dxdy, _ids, _frames, _x_mean] = stage_1()
_left_ids, _right_ids                = stage_2(_xy, _dxdy, _ids, _frames, _x_mean)
_L, _R                               = stage_3(_left_ids, _right_ids, _dxdy, _frames)

# Result
print("p value is", stats.marietan_test.marietan_T1_test(_L,_R))
