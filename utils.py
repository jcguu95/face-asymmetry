import math

def distance (array_floats):
    '''An util function: Euclidian distance.'''
    result = 0
    for float in array_floats:
        result += float**2
    return result**(1/2)

assert(distance([3,4])==5.0)

def scalar_mul (scalar, list):
    '''An util function.'''
    return [scalar * x for x in list]

assert(scalar_mul(3, [1,2]) == [3,6])

def vect_minus (v0, v1):
    '''An util function.'''
    return list(map((lambda x,y: x-y), v0, v1))

def angle_diff (v0, v1):
    '''An util function: calculate the angle difference between
    the vector v0 and v1.'''
    nv0 = scalar_mul(1/distance(v0), v0)
    nv1 = scalar_mul(1/distance(v1), v1)
    L = distance(vect_minus(nv0, nv1))
    d_theta = math.acos(1 - L*L/2)
    return d_theta

assert(angle_diff([0,1], [1,1])/math.pi == 0.25)
