# global modules
import numpy as np
import functools

# local modules
import profile

### User Configuration

### Set the global variable CURRENT_PROFILE before running the
### code. This tells the program where to look for data, and what
### kinds of data to expect.
###
### For performance issue, we cache some functions in this
### program. It is safer to reload everything if you want to
### switch profile.

#CURRENT_PROFILE = profile.pawel_orig_example()
#CURRENT_PROFILE = profile.ellison_1()
CURRENT_PROFILE = profile.ellison_2()
#CURRENT_PROFILE = profile.ellison_3()

def get_file (frame):
    return CURRENT_PROFILE['get_file'](frame)

def ids ():
    return CURRENT_PROFILE['ids']

def frames ():
    return CURRENT_PROFILE['frames']

@functools.cache
def content (frame):
    result = {}
    input = np.load(get_file(frame), allow_pickle=True)
    # change the following if data format changes
    result['idx'] = input['idx']
    result['pointi'] = input['pointi']
    result['pointf'] = input['pointf']
    # Check sanity
    assert(len(result['idx']) == len(result['pointi']) == len(result['pointf']))
    return result

@functools.cache
def extract (frame):
    result = {}
    for k in range(len(content(frame)['idx'])):
        (x0,y0) = (content(frame)['pointi'][k][0], content(frame)['pointi'][k][1])
        (x1,y1) = (content(frame)['pointf'][k][0], content(frame)['pointf'][k][1])
        (dx,dy) = (x1-x0, y1-y0)
        result[content(frame)['idx'][k]] = {'x0': x0, 'y0': y0, 'dx': dx, 'dy': dy}
    for id in ids():
        if not id in content(frame)['idx']:
            result[id] = np.nan
    return result

@functools.cache
def data (frame, id):
    ef = extract(frame)
    if ef:
        return extract(frame)[id]
    else:
        return np.nan
