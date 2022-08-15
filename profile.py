# Usage :: Manually mount
#   `/media/pawel/disk12T/FaceAnalysis/fm_annot/img/` to `./data/mnt/`
# with the shell command
#
#   `sshfs jinchengguu@129.49.109.97:/media/pawel/disk12T/ ./data/mnt` and run

def pawel_orig_example ():
    '''Returns the profile for pawel original examples.'''
    # Aux variables.
    directory = "./data/mnt/FaceAnalysis/fm_annot/img/disp/"
    filename_fmt = "disp_frame{frame}_pointf_ref"
    id_min, id_max = 0, 673
    frame_min, frame_max = 60, 1320
    #
    profile = \
      {'name'    : "Pawel original examples.",
       'doc'     : "The very first example we used for analyzing asymmetry with the current code.",
       'get_file': lambda frame: directory+filename_fmt.format(frame = frame),
       'ids'     : range(id_min,id_max),
       'frames'  : range(frame_min,frame_max,10)}
    return profile

def ellison_1 ():
    '''Returns the profile for Ellison-1.'''
    # Aux variables.
    directory = "./data/mnt/jin_cheng_guu/Ellison-asymmetry-stuff/Ellison-1/pre_out/disp/"
    filename_fmt = "disp_frame{frame}_pointf_ref"
    id_min, id_max = 0, 1501
    frame_min, frame_max = 1, 10 #16
    #
    profile = \
      {'name'    : "Ellison 1",
       'doc'     : "Truncated data for Ellison 1. Originally it has 16 frames, \
                    but we only take the first 10 in order to meaningfully \
                    compare with Ellison 2 and 3.",
       'get_file': lambda frame: directory+filename_fmt.format(frame = frame),
       'ids'     : range(id_min,id_max),
       'frames'  : range(frame_min,frame_max,1)}
    return profile

def ellison_2 ():
    '''Returns the profile for Ellison-2.'''
    # Aux variables.
    directory = "./data/mnt/jin_cheng_guu/Ellison-asymmetry-stuff/Ellison-2/pre_out/disp/"
    filename_fmt = "disp_frame{frame}_pointf_ref"
    id_min, id_max = 0, 1501
    frame_min, frame_max = 1, 10 #11
    #
    profile = \
      {'name'    : "Ellison 2",
       'doc'     : "Truncated data for Ellison 2. Originally it has 11 frames, \
                    but we only take the first 10 in order to meaningfully \
                    compare with Ellison 1 and 3.",
       'get_file': lambda frame: directory+filename_fmt.format(frame = frame),
       'ids'     : range(id_min,id_max),
       'frames'  : range(frame_min,frame_max,1)}
    return profile

def ellison_3 ():
    '''Returns the profile for Ellison-3.'''
    # Aux variables.
    directory = "./data/mnt/jin_cheng_guu/Ellison-asymmetry-stuff/Ellison-3/pre_out/disp/"
    filename_fmt = "disp_frame{frame}_pointf_ref"
    id_min, id_max = 0, 1501
    frame_min, frame_max = 1, 10
    #
    profile = \
      {'name'    : "Ellison 3",
       'doc'     : "data for Ellison 3",
       'get_file': lambda frame: directory+filename_fmt.format(frame = frame),
       'ids'     : range(id_min,id_max),
       'frames'  : range(frame_min,frame_max,1)}
    return profile
