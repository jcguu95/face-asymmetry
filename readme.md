# Face Asymmetry

This project uses higher-dimensional statistics to test if a
patient's face muscle movements are fundamentally asymmetric. The
core statistical test we use is Marietan's "T1-test".

In `main.py`, you can find an example of how to use the
utilities. In particular, there are three stages of
transformations from raw data (residing in `/data/`) to the final
p-value.

``` python
[_xy, _dxdy, _ids, _frames, _x_mean] = stage_1()
_left_ids, _right_ids                = stage_2(_xy, _dxdy, _ids, _frames, _x_mean)
L, R                                 = stage_3(_left_ids, _right_ids, _dxdy, _frames)
print("p value is", stats.marietan_test.marietan_T1_test(L,R))
```

In the first stage (see `stage_1()`), we transform the raw data
to data: `xy, dxdy, ids, frames, x_mean`. Here, `xy` encodes the
(x,y) coordinates of each id, `dxdy` encodes the smoothed vectors
at any given frame and given id, `ids` collect the available ids,
`frames` collect the available frames, and `x_mean` is the mean
of the x-coordinates of all the points.

In the second stage (see `stage_2()`), the main goal is to obtain
a matching between the points on the left-half of the face and
the points on the right-half of the face. We use Hungarian
algorithm to achieve this. The outputs of `stage_2` are two lists
`_left_ids` and `_right_ids`. The first list encodes the ids that
belong to the left-half of the face, while the second does so for
the ids that belong to the right-half of the face. That means
`_left_ids[j]` is always on the left-half of the face, and its
best mirrored point is `_right_ids[j]`.

In the third stage (see `stage_3()`), we prepare the Left Matrix
`L` and the Right Matrix `R`. We first explain content of `L`. In
our data set (under `/data/`), there are `10` frames and `3000`
points (roughly, lets pretend it is accurate). `L` contains `10`
vectors of the same size. Each vector of a given frame is just
the concatenation of all the `[dx, dy]`'s of all points on the
left face, following the order given by `_left_ids`. The matrix
`R` is almost the same, except that its `dx` are all multiplied
by `-1`. 

Then we apply `marietan_T1_test()` to `L` and `R` and obtain the
`p_value`. If the `p_value` is less than `0.05`, we can
statistically conclude that the face muscle movements are
fundamentally asymmetric.
