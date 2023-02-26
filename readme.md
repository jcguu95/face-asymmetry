# Face Asymmetry

This undertaking employs advanced statistical techniques in
higher dimensions to evaluate the fundamental asymmetry of a
patient's facial muscle movements. The primary statistical test
utilized is Marietan's "T1-test."

An illustration of the utility usage is present in the `main.py`
file. The application involves three distinct stages of data
transformations, commencing from the raw data situated in the
`/data/` directory, culminating in the computation of the final
p-value.

``` python
[_xy, _dxdy, _ids, _frames, _x_mean] = stage_1()
_left_ids, _right_ids                = stage_2(_xy, _dxdy, _ids, _frames, _x_mean)
L, R                                 = stage_3(_left_ids, _right_ids, _dxdy, _frames)
print("p value is", stats.marietan_test.marietan_T1_test(L,R))
```

The initial phase (refer to `stage_1()`) entails converting the
raw data into a more usable format, which includes `xy`, `dxdy`,
`ids`, `frames`, `x_mean`. In this context, `xy` represents the
`(x,y)` coordinates of each `id`, `dxdy` indicates the smoothed
vectors corresponding to any given frame and id, `ids` records
the available ids, `frames` gathers the available frames, and
`x_mean` denotes the average x-coordinate value of all the points.

In the second stage (see `stage_2()`), the main goal is to obtain
a matching between the points on the left-half of the face and
the points on the right-half of the face. We use Hungarian
algorithm to achieve this. The outputs of `stage_2` are two lists
`_left_ids` and `_right_ids`. The first list encodes the ids that
belong to the left-half of the face, while the second does so for
the ids that belong to the right-half of the face. That means
`_left_ids[j]` is always on the left-half of the face, and its
best mirrored point is `_right_ids[j]`.

Subsequently, in the second stage (refer to `stage_2()`), the
primary objective is to establish a correspondence between the
left and right half points of the face. This is accomplished
using the Hungarian algorithm. The outcomes of `stage_2` are two
lists, namely `_left_ids` and `_right_ids`. The first list contains
the ids of the points on the left-half of the face, while the
second list contains the ids of the points on the right-half of
the face. Specifically, `_left_ids[j]` represents the point on the
left-half of the face, and its corresponding mirrored point is
`_right_ids[j]`.

In the third stage (refer to `stage_3()`), we create two
matrices, the Left Matrix `L` and the Right Matrix `R`. `L`
contains ten vectors of the same size, one for each frame. Each
vector corresponds to a particular frame and comprises the
concatenation of all `[dx, dy]` pairs for all the points on the
left side of the face, as per the order specified by `_left_ids`.
The Right Matrix `R` is similar to `L`, except that all its `dx` values
are multiplied by `-1`.

Finally, we use the `marietan_T1_test()` function to evaluate `L`
and `R` and obtain the `p_value`. If the computed `p_value` is
less than `0.05`, it implies that the face muscle movements are
significantly asymmetrical, based on statistical analysis.
