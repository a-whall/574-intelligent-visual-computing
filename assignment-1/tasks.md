# Assignment 1: Basic Surface Reconstruction from Point Clouds
In this assignment you will implement two basic implicit-surface reconstruction algorithms to approximate a surface represented by scattered point data.

---
# Overview

The problem can be stated as follows:

Given a set of points `P = { p1, p2, ... , pn }` in a point cloud, we will define an implicit function `f(x, y, z)` that measures the signed-distance to the surface approximated by these points. The surface is extracted at `f(x, y, z) = 0` using the marching cubes algorithm.

All you need to implement are two implicit functions that measure distance in the following ways: 
1. Signed-distance to tangent plane of the surface point nearest to each point `(x, y, z)` of the grid storing the implicit function.
2. Moving Least Squares (MLS) distance to tangent planes of the K nearest surface points to each point (x,y,z) of the grid storing the implicit function.

The scikit-image package already provides implementations of marching cubes. Thus, all you need to do is to fill the code in the script 'basicReconstruction.py' to implement the above implicit functions. The implicit functions rely on surface normal information per input surface point.

In the provided test data files, surface normals are included. The format of the point cloud file is:
- `x_coordinate y_coordinate z_coordinate x_normal y_normal z_normal`

Three test point-cloud files are included with this assignment:
- bunny-500.pts
- bunny-1000.pts
- sphere.pts

To run the code, you need the following packages installed:
- open3d
- skimage
- sklearn

---
# Tasks

## Task A (30%):
One way to estimate the signed distance of any point `p = (x, y, z)` of the 3D grid to the sampled surface points `pi` is to compute the distance of `p` to the tangent plane of the surface point `pi` that is nearest to `p`. In this case, your signed distance functon is:

`f(p) = nj · (p - pj), where j = argmin_i(||p - pi||)`

Implement this distance function in the naiveReconstruction function of the script `basicReconstruction.py`.

Show screenshots of the reconstructed bunny (500 and 1000 points) and sphere in your report.

---

## Task B (70%):
The above scheme results in a [C0](https://en.wikipedia.org/wiki/Smoothness#Differentiability_classes) surface, i.e. the surface is continuous, but the derivatives of the implicit function are not continuous, if they were then the surface would be called "C1-continuous", if the second derivatives were continuous it would be called "C2-continuous", and so on.

To get a smoother result, the [Moving Least Squares](https://en.wikipedia.org/wiki/Moving_least_squares) (MLS) distance from tangent planes is more preferred. The MLS distance is defined as the weighted sum of the signed distance functions to all points `pi`:

`f(p) = Σi di(p) φ(||p-pi||) / Σi φ(||p - pi||)`

`where:`

`di(p) = ni · (p - pi)`

`φ(||p - pi||) = exp(-||p - pi||^2 / β^2)`

Practically, computing the signed distance function to all points `pi` is computationally expensive. Since the weights `φ(||p - pi||)` become very small for surface sample points that are distant to points `p` of the grid, in your implementation you will compute the MLS distance to the `K`-nearest-neighboring surface points for each grid point, with `k = 50`.

Implement this distance function in the `mlsReconstruction` function of the script `basicReconstruction.py`. You will also need to compute an estimate of `1/β^2`.

Set `β` to be twice the average of the distances between each surface point and its closest neighboring surface point.

Show screenshots of the reconstructed bunny (500 and 1000 points) and sphere in your report.

---