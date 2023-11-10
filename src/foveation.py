"""
Methods for sampling points and creating a foveated image
"""
import numpy as np

"""
How do we want to represent the foveation?
1. Set of sample points 
    (Option 1) Low number (~16) of sample points
         At each sample point, take an image sample (at some resolution) and assign it a position encoding
    (Option 2) High number of sample points
        At each sample point, take a max pool. (these are more pixelwise samples)
2. CNN style 
    Define the kernel size, step and stride, at several different scales

    (Essentially a CNN that is parametrized such that the step and stride are 
    variable and do not cover the entire image + positional encodings to keep things straight)
"""
# How do we want to represent the foveation?
# 1. Set of sample points
#     1.1 At each sample point, take an image patch (at some resolution) and assign it a


#### Sampling Techniques ####
def sample_geometric_sequence(center_x, center_y, nx, ny, width, height):
    """
    Input:
    Center pixel coordinates of the foveated region
    nx, ny: number of points to sample in the +x and +y directions
    width: distance between leftmost and rightmost sampled points
    height: distance between topmost and bottommost sampled points
    Ouput: (2*nx+1 length np.array of x pixel coordinates, 2*ny+1 length np.array of y pixel coordinates)
    """
    x_points = np.zeros(2 * nx + 1)
    y_points = np.zeros(2 * ny + 1)
    x_points[-1] = center_x
    y_points[-1] = center_y

    # Generate geometric sequence of points
    rx = 2
    ry = 2
    a0_x = width / 2.0 * (1 - rx) / (1 - rx ** (nx))
    a0_y = height / 2.0 * (1 - ry) / (1 - ry ** (ny))
    a_x = 0
    a_y = 0
    for i in range(nx):
        a_x += a0_x * (rx**i)
        x_points[i] = center_x + a_x
        x_points[i + nx] = center_x - a_x
    for i in range(ny):
        a_y += a0_y * (ry**i)
        y_points[i] = center_y + a_y
        y_points[i + ny] = center_y - a_y

    x_sorted_integers = np.sort(x_points.astype(int))
    y_sorted_integers = np.sort(y_points.astype(int))
    return x_sorted_integers, y_sorted_integers


def sample_uniform(center_x, center_y, nx, ny, width, height):
    """
    Sample uniformly across the entire region (used for the base pooling)
    """
    x_points = np.linspace(center_x - width / 2, center_x + width / 2, nx)
    y_points = np.linspace(center_y - height / 2, center_y + height / 2, ny)
    return x_points.astype(int), y_points.astype(int)


def sample_trapezoidal(center_x, center_y, nx, ny, width, height):
    """
    Sample a uniform distribution in the center region (width/2, height/2),
    then a linear decay for the outer region
    """
    proportion_inv = 2
    outer_points_x, outer_points_y = sample_uniform(
        center_x, center_y, nx // 2, ny // 2, width, height
    )
    # remove from the outer points any that are within the inner width/2, height/2 region
    outer_points_x = outer_points_x[
        abs(outer_points_x - center_x) < width // proportion_inv
    ]
    outer_points_y = outer_points_y[
        abs(outer_points_y - center_y) < height // proportion_inv
    ]
    nx_inner = nx - len(outer_points_x)
    ny_inner = ny - len(outer_points_y)
    center_points_x, center_points_y = sample_uniform(
        center_x,
        center_y,
        nx_inner,
        ny_inner,
        width // proportion_inv,
        height // proportion_inv,
    )
    x_points = np.concatenate((outer_points_x, center_points_x))
    y_points = np.concatenate((outer_points_y, center_points_y))
    x_sorted_integers = np.sort(x_points.astype(int))
    y_sorted_integers = np.sort(y_points.astype(int))
    return x_sorted_integers, y_sorted_integers


def sample_gaussian(center_x, center_y, nx, ny, width, height):
    """
    Deterministically Generate x and y points from 1D Gaussian distributions
    the covariance is [[width, 0], [0, height]]
    """
    pass
    # uniform_points_x = np.linspace(0, 1, nx)
    # uniform_points_y = np.linspace(0, 1, ny)

    # # Apply the inverse cumulative distribution function (ICDF) of the normal distribution
    # # This uses the percent point function (PPF) of the normal distribution
    ## from scipy.special import erfinv

    # # x_points = center_x + width * np.sqrt(2) * erfinv(2 * uniform_points_x - 1)
    # # y_points = center_y + height * np.sqrt(2) * erfinv(2 * uniform_points_y - 1)
    # x_sorted_ints = np.sort(x_points.astype(int))
    # y_sorted_ints = np.sort(y_points.astype(int))
    # return x_sorted_ints, y_sorted_ints


def sample_polar(center_x, center_y, nx, ny, width, height):
    """
    Generate nx*ny points in a polar pattern
    """
    pass


#### Foveation Techniques ####
def max_pool_image(img_array, factor):
    """
    Input:
    img_array: (width, height, 3) numpy array of the original image
    factor: factor by which to downsample the image
    Output:
    numpy array of the downsampled image
    """
    # Calculate the dimensions of the downsampled image
    downsampled_width = img_array.shape[0] // factor
    downsampled_height = img_array.shape[1] // factor

    # Reshape the image array and perform max pooling using strides
    downsampled_array = img_array.reshape(
        downsampled_width, factor, downsampled_height, factor, 3
    ).max(axis=(1, 3))

    return downsampled_array


def max_pool_foveation(points_x, points_y, img_array):
    """
    At each sample point, take a max pool of the surrounding pixels
    (size of patch is proportional to distance from fovea center)
    Input:
    points_x and points_y - ordered sample points from sample_geometric_sequence
    img_array - (width, height, 3) numpy array of the foveated image (will be modified)
    Output:
    (width, height, 3) numpy.array of Foveated image
    """
    width = img_array.shape[0]
    height = img_array.shape[1]

    output_img_array = np.zeros((width, height, 3))  # Will modify this one

    # Decimate image by only taking every n-th pixel
    decimation_factor = 7
    output_img_array[::decimation_factor, ::decimation_factor] = img_array[
        ::decimation_factor, ::decimation_factor, :
    ]

    # Add high-resolution fovea
    for i, px in enumerate(points_x):
        for j, py in enumerate(points_y):
            # patch radius is half the distance between the
            # current point and the next point towards the center (or 0.5)
            patch_scale = 0.5
            if i < len(points_x) / 2:
                patch_radius_x = int(abs(px - points_x[i + 1]) * patch_scale)
            else:
                patch_radius_x = int(abs(px - points_x[i - 1]) * patch_scale)
            if j < len(points_y) / 2:
                patch_radius_y = int(abs(py - points_y[j + 1]) * patch_scale)
            else:
                patch_radius_y = int(abs(py - points_y[j - 1]) * patch_scale)

            x_min = max(0, px - patch_radius_x)
            y_min = max(0, py - patch_radius_y)
            x_max = min(width, px + patch_radius_x)
            y_max = min(height, py + patch_radius_y)
            if x_min >= x_max or y_min >= y_max:
                continue

            max_pool = np.max(img_array[x_min:x_max, y_min:y_max, :], axis=(0, 1))
            output_img_array[x_min:x_max, y_min:y_max, :] = max_pool

    return output_img_array

def direct_foveation(points_x, points_y, img_array):
    """
    Sample only the given points, and sparsely from the rest of the image
    """
    width = img_array.shape[0]
    height = img_array.shape[1]

    output_img_array = np.zeros((width, height, 3))  # Will modify this one

    # Decimate image by only taking every n-th pixel
    decimation_factor = 7
    output_img_array[::decimation_factor, ::decimation_factor] = img_array[
        ::decimation_factor, ::decimation_factor, :
    ]

    # Add high-resolution fovea
    for i, px in enumerate(points_x):
        for j, py in enumerate(points_y):
            if px < width and py < height and px >= 0 and py >= 0:
                output_img_array[px, py, :] = img_array[px, py, :]

    return output_img_array


# CNN Foveation
# FoveaTR foveation
