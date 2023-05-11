# Initial code for ex4.
# You may change this code, but keep the functions' signatures
# You can also split the code to multiple files as long as this file's API is unchanged 

import numpy as np
import os
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import generate_binary_structure
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import label, center_of_mass
from scipy.ndimage import map_coordinates
import shutil
from imageio import imwrite

import helper

LEVEL_ONE_OF_PYR = 0
LEVEL_THREE_OF_PYR = 2

SPREAD_OUT_RADIUS = 3
DESCRIPTOR_RADIUS = 3
DESCRIPTOR_PYR_LEVELS = 3
FILTER_SIZE = 7

VERTICAL_RECT_NUM = 7
HORIZONTAL_RECT_NUM = 7

K = 0.04
KERNEL_SIZE = 3

GRAYSCALE = 1
RGB = 2


def display_points(im1, im2, pos1, pos2):
    # create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2)

    # plot the first image in the first subplot
    ax1.imshow(im1, cmap='gray')
    for [x, y] in pos1:
        ax1.plot(x, y, color='green', linestyle="dashed", marker='o', markerfacecolor='blue', markersize=2, lw=.5)

    # plot the second image in the second subplot
    ax2.imshow(im2, cmap='gray')
    for [x, y] in pos2:
        ax2.plot(x, y, color='green', linestyle="dashed", marker='o', markerfacecolor='blue', markersize=2, lw=.5)
    # show the plot
    plt.show()


def get_derivatives(image):
    # create filters:
    ix_filter = np.array([[1, 0, -1]])
    iy_filter = np.array([[1, 0, -1]]).T

    ix_img = sol4_utils.convolve(image, ix_filter, mode='mirror')
    iy_img = sol4_utils.convolve(image, iy_filter, mode='mirror')

    return ix_img, iy_img


def all_zeros(matrix):
    for row in matrix:
        for element in row:
            if element != 0:
                return False
    return True


def create_key_points_list(list1, list2):
    return [[x, y] for x, y in zip(list1, list2)]


def harris_corner_detector(im):
    """
  Detects harris corners.
  Make sure the returned coordinates are x major!!!
  :param im: A 2D array representing an image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    # get the derivatives in X and Y of the image
    ix_img, iy_img = get_derivatives(im)

    # get the power of 2
    ix_img_pow2 = ix_img ** 2
    iy_img_pow2 = iy_img ** 2

    # multiply Ix and Iy
    ix_iy_img = ix_img * iy_img

    # blur the derivatives of the image
    ix_img_pow2 = sol4_utils.blur_spatial(ix_img_pow2, KERNEL_SIZE)
    iy_img_pow2 = sol4_utils.blur_spatial(iy_img_pow2, KERNEL_SIZE)
    ix_iy_img = sol4_utils.blur_spatial(ix_iy_img, KERNEL_SIZE)

    # compute the trace and determinant
    trace = ix_img_pow2 + iy_img_pow2
    determinant = ix_img_pow2 * iy_img_pow2 - ix_iy_img ** 2

    # compute edge measure R
    response_img = determinant - K * trace ** 2

    # incase there is no maximum points
    # if all_zeros(response_img):
    #     return np.array([])

    # compute the local maximum points and get a binary image of that local maximum points.
    response_img_max = non_maximum_suppression(response_img)

    # extract key points:
    indices = np.where(response_img_max == 1)
    # indices[0] = rows, indices[1] = cols
    corner_list = np.array(create_key_points_list(indices[1], indices[0]))

    return corner_list


def sample_descriptor(im, pos, desc_rad):
    """
  Samples descriptors at the given corners.
  :param im: A 2D array representing an image. (Added by me: its the 3rd level of a gaussian pyramid)
  :param pos: An array with shape (N,2), where pos[i,:] are the [x,y] coordinates of the ith corner point.
  :param desc_rad: "Radius" of descriptors to compute.
  :return: A 3D array with shape (N,K,K) containing the ith descriptor at desc[i,:,:].
  """
    # We'll transform the points in the pos array, which are in the 0'th level of the gaussian pyramid, into points in
    # the 3rd level of the pyramid.
    coefficient = 2 ** (LEVEL_ONE_OF_PYR - LEVEL_THREE_OF_PYR)
    # coordinates interpolation into the img coordinates, so they will fit
    new_pos = coefficient * pos

    # make a descriptor array of kxk matrices of each point radius.
    k = 1 + 2 * desc_rad
    n = len(new_pos[:, 0])
    descriptor_arr = np.empty((n, k, k))

    for i in range(n):
        # get all the points in the 7x7 radius
        x0 = new_pos[i][0]
        y0 = new_pos[i][1]
        x_range = np.arange(x0 - desc_rad, x0 + desc_rad + 1)
        y_range = np.arange(y0 - desc_rad, y0 + desc_rad + 1)
        X, Y = np.meshgrid(x_range, y_range)
        cords = np.stack((Y, X), axis=2).T

        # map_coordinates returns a list of values in the coordination's we gave that are interpolated to the image.
        val_7x7_matrix = map_coordinates(im, cords, order=1, prefilter=False).reshape((k, k))
        # normalize
        mean = np.mean(val_7x7_matrix)
        vec = val_7x7_matrix - mean
        norm = np.linalg.norm(vec)
        if norm != 0:
            descriptor_arr[i, :, :] = vec / norm
        else:
            descriptor_arr[i, :, :] = val_7x7_matrix

    return descriptor_arr


def find_features(pyr):
    """
  Detects and extracts feature points from a pyramid.
  :param pyr: Gaussian pyramid of a grayscale image having 3 levels.
  :return: A list containing:
              1) An array with shape (N,2) of [x,y] feature location per row found in the image.
                 These coordinates are provided at the pyramid level pyr[0].
              2) A feature descriptor array with shape (N,K,K)
  """
    # get feature points of the first lvl of the gaussian pyramids arguments(pyramid[0],7,7,radius = 13)
    corner_lst = spread_out_corners(pyr[LEVEL_ONE_OF_PYR], VERTICAL_RECT_NUM, HORIZONTAL_RECT_NUM, SPREAD_OUT_RADIUS)

    # get the feature descriptor array
    descriptor_arr = sample_descriptor(pyr[LEVEL_THREE_OF_PYR], corner_lst.astype(np.float64), DESCRIPTOR_RADIUS)

    return [corner_lst, descriptor_arr]


def get_second_max(matrix):
    return np.partition(matrix.flatten(), -2)[-2]


def match_features(desc1, desc2, min_score):
    """
  Return indices of matching descriptors.
  :param desc1: A feature descriptor array with shape (N1,K,K).
  :param desc2: A feature descriptor array with shape (N2,K,K).
  :param min_score: Minimal match score.
  :return: A list containing:
              1) An array with shape (M,) and dtype int of matching indices in desc1.
              2) An array with shape (M,) and dtype int of matching indices in desc2.
  """
    n1 = desc1.shape[0]
    n2 = desc2.shape[0]
    k = desc1.shape[1]

    # calculate score list
    score_lst = np.sum((desc1.reshape((n1, 1, k, k)) * desc2.reshape((1, n2, k, k))), axis=(2, 3))
    # partition each column by the second-largest value, every value under it will be placed before it,
    # and every value bigger than it will be placed after it.
    second_largest_in_cols = np.partition(score_lst, kth=-2, axis=0)[-2].reshape(1, n2)
    # do the same for the rows
    second_largest_in_rows = np.partition(score_lst, kth=-2, axis=1)[:, -2].reshape(n1, 1)

    # check if the selected feature is the two largest values in the score matrix and is bigger than 'min_score'
    # using numpy logical 'AND' and return its index
    score_lst_indices = np.argwhere(((score_lst >= second_largest_in_cols) &
                                     (score_lst >= second_largest_in_rows) &
                                     (score_lst >= min_score)))

    desc1_matches = score_lst_indices[:, 0]
    desc2_matches = score_lst_indices[:, 1]

    return [desc1_matches, desc2_matches]


def apply_homography(pos1, H12):
    """
  Apply homography to inhomogenous points.
  :param pos1: An array with shape (N,2) of [x,y] point coordinates.
  :param H12: A 3x3 homography matrix.
  :return: An array with the same shape as pos1 with [x,y] point coordinates obtained from transforming pos1 using H12.
  """
    # Add a column of ones to the pos1 array
    ones = np.ones((pos1.shape[0], 1))
    pos1 = np.hstack((pos1, ones))

    # Transform the points from pos1 with the homography matrix H12
    points = np.dot(pos1, H12.T)
    # Normalize
    points = points / (points[:, 2][:, np.newaxis])

    return points[:, :2]


def ransac_homography(points1, points2, num_iter, inlier_tol, translation_only=False):
    """
  Computes homography between two sets of points using RANSAC.
  :param points1: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 1.
  :param points2: An array with shape (N,2) containing N rows of [x,y] coordinates of matched points in image 2.
  :param num_iter: Number of RANSAC iterations to perform.
  :param inlier_tol: inlier tolerance threshold.
  :param translation_only: see estimate rigid transform
  :return: A list containing:
              1) A 3x3 normalized homography matrix.
              2) An Array with shape (S,) where S is the number of inliers,
                  containing the indices in pos1/pos2 of the maximal set of inlier matches found.
  """
    num_of_points = points1.shape[0]
    max_inlier = np.array([[]])
    max_pos1, max_pos2 = None, None

    for i in range(num_iter):
        # generate random index, if only translation = 1 index, if rigid = 2 indexes
        index = np.random.choice(num_of_points, size=2)
        if translation_only:
            pos1 = points1[index[0]].reshape((1, 2))
            pos2 = points2[index[0]].reshape((1, 2))
        else:
            pos1 = points1[index].reshape((2, 2))
            pos2 = points2[index].reshape((2, 2))

        # estimate the rigid homography between the two points
        rigid_mat = estimate_rigid_transform(pos1, pos2, translation_only)

        # apply it on points from image 1, and get estimation of points in image 2
        estimated_points2 = apply_homography(points1, rigid_mat)

        # check the estimation using squared euclidean distance
        euc = np.linalg.norm(estimated_points2 - points2, axis=1) ** 2

        # check if it is inlier or ourlier
        if not np.all((euc < inlier_tol) is False):
            inlier_matches = np.where(euc < inlier_tol)
            # check if this is the maximum number of inlier matches
            if inlier_matches[0].size > max_inlier[0].size:
                max_inlier = inlier_matches[0]
                max_pos1, max_pos2 = pos1, pos2

    # generate a rigid transformation for the best inliers found:
    final_transformation = estimate_rigid_transform(max_pos1, max_pos2, translation_only)

    return [final_transformation, max_inlier]


def display_matches(im1, im2, points1, points2, inliers):
    """
  Dispalay matching points.
  :param im1: A grayscale image.
  :param im2: A grayscale image.
  :parma points1: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im1.
  :param points2: An aray shape (N,2), containing N rows of [x,y] coordinates of matched points in im2.
  :param inliers: An array with shape (S,) of inlier matches.
  """
    # stack the images side by side
    stacked_im = np.hstack((im1, im2))
    plt.imshow(stacked_im, cmap='gray')

    # separate points into inliers and outliers based in the argument 'inliers'
    inliers_points1 = points1[inliers]
    inliers_points2 = points2[inliers]

    euc_indexes = np.arange(0, points1.shape[0])
    inliers_mask = np.isin(euc_indexes, inliers)
    outliers = euc_indexes[np.logical_not(inliers_mask)]

    outliers_points1 = points1[outliers]
    outliers_points2 = points2[outliers]

    # im2 moves right, so does its points.
    # move points of image 2 in the x-axis by image 1 number of columns
    inliers_points2[:, 0] += im1.shape[1]
    outliers_points2[:, 0] += im1.shape[1]

    # plot inlier and outliers lines and points:
    inliers_num = inliers_points1[:, 0].size
    outliers_num = outliers_points1[:, 0].size
    for j in range(outliers_num):
        plt.plot([outliers_points1[j, 0], outliers_points2[j, 0]], [outliers_points1[j, 1], outliers_points2[j, 1]],
                 mfc='r', c='b', lw=.1, ms=8, marker='.')
    for i in range(inliers_num):
        plt.plot([inliers_points1[i, 0], inliers_points2[i, 0]], [inliers_points1[i, 1], inliers_points2[i, 1]],
                 mfc='r', c='y', lw=.8, ms=10, marker='.')
    plt.show()


def accumulate_homographies(H_succesive, m):
    """
  Convert a list of succesive homographies to a 
  list of homographies to a common reference frame.
  :param H_succesive: A list of M-1 3x3 homography
    matrices where H_successive[i] is a homography which transforms points
    from coordinate system i to coordinate system i+1.
  :param m: Index of the coordinate system towards which we would like to 
    accumulate the given homographies.
  :return: A list of M 3x3 homography matrices, 
    where H2m[i] transforms points from coordinate system i to coordinate system m
  """
    M = len(H_succesive) + 1

    # Initialize a list of M 3x3 identity matrices
    h2m = [np.eye(3) for _ in range(M)]

    # Initialize accumulators
    accumulator_H, accumulator_inv_H = np.eye(3), np.eye(3)

    # accumulate homographs in the case: i < m
    for i in range(m - 1, -1, -1):
        accumulator_H = np.dot(accumulator_H, H_succesive[i])
        # re-normalize to keep the 3rd element equal to 1
        h2m[i] = accumulator_H / accumulator_H[2, 2]

    # accumulate homographs in the case: i > m
    for i in range(m, M - 1):
        accumulator_inv_H = np.dot(accumulator_inv_H, np.linalg.inv(H_succesive[i]))
        # re-normalize to keep the 3rd element equal to 1
        h2m[i + 1] = accumulator_inv_H / accumulator_inv_H[2, 2]

    return h2m


def compute_bounding_box(homography, w, h):
    """
  computes bounding box of warped image under homography, without actually warping the image
  :param homography: homography
  :param w: width of the image
  :param h: height of the image
  :return: 2x2 array, where the first row is [x,y] of the top left corner,
   and the second row is the [x,y] of the bottom right corner
  """

    # all corners of the image as [x,y] coordinates
    corner_arr = np.array([[0, 0], [w-1, 0], [0, h-1], [w - 1, h - 1]])

    # we need all the corners to compute the new corners
    # apply the homography on all the corners
    transformed_points = apply_homography(corner_arr, homography).astype(np.int)
    # get the new top left and bottom right corners
    new_corners = np.array([np.min(transformed_points, axis=0),
                            np.max(transformed_points, axis=0)]).astype(np.int)

    return new_corners


def get_coordinates(top_left, bottom_right):
    # generate an array representing the x-coordinates and y-coordinates, respectively, within the image.
    x_arr = np.arange(top_left[0], bottom_right[0] + 1)
    y_arr = np.arange(top_left[1], bottom_right[1] + 1)

    # generate 2D matrices for x-coordinates and y-coordinates x_mat[i,j] and y_mat[i,j]
    x_mat, y_mat = np.meshgrid(x_arr, y_arr)

    # stack x and y along the last axis to get a matrix of shape (n,m,2)
    return np.stack((x_mat, y_mat), axis=-1)


def warp_channel(image, homography):
    """
  Warps a 2D image with a given homography.
  :param image: a 2D image.
  :param homography: homograhpy.
  :return: A 2d warped image.
  """
    # get wrapped image top left corner and bottom right corner
    width = image.shape[1]
    height = image.shape[0]
    bounding_box = compute_bounding_box(homography, width, height)
    top_left = bounding_box[0]
    bottom_right = bounding_box[1]

    # create a box of the wrapped image x and y coordinates in a (n,m,2) matrix.
    wrapped_coords = get_coordinates(top_left, bottom_right)

    # change the matrix into [[x0,y0], [x1,y1]....] which is shape (n * m, 2)
    n, m = wrapped_coords.shape[0], wrapped_coords.shape[1]
    wrapped_coords = wrapped_coords.reshape((n * m, 2))

    # apply inverse homography
    inv_homography = np.linalg.inv(homography)
    # we get back a matrix with coordinates of the format [[x,y]]
    inv_coords = apply_homography(wrapped_coords, inv_homography)

    # wrap the image using the inverted coordinates we've found
    # map_coordinates takes a array of [[x,x,....][y,y,...]] so we need to transpose the inverted coordinates matrix
    warped_im = map_coordinates(image, inv_coords.T, order=1, prefilter=False).reshape((n, m))

    return warped_im

def warp_image(image, homography):
    """
  Warps an RGB image with a given homography.
  :param image: an RGB image.
  :param homography: homograhpy.
  :return: A warped image.
  """
    return np.dstack([warp_channel(image[..., channel], homography) for channel in range(3)])


def filter_homographies_with_translation(homographies, minimum_right_translation):
    """
  Filters rigid transformations encoded as homographies by the amount of translation from left to right.
  :param homographies: homograhpies to filter.
  :param minimum_right_translation: amount of translation below which the transformation is discarded.
  :return: filtered homographies..
  """
    translation_over_thresh = [0]
    last = homographies[0][0, -1]
    for i in range(1, len(homographies)):
        if homographies[i][0, -1] - last > minimum_right_translation:
            translation_over_thresh.append(i)
            last = homographies[i][0, -1]
    return np.array(translation_over_thresh).astype(np.int)


def estimate_rigid_transform(points1, points2, translation_only=False):
    """
  Computes rigid transforming points1 towards points2, using least squares method.
  points1[i,:] corresponds to poins2[i,:]. In every point, the first coordinate is *x*.
  :param points1: array with shape (N,2). Holds coordinates of corresponding points from image 1.
  :param points2: array with shape (N,2). Holds coordinates of corresponding points from image 2.
  :param translation_only: whether to compute translation only. False (default) to compute rotation as well.
  :return: A 3x3 array with the computed homography.
  """
    centroid1 = points1.mean(axis=0)
    centroid2 = points2.mean(axis=0)

    if translation_only:
        rotation = np.eye(2)
        translation = centroid2 - centroid1

    else:
        centered_points1 = points1 - centroid1
        centered_points2 = points2 - centroid2

        sigma = centered_points2.T @ centered_points1
        U, _, Vt = np.linalg.svd(sigma)

        rotation = U @ Vt
        translation = -rotation @ centroid1 + centroid2

    H = np.eye(3)
    H[:2, :2] = rotation
    H[:2, 2] = translation
    return H


def non_maximum_suppression(image):
    """
  Finds local maximas of an image.
  :param image: A 2D array representing an image.
  :return: A boolean array with the same shape as the input image, where True indicates local maximum.
  """
    # Find local maximas.
    neighborhood = generate_binary_structure(2, 2)
    local_max = maximum_filter(image, footprint=neighborhood) == image
    local_max[image < (image.max() * 0.1)] = False

    # Erode areas to single points.
    lbs, num = label(local_max)
    centers = center_of_mass(local_max, lbs, np.arange(num) + 1)
    centers = np.stack(centers).round().astype(np.int)
    ret = np.zeros_like(image, dtype=np.bool)
    ret[centers[:, 0], centers[:, 1]] = True

    return ret


def spread_out_corners(im, m, n, radius):
    """
  Splits the image im to m by n rectangles and uses harris_corner_detector on each.
  :param im: A 2D array representing an image.
  :param m: Vertical number of rectangles.
  :param n: Horizontal number of rectangles.
  :param radius: Minimal distance of corner points from the boundary of the image.
  :return: An array with shape (N,2), where ret[i,:] are the [x,y] coordinates of the ith corner points.
  """
    corners = [np.empty((0, 2), dtype=np.int)]
    x_bound = np.linspace(0, im.shape[1], n + 1, dtype=np.int)
    y_bound = np.linspace(0, im.shape[0], m + 1, dtype=np.int)
    for i in range(n):
        for j in range(m):
            # Use Harris detector on every sub image.
            sub_im = im[y_bound[j]:y_bound[j + 1], x_bound[i]:x_bound[i + 1]]
            sub_corners = harris_corner_detector(sub_im)
            sub_corners += np.array([x_bound[i], y_bound[j]])[np.newaxis, :]
            corners.append(sub_corners)
    corners = np.vstack(corners)
    legit = ((corners[:, 0] > radius) & (corners[:, 0] < im.shape[1] - radius) &
             (corners[:, 1] > radius) & (corners[:, 1] < im.shape[0] - radius))
    ret = corners[legit, :]
    return ret


class PanoramicVideoGenerator:
    """
  Generates panorama from a set of images.
  """

    def __init__(self, data_dir, file_prefix, num_images, bonus=False):
        """
    The naming convention for a sequence of images is file_prefixN.jpg,
    where N is a running number 001, 002, 003...
    :param data_dir: path to input images.
    :param file_prefix: see above.
    :param num_images: number of images to produce the panoramas with.
    """
        self.bonus = bonus
        self.file_prefix = file_prefix
        self.files = [os.path.join(data_dir, '%s%03d.jpg' % (file_prefix, i + 1)) for i in range(num_images)]
        self.files = list(filter(os.path.exists, self.files))
        self.panoramas = None
        self.homographies = None
        print('found %d images' % len(self.files))

    def align_images(self, translation_only=False):
        """
    compute homographies between all images to a common coordinate system
    :param translation_only: see estimte_rigid_transform
    """
        # Extract feature point locations and descriptors.
        points_and_descriptors = []
        for file in self.files:
            image = sol4_utils.read_image(file, 1)
            self.h, self.w = image.shape
            pyramid, _ = sol4_utils.build_gaussian_pyramid(image, 3, 7)
            points_and_descriptors.append(find_features(pyramid))

        # Compute homographies between successive pairs of images.
        Hs = []
        for i in range(len(points_and_descriptors) - 1):
            print(i)
            points1, points2 = points_and_descriptors[i][0], points_and_descriptors[i + 1][0]
            desc1, desc2 = points_and_descriptors[i][1], points_and_descriptors[i + 1][1]

            # Find matching feature points.
            ind1, ind2 = match_features(desc1, desc2, .7)
            points1, points2 = points1[ind1, :], points2[ind2, :]

            # Compute homography using RANSAC.
            H12, inliers = ransac_homography(points1, points2, 100, 6, translation_only)

            # Uncomment for debugging: display inliers and outliers among matching points.
            # In the submitted code this function should be commented out!
            # display_matches(self.images[i], self.images[i+1], points1 , points2, inliers)

            Hs.append(H12)

        # Compute composite homographies from the central coordinate system.
        accumulated_homographies = accumulate_homographies(Hs, (len(Hs) - 1) // 2)
        self.homographies = np.stack(accumulated_homographies)
        self.frames_for_panoramas = filter_homographies_with_translation(self.homographies, minimum_right_translation=5)
        self.homographies = self.homographies[self.frames_for_panoramas]

    def generate_panoramic_images(self, number_of_panoramas):
            """
      combine slices from input images to panoramas.
      :param number_of_panoramas: how many different slices to take from each input image
      """
            if self.bonus:
                self.generate_panoramic_images_bonus(number_of_panoramas)
            else:
                self.generate_panoramic_images_normal(number_of_panoramas)

    def generate_panoramic_images_normal(self, number_of_panoramas):
        """
    combine slices from input images to panoramas.
    :param number_of_panoramas: how many different slices to take from each input image
    """
        assert self.homographies is not None

        # compute bounding boxes of all warped input images in the coordinate system of the middle image (as given by the homographies)
        self.bounding_boxes = np.zeros((self.frames_for_panoramas.size, 2, 2))
        for i in range(self.frames_for_panoramas.size):
            self.bounding_boxes[i] = compute_bounding_box(self.homographies[i], self.w, self.h)

        # change our reference coordinate system to the panoramas
        # all panoramas share the same coordinate system
        global_offset = np.min(self.bounding_boxes, axis=(0, 1))
        self.bounding_boxes -= global_offset

        slice_centers = np.linspace(0, self.w, number_of_panoramas + 2, endpoint=True, dtype=np.int)[1:-1]
        warped_slice_centers = np.zeros((number_of_panoramas, self.frames_for_panoramas.size))
        # every slice is a different panorama, it indicates the slices of the input images from which the panorama
        # will be concatenated
        for i in range(slice_centers.size):
            slice_center_2d = np.array([slice_centers[i], self.h // 2])[None, :]
            # homography warps the slice center to the coordinate system of the middle image
            warped_centers = [apply_homography(slice_center_2d, h) for h in self.homographies]
            # we are actually only interested in the x coordinate of each slice center in the panoramas' coordinate system
            warped_slice_centers[i] = np.array(warped_centers)[:, :, 0].squeeze() - global_offset[0]

        panorama_size = np.max(self.bounding_boxes, axis=(0, 1)).astype(np.int) + 1

        # boundary between input images in the panorama
        x_strip_boundary = ((warped_slice_centers[:, :-1] + warped_slice_centers[:, 1:]) / 2)
        x_strip_boundary = np.hstack([np.zeros((number_of_panoramas, 1)),
                                      x_strip_boundary,
                                      np.ones((number_of_panoramas, 1)) * panorama_size[0]])
        x_strip_boundary = x_strip_boundary.round().astype(np.int)

        self.panoramas = np.zeros((number_of_panoramas, panorama_size[1], panorama_size[0], 3), dtype=np.float64)
        for i, frame_index in enumerate(self.frames_for_panoramas):
            # warp every input image once, and populate all panoramas
            image = sol4_utils.read_image(self.files[frame_index], 2)
            warped_image = warp_image(image, self.homographies[i])
            x_offset, y_offset = self.bounding_boxes[i][0].astype(np.int)
            y_bottom = y_offset + warped_image.shape[0]

            for panorama_index in range(number_of_panoramas):
                # take strip of warped image and paste to current panorama
                boundaries = x_strip_boundary[panorama_index, i:i + 2]
                image_strip = warped_image[:, boundaries[0] - x_offset: boundaries[1] - x_offset]
                x_end = boundaries[0] + image_strip.shape[1]
                self.panoramas[panorama_index, y_offset:y_bottom, boundaries[0]:x_end] = image_strip

        # crop out areas not recorded from enough angles
        # assert will fail if there is overlap in field of view between the left most image and the right most image
        crop_left = int(self.bounding_boxes[0][1, 0])
        crop_right = int(self.bounding_boxes[-1][0, 0])
        assert crop_left < crop_right, 'for testing your code with a few images do not crop.'
        print(crop_left, crop_right)
        self.panoramas = self.panoramas[:, :, crop_left:crop_right, :]

    def generate_panoramic_images_bonus(self, number_of_panoramas):
        """
    The bonus
    :param number_of_panoramas: how many different slices to take from each input image
    """
        pass

    def save_panoramas_to_video(self):
        assert self.panoramas is not None
        out_folder = 'tmp_folder_for_panoramic_frames/%s' % self.file_prefix
        try:
            shutil.rmtree(out_folder)
        except:
            print('could not remove folder')
            pass
        os.makedirs(out_folder)
        # save individual panorama images to 'tmp_folder_for_panoramic_frames'
        for i, panorama in enumerate(self.panoramas):
            imwrite('%s/panorama%02d.png' % (out_folder, i + 1), panorama)
        if os.path.exists('%s.mp4' % self.file_prefix):
            os.remove('%s.mp4' % self.file_prefix)
        # write output video to current folder
        os.system('ffmpeg -framerate 3 -i %s/panorama%%02d.png %s.mp4' %
                  (out_folder, self.file_prefix))

    def show_panorama(self, panorama_index, figsize=(20, 20)):
        assert self.panoramas is not None
        plt.figure(figsize=figsize)
        plt.imshow(self.panoramas[panorama_index].clip(0, 1))
        plt.show()
