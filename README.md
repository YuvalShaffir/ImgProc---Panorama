# ImgProc---Panorama

This is the 4th excercise of the Image Proccessing course. Here we learned to create a panorama image from a group of images.

Functionalities:
- Harris corner detector
- Descriptor - samples descriptors in the corners found
- Match two images by their descriptors
- Computes homography between two sets of points using RANSAC
- Convert a list of succesive homographies to a 
  list of homographies to a common reference frame
- Compute the bounding box of warped image under homography
- Filters rigid transformations encoded as homographies by the amount of translation from left to right.
- Finds local maximas of an image.
- Generate panorama.
