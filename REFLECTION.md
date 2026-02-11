# Lab
This lab is inspired by the DARPA Grand Challenge paper

# Adaptive Segmentation
## update_samples()
Noticeable segmentation difference/performance when lighting changes! Shuffle + Otsu f.ex. works well when lighting changes in the scene, whereas the static sampling is less reactive when the lighting changes

Other example of this can be seen here:
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

Also mentioned in the DARPA Grand Challenge Paper from 2005, where the "adaptive approach is necessary" (section 1.6 Computer Vision Terrain Analysis)

## Effect of changing update_ratio
TODO understand better, didn't really notice anything with video 4 - other than the fact that it takes more resources as we notice a slight lag in the viz


# Mahalanobis
Using mahalanobis as a distribution/processed image, then thresholding that



# Morphological Transformations (inside perform_segmentation())
https://docs.opencv.org/4.x/d9/d61/tutorial_py_morphological_ops.html

## Effect of larger kernel?

## Effect of Opening
Less noise from side of road (vid 1)

## Effect of Closing
Bigger shapes with more consequent fill for road, looks cleaner, but more stuff on the side of the road (grass small shapes and noise) (vid 1)

## Test with Gradient
Did not work well with road shape (vid 1) 
