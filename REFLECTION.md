# Adaptive Segmentation
## update_samples()
Noticeable segmentation difference/performance when lighting changes! Shuffle + Otsu f.ex. works well when lighting changes in the scene, whereas the static sampling is less reactive when the lighting changes

Other example of this can be seen here:
https://docs.opencv.org/4.x/d7/d4d/tutorial_py_thresholding.html

Also mentioned in the DARPA Grand Challenge Paper from 2005, where the "adaptive approach is necessary" (section 1.6 Computer Vision Terrain Analysis)

## Effect of changing update_ratio
TODO understand better, didn't really notice anything with video 4 - other than the fact that it takes more resources as we notice a slight lag in the viz




