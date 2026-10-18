# Lab Log

Notes on observations, experiments, and failures.

## Learning Resources

- [Cropping an Image Using OpenCV](https://opencv.org/cropping-an-image-using-opencv/)
- [scikit-learn RandomForestClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)
- [Kernel (image processing)](https://en.wikipedia.org/wiki/Kernel_(image_processing))

## Initial Image Observations

- Barnacles often have an eye-like shape on top.
- Some barnacles are sideways, so the opening on top is not clearly visible.
- Barnacles vary in size.
- The frames are both green, but their shades differ slightly. This should be manageable.
- Barnacles in the two photos differ in color.
- `unseen_img2` does not have a green frame. If the detector cannot find one to crop, it should leave the image uncropped.
- `unseen_img1` looks similar to `img1`.
- `unseen_img2` looks straightforward; there do not appear to be any ambiguous barnacles.
