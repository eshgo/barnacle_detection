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

## Creating data.py, which loads the images and turns the masks into easily countable barnacles

- `DATA_DIR` and `_resolve` helps make the file paths to the images easier, so you just have to type in the filename instead of the whole path
- `load_image` converts BGR for OpenCV to RGB for everything else
- `describe_file` describes the image for me so I have a better idea of what is going on before I wrote later steps
- after running `describe_file` on `mask1`, i found that the background white of the image has alpha=0, meaning it is transparent. This makes my life easier because to detect for the circles in the image i can just serach for pixels with alpha>0
- `load_mask_lines` looads an array of true or false for whether the pixel is a contour or not. It seems to work because most of the pixel values are false.
- `mask_to_regions` is supposed to convert the masks to regions of each barnacle so they can be counted. There are some floating blue pixels that aren't a circled barnacle, and I don't want those to get counted as a barnacle. I tried to look inside each circle for empty pixels, and record these pockets of empty pixels within circles as barnacles. However, it counts small bubbles inside the drawn circle as another contour, and it doesn't count an outline if it has a small gap.
- I rewrote `mask_to_regions` using `cv2.findContours` so that small bubbles inside circles aren't counted as their own barnacle. However, I still needed account for the floating pixels and contours with gaps.

## Iterating over mask_to_regions

- My first version ignored the lines and found the empty regions they enclose, one pocket per barnacle. I chose this because I assumed outlines would touch each other, and tracing touching lines would merge them. Looking at the overlays, outlines don't actually touch. The real problems were floating specks and outlines with small gaps. A gapped outline encloses no pocket, so those barnacles silently disappeared.
- Switched to tracing the outlines directly. Removed specks by size, used a morphological close to bridge small gaps, and fell back to the convex hull for outlines with gaps too big to bridge. A RETR_CCOMP hierarchy tells closed outlines (they have a hole inside) from open ones. Barnacles that needed the hull are flagged so I can count them.
- In dense areas, barnacles were disappearing. The close ran on the whole image, so it bridged the space between nearby outlines as well as gaps within them, fusing neighbors into one shape. On mask1 I saw big clusters of contours only counted as a few dots.
- Since outlines don't touch, each connected cluster of line pixels is one barnacle. Now each cluster is cut out and closed on its own, so it has no neighbors to merge with.

