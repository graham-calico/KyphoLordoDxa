[Back to home.](../README.md)

# Get Started: Running the Analysis

These python scripts use locally-referenced data files that encode the ML
models that are being applied, so they should be run in-place
from a clone of this repo.  They are [here](developer.md).

## Scoring DEXA images using a directory

The estimation of Cobb angles from either a directory of images or a text file that lists the
image files to be scored, one per line:

```shell
KYPHOSIS:$ python spineCurve.py -i ${INPUT_DIR} -o ${RESULT_FILE} [optional args]
LORDOSIS:$ python spineCurve.py -i ${INPUT_DIR} -o ${RESULT_FILE} --lumbar [optional args]
```

**Input/output arguments:**
- `-i INPUT_DIR`: a directory of source images.  The file names, minus appends, will be
  used as the identifiers in the output file.  Corrupted or non-image files and
  sub-directories will be skipped over.
- `-o RESULT_FILE`: a two-column, tab-delimited file with DEXA photo ID's in the left
  column and estimated Cobb angles in the right.  Specifying "stdout" will result in
  results being written to standard out.

**Optional input/output:**
- `-d DRAWING_DIR`: if specified, then marked-up versions of the input images showing
  some aspects of the image analytics will be written into this directory.

**Optional behavioral arguments:**
- `--lumbar`: invoked to switch from analysis of kyphosis (i.e. forward curvature of the thoracic
  spine; default) to lordosis (i.e. backward curvature of the lumbar spine).
- `--side_facing SIDE_FACING`: if the direction that the person is facing in all images
  is known, this flag can specify that direction (`left` or `right`).  Otherwise, an ML
  model will be applied to determine the direction.
- `--box_fraction BOX_FRACTION`: for either the thoracic or lumbar box (i.e. the portion of the
 spine across which kyphosis or lordosis is measured), this parameter determines the fraction
  of the box's length across which angle-determining vectors will be measured (yellow lines in
  Diagrams 6 & 7, in [Methods](analysis.md)). <i>Allowed</i>: between & not including 0 and 1.  <i>Recommended</i>:
  less than 0.5.  <i>Default</i>: exactly 1/3.
- `--thoracic_bottom THORACIC_BOTTOM`: which vertebral disc defines the bottom of the thoracic
  spine (see [Methods](analysis.md) Diagram 2 for details).  This defines where the mid-spine edge of
  the region of interest will be (see [Methods](analysis.md) Diagram 3). <i>Allowed</i>: range 1-14, with vertebral
  discs 1-indexed from the top of the spine/image. Fractional values are allowed: they will indicate a placement
  that fraction of the distance between the adjacent integers' disc positions.  <i>Default</i>: 11.5 for kyphosis;
  10.25 for lordosis.
- `--bottom_cut BOTTOM_CUT`: option to cut some fraction off the bottom of the region-of-interest.  This can be used
  to avoid confusion around the tail bone.  <i>Allowed</i>: 0-1, including 0 but not 1.
  <i>Recommended</i>: less than 0.2.  <i>Default</i>: 0.
- `--legacy`: if invoked, the program will behave as originally programmed/used to derive the values in the accompanying
  scientific manuscript.  Note that, among test data from the UK BioBank, the values calculated with or without this flag,
  for both kyphosis and lordosis, without adding any data augmentation (flags described below), 
  correlated with Pearson coefficients >0.9995.

**Augmentation option arguments:** 
- `--aug_flip`: will flip each image horizontally and repeat the analysis, and output the
  average of the flipped and non-flipped angles.
- `--aug_tilt AUG_TILT`: addionally score each image tilted in each direction (i.e. each invocation of this flag adds
  two analyses) by the specified number of degrees, rotating the image around its center point.  Values must be above
  0 and below 45, and recommended values are less than 5.  The returned value will be a weighted average across all
  tilted images, with the weight of each image equal to 0.1^(4 * <i>r</i>), where <i>r</i> is the rotational angle in <i>radians</i>.
  This argument can be invoked multiple times, for rotational augmentation at multiple different angles.  It can also be
  invoked in conjunction with `--aug_flip`, in which case tilt augmentation will be applied to both horizontally-flipped
  images.

## Evaluating performance versus pre-scored DEXA images

An easy way to get statistics quickly for performance versus a small set of 
your own annotations:

```shell
python spineCurve.py -a ${ANNOT_FILE} [--aug_flip/--aug_one]
```
**Invoking argument:**
- `-a ANNOT_FILE`: an alternative calling method: input a text file with two tab-separated
  columns (left: file name path; right: annotated kyphosis or lordosis angles), and this script
  will apply the angle algorithm to those images and print a statistical analysis
  of performance versus the annotations to stdout.
