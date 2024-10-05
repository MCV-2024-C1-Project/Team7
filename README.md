# Master Computer Vision, Module C1
# Week 1

## Code execution
### Install python and pip
```
sudo apt install python3
```
```
sudo apt install python3-pip
```

### Install dependencies
```
pip install -r requirements.txt
```

### Generate histograms
```
python create_histograms.py
```

### Run the main code
The arguments shown in the following command line are the ones by default. Use ``python main.py -h`` to see all the possible arguments.
```
main.py \
--queries-hist-dir qst1_w1 \
--color-space LAB \
--similarity-measure Correlation \
--k-val 10 \
--results-file result.pkl
```

## This week's task
The goal of this week's task is to match pictures of paintings from a large image database with other image queries that feature those artworks. The resemblance between each query and each image in the database is assessed based on their visual content; specifically, the similarity between their histograms.

### Task 1
In task one, up to two methods could be chosen for computing the image descriptors (histograms). We decided to calculate all the histograms in advance for the following color representations: RGB, CIELAB, HSV, YCrCb, and Grayscale, so we could experiment if we wanted to. After the process of creation, the histograms are saved in pickle files. This is done by executing the ``create_histograms.py`` script and having the databases in the ``data`` directory.

It is required to have downloaded the three databases (BBDD, qsd1_w1, qst1_w1) and to have placed them in ``data`` in order to execute the script. After the execution, the files will be distributed as the following:
```
data/histograms/<DataBaseFolderName>/<ColorRepresentation>/<OriginalFileName>.pkl
```

Although the code in ``create_histograms.py`` generates the histograms for five different color spaces, the subsequent tasks require the choice of only two color representations. In our case, the selected ones have been CIELAB and a combination of RGB and Grayscale.

#### Method 1: CIELAB
[CIELAB](https://en.wikipedia.org/wiki/CIELAB_color_space) was chosen for its perceptual uniformity, allowing for more accurate color comparisons that align with human vision (as the perceived difference is proportional to the distance in the space).

- __L* Channel__: Represents lightness, ranging from black (0) to white (100), showing the image's intensity without any color information.
- __a* Channel__: Encodes color information on a red-green axis, where negative values indicate green and positive values indicate red.
- __b* Channel__: Encodes color information on a blue-yellow axis, with negative values representing blue and positive values representing yellow.

The following figure represents the decomposition of an image in the three channels in CIELAB. The histograms below each channel visualize the distribution of intensity values for the respective components of the image.

![Example of the CIELAB channel decomposition of an image](figs/CIELAB_example.png)

#### Method 2: GRAY and RGB
In Method 2, we combine [Grayscale](https://en.wikipedia.org/wiki/Grayscale) and [RGB](https://en.wikipedia.org/wiki/RGB_color_model) representations. Grayscale focuses on intensity, highlighting textures and patterns, while RGB retains essential color information. This combination provides a balanced approach, leveraging texture and color details.

- __Red Channel__: Contains the intensity of red hues in the image. Brighter areas represent stronger red content.
- __Green Channel__: Represents the intensity of green hues, where brighter areas have more green content.
- __Blue Channel__: Shows the intensity of blue hues. Brighter areas indicate stronger blue content.
- __Grayscale Image__: A representation of the image where all color information is removed, focusing solely on brightness (light and dark areas) across the entire image.

As before, the figure below shows the RGB channels of an image and its grayscale version. The histograms illustrate the distribution of intensity values for each color or grayscale component.

![Example of the RGB channel decomposition and grayscale version of an image](figs/RGB_grey_example.png)

### Task 2
Task two consisted on choosing between different similarity measures that would be used to compute the likeliness of the histograms. The measure we chose is correlation, which is defined as:

$$
d(H_1, H_2) = \frac{\sum_I (H_1(I) - \bar{H}_1)(H_2(I) - \bar{H}_2)}{\sqrt{\sum_I (H_1(I) - \bar{H}_1)^2 \sum_I (H_2(I) - \bar{H}_2)^2}}
$$

### Task 3
In task three, the similarity between the queries and all the images in the database is computed according to the described criteria in the two previous tasks. The top ``K`` predictions for each query (i.e., the indices of the ``K`` most similar histograms for each of the queries, sorted) are returned by the function ``generate_results()``. These results allow us to compute the mAP@K. The necessary computations are encapsulated in ``test_methods.py``. The obtained values for the mAP@K with the pre-selected methods are the following:
|                                   |  mAP@1  |  mAP@5  |
|-----------------------------------|---------|---------|
| Method 1 (Correlation & CIELAB)   |  0.300  |  0.326  |
| Method 2 (Correlation & GRAY+RBG) |  0.300  |  0.351  |

### Task 4
The creation of the submition for the blind competition is done in the ```main.py``` file. Remember to execute the file ```create_histograms.py``` before.

## Files in this project
Here you can find a bief description of the files that we created to solve this week's tasks.

### distance_matrix.py
### histograms.py
### create_histogram.py
### main.py

## Extra functionalities
The code in this repository also includes some extra functions that are not needed to solve the compulsory tasks for this project, but that have been used to generate visualizations. They are described below.

* ```plot_histograms```: makes possible the visualization of the histogram extracted from an image. If there is more than one chanel the histograms appear in the same plot.