# Master in Computer Vision, Module C1
# Week 2
[▶️ Code execution](#code-execution)

[💡 This week's tasks](#this-weeks-tasks)
- [Task 1](#task-1)
- [Task 2](#task-2)
- [Task 3](#task-3)
- [Task 4](#task-4)

[📂 Files in this project](#files-in-this-project)

<h2 id="code-execution">▶️ Code execution</h2>

### Clone the repository
```
git clone https://github.com/MCV-2024-C1-Project/Team7
```

### Move to the corresponding directory
```
cd Team7
```

### Install dependencies
```
pip install -r requirements.txt
```

### Organize datasets
Even though different locations can be specified, in general the scripts assume a specific structure. To make sure everything works as expected, please create a file named ``data`` at the top level of the repository. Then, place the datasets ``BBDD``, ``qsd1_w2``, ``qsd2_w2``, ``qst1_w2`` and ``qst2_w2`` inside the folder ``data``.

### Run the main code
The are two main files that can be run to generate the 2 sumbissions for this week: ``generate_sumbission_cropped.py``, for the images without background, and ``generate_sumbission_background.py`` for the images with background. They are already configured with the best parameters found during our experiments, so the only arguments that need to be specified are the folders with the query and DB images. They can be exectued as follows:

```
python generate_submission_cropped.py \
--queries-dir "./data/qst1_w2/" \
--bbdd-dir "./data/BBDD/"
```

```
python generate_submission_background.py \
--queries-dir "./data/qst2_w2/" \
--bbdd-dir "./data/BBDD/"
```

The arguments shown in the previous command line are the ones by default, which incorporate the best parameter values according to our experiments. Use ``python main.py -h`` to see all the possible arguments.

<h2 id="this-weeks-tasks">💡 This week's tasks</h2>
This week's goal is two fold. On one hand, we are asked to explore more advanced descriptors for the images: 2D/3D histograms and pyramid histograms. On the other hand, we must develop a color-based system to detect and remove the background of uncropped painting images. We then must evaluate both tasks separately and in combination. As in the previous week, the overall, overarching goal is to match pictures of paintings from a large image database with other image queries that feature those artworks.

### Task 1
In task one we had to explore the potential of 2D/3D histograms, pyramid histograms and, possibly, a combination of the two. There are many different way to analyise all the possible combination of these methods. For this reason, we first asked ourselved several questions of interest:

**Regarding 2D/3D histograms**
- Do 2D histograms improve results with respect to 1D histograms?
- Do 3D histograms improve results with respect to 1D histograms?
- Which ones perform better, 2D or 3D histograms?
- What is the best combinations of number of bins and histogram dimensionality (all other parameters fixed)?

**Regarding pyramid histograms**
- Do pyramidal histograms improve results?
- What is a good combination of pyramid levels? Does stacking them work?
- Does the number of bins affect pyramid histograms?

To efficiently answer these questions, we subsequently divided our study in two parts, corresponding to the separate analysis of 2D/3D histograms and pyramid histograms.

The results of the first part can be entirely reproduced executing the following script:
```
python test_2D_3D_histograms.py \
--queries-dir "./data/qsd1_w2/" \
--bbdd-dir "./data/BBDD/" \
--color-space "LAB" \
--similarity-measure "correlation" \
--normalize "None"
```

Given a query dataset and the database dataset, this script explores different combinations of histogram bins and histogram dimensions and calculates MAP@1 and MAP@5 for each one. In particular, it tests all combinations resulting from 256, 128, 64, 32 and 16 bins, and 1, 2, and 3 dimensional histograms. The parameters for color space, similarity measure and normalization have been fixed to match those of last week's Method 1, so the results can be compared. To see the details of the implementation, please see [📂 Files in this project](#files-in-this-project).

*Note: the combination of 256 bins and 3D histograms has not been tested due to the extremely high dimensionality (>16.7 million per descriptor).*

The results obtained for MAP@1 are the following:

| Histogram Dim \ Bins | 256   | 128   | 64    | 32    | 16    |
|----------------------|-------|-------|-------|-------|-------|
| 1                    | 0.333 | 0.367 | 0.400 | 0.300 | 0.367 |
| 2                    | 0.300 | 0.300 | 0.367 | 0.333 | **0.400** |
| 3                    |       | 0.267 | 0.300 | 0.300 | **0.400** |

The results of the second part can be entirely reproduced executing the following script:
```
python test_pyramid_histograms.py \
--queries-dir "./data/qsd1_w2/" \
--bbdd-dir "./data/BBDD/" \
--color-space "LAB" \
--similarity-measure "correlation" \
--histogram-dim 1 \
--normalize "None"
```

Again, given a query dataset and the database dataset, this script explores different combinations of parameters and calculates MAP@1 and MAP@5. In this case, we are combining number of bins and pyramid levels. The number of bins tested have been 128, 64, 32, 16, and 128+Adaptive. The pyramid levels tested have been 1, 2, 3, 4, 5, [1, 2], [1, 2, 3], [1, 2, 3, 4] and [1, 2, 3, 4, 5]. Here, [x, y, x] means we are concatenating pyramid levels x, y and z, and Adaptive bins means that, for each level, bins are calculated dynamically according to an equation. In particular, for level L and initial bins B, we use B/2^(L-1) bins. For example, for level 4 using 128 initial bins, only 128/2^(4-1) = 128/8 = 16 bins would be used. To see more details of the implementation, please see [📂 Files in this project]

The results obtained for MAP@1 are the following:

| Pyramid Level \ Bins  |   (128, False) |   (64, False) |   (32, False) |   (16, False) |   (128, True) |
|:----------------------|---------------:|--------------:|--------------:|--------------:|--------------:|
| [1]                   |          0.367 |         0.400 |         0.300 |         0.367 |         0.367 |
| [2]                   |          0.400 |         0.467 |         0.533 |         0.567 |         0.467 |
| [3]                   |          0.633 |         0.667 |         0.700 |         0.667 |         0.700 |
| [4]                   |          0.867 |         0.867 |         0.867 |         0.767 |         0.767 |
| [5]                   |          **0.900** |         0.867 |         0.867 |         0.800 |         0.700 |
| [1, 2]                |          0.400 |         0.433 |         0.500 |         0.500 |         0.433 |
| [1, 2, 3]             |          0.600 |         0.633 |         0.667 |         0.600 |         0.667 |
| [1, 2, 3, 4]          |          0.833 |         0.867 |         0.833 |         0.767 |         0.767 |
| [1, 2, 3, 4, 5]       |          **0.900** |         0.867 |         0.867 |         0.767 |         0.700 |

*Note: in the preivous table, True or False means whether the Adaptive bins method has been used or not.*

### Task 2
In task two we must compare the results of this week with last week, using the best descriptors from each.

As can be seen in the results of [Task 1](#task-1), the best descriptor of this week achieves a much higher MAP@1.

The following table summarizes the comparison:
| Week | Color Space | Similarity  | Pyramid Level | Hist. Dim. | Bins | Normalization | **MAP@1** | **MAP@5** |
|:-----|:------------|:------------|:--------------|:-----------|:-----|:--------------|----------:|----------:|
| 1    | CIELAB      | Correlation | [1]           | 1          | 256  | None          |     **0.300** |     **0.333** |
| 2    | CIELAB      | Correlation | [5]           | 1          | 128  | None          |     **0.900** |     **0.908** |

As can be seen, there has been a **3-fold increase in performance** thanks to adding a pyramid level 5 and reducing the number of bins.

- ``Higher Pyramid Levels Drive Significant Performance Improvements:`` Employing pyramid histograms with higher division levels resulted in a notable boost in performance. The finer spatial divisions allow the histograms to capture more intricate details within the images, leading to much better matching accuracy.

- ``1D Histograms Outperform 2D and 3D Variants:`` Despite experimenting with 2D and 3D histograms, the results show that 1D histograms, when combined with pyramid levels, deliver the best performance. Among the tested configurations, 1D histograms with 128 bins and pyramid level 5 emerged as the most effective.

And Although combining 2D histograms with pyramids seemed promising initially, it proved ineffective. The results were consistently worse than the 1D histogram-pyramid combination, with MAP@1 scores on average 5.8% lower.

Additionally, we discovered that the Bhattacharyya Metric Yields Even Better Results after fixing a bug in Week 1. Upon recomputing the metrics, we found that using Bhattacharyya distance improved MAP@1 to 0.933 for CIELAB, 1D histograms, pyramid level 5, and 64 bins (non-adaptive). This suggests that Bhattacharyya distance is a more suitable similarity measure for this task compared to correlation.

### Task 3
The goal of the third task is to remove the background from each image. In order to do so, the task has been solved as a binary segmentation problem, where the foreground to segment is the painting that appears in each of the images. Several strategies have been studied.

#### Adaptive Thresholding
Determines the threshold for a pixel based on a small region arround it. For this method we tried two different aproaches:
- **Mean**: the pixels arround are all equally weighted.
- **Gausian**: the pixels arround weight is assigned based on a Gaussian distribution.
#### Otsu's Thresholding
Selects the optimal threshold that minimizes the within-class variance, which is defined as:
$$\omega^{2} = q_1(t)\sigma_1^2(t)+q_2(t)\sigma_2^2(t)$$
Where $q_1(t)$ and $q_2(t)$ are the class probabilities and $\sigma_1^2(t)$ and $\sigma_2^2(t)$ are the variances of the two classes separated by the threshold t.

We also tried to intruduce a **bias factor** to push the background's mean and standard deviation to be as close as possible to the ones extracted from the images borders.

#### Mean shift
Creates clusters based on the density, in this case we use the value of each pixel so it generates clusters based on the colors. In this case makes possible detecting the background based on the color it has. 

For further improvement of this method after de mean shift we also applied otsu's and binary thresholding.

#### Post-processing
After all these techniques we identified that even most of the edges are segmented the inside of the painting is considered background and, thus, not filled. So we tried two different post-processing strategies to solve it.

- Erosion + Convex Hull Filling: after eroding the edges we generate a convex hull arround those and fill it as the inside of the convex hull should contain the painting.

- Closing + Largest Contour Filling: after appling a clossing to the obtained mask to assure a better painting coverage in the mask we then apply the largest countour filling wich finds the painting shape and fills it.

### Task 4
The choosen method for the final results and therefore the metrics for task 4 has been a combination of Mean Adaptative Threshold (MAT), Closing and Largest Contour Filling (LCF). This combination gave us the values on the next table:

| Method | Precision | Recall  | F1 |
|:-----|:------------|:------------| :------------|
| MAT + Closing + LCF   | **0.96**      | **0.99** | **0.98** |

<h2 id="files-in-this-project">📂 Files in this project</h2>
Here you can find a bief description of the most relevant files and functions that we created to solve this week's tasks. They will be divided into two sections, corresponding to the main two tasks of this week.

### Files related to advanced histogram features

#### ``histograms.py``
- **get_histograms()**
This function works similarly to the one we had before but with two additional arguments: dimension and bins. Its role is to generate histograms for a given image, which can be 1D, 2D, or 3D depending on the specified parameters.
- For 1D histograms, it functions as before, but now allows the number of bins to be modified.
- For 2D and 3D histograms, it computes histograms for all possible combinations between the color space dimensions of the image. For example, with dimension=2 and an image in RGB, it will calculate 3 histograms: [R, G], [R, B], and [G, B].

- **plot_all_histograms()**
This function allows us to plot all the histograms returned by get_histograms(). It can automatically detect and visualize the histograms, regardless of their dimension or how many histograms there are.

- **compare_histograms()**
This function generalizes OpenCV’s compareHist(). It takes in two lists of histograms, the output from get_histograms() for two different images, and computes the global distance between them. It calculates the distance between each pair of histograms and then averages the results. Additionally, the function allows you to specify the type of distance to use and includes an option to normalize the histograms (True/False). This function should be called within create_distance_matrix() to replace the default OpenCV compareHist() method.

- **get_pyramid_histograms()**
This function generates pyramid histograms for an image, returning a list of histograms for all sub-images across different pyramid levels. It takes several arguments: the image, the histogram dimension, the number of bins, and a list of levels (e.g., [1, 2, 3]). For instance, with dimension=2, bins=64, and levels=[1, 2, 3], the function will return a list containing 1 + 4 + 16 (i.e., 21) 2D histograms, each with 64x64 bins.
- There’s also an optional adaptive_bins() argument, which reduces the number of bins as the pyramid levels increase. For example, with bins=256 and levels=[1, 2, 3], level 1 will have 256 bins, level 2 will have 256/2 bins, and level 3 will have 256/4 bins. This helps manage bin sizes across different pyramid levels.

### Files related to background detection and removal

#### ``segmentation_analysis.ipynb``
This notebook contains the experiments computed to test the different proposed segmentation methods. It includes:
- The generation of the histograms of the images' border pixels
- Advanced thresholding techniques
- Post-processing techniques
