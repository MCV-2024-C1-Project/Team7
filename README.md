# Master in Computer Vision, Module C1
# Week 2
[▶️ Code execution](#code-execution)

[💡 This week's tasks](#this-weeks-tasks)
- [Task 1](#task-1)
- [Task 2](#task-2)
- [Task 3](#task-3)
- [Task 4](#task-4)
- [Task 5](#task-5)
- [Task 6](#task-6)

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
The are two main files that can be run to generate the 2 sumbissions for this week: ``generate_sumbission_background.py``, for the images with background, and ``generate_sumbission_cropped.py`` for the images without background. They can be exectued as follows:

```
python generate_sumbission_background.py \
TO DO
```

```
python generate_sumbission_cropped.py \
TO DO
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

Again, given a query dataset and the database dataset, this script explores different combinations of parameters and calculates MAP@1 and MAP@5. In this case, we are combining number of bins and pyramid levels. The number of bins tested have been 128, 64, 32, 16, and 128+Adaptive. The pyramid levels tested have been 1, 2, 3, 4, 5, [1, 2], [1, 2, 3], [1, 2, 3, 4] and [1, 2, 3, 4, 5]. Here, [x, y, x] means we are concatenating pyramid levels x, y and z, and Adaptive bins means that, for each level, bins are calculated dynamically according to an equation. In particular, for level L and initial bins B, we use B/2^(L-1) bins. For example, for level 4 using 128 initial bins, only 128/2^(4-1) = 128/8 = 16 bins would be used.

The results obtained for MAP@1 are the following:

| pyramid_level \ bins  |   (128, False) |   (64, False) |   (32, False) |   (16, False) |   (128, True) |
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

### Task 3

### Task 4

### Task 5

### Task 6

<h2 id="files-in-this-project">📂 Files in this project</h2>
Here you can find a bief description of the most relevant files and functions that we created to solve this week's tasks. They will be divided into two sections, corresponding to the main two tasks of this week.

### Files related to advanced histogram features

#### ``file.py``
Explanation

### Files related to background detection and removal

#### ``file.py``
Explanation