# Master in Computer Vision, Module C1
# Week 3
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
Even though different locations can be specified, in general the scripts assume a specific structure. To make sure everything works as expected, please create a folder named ``data`` at the top level of the repository. Then, place the datasets ``BBDD``, ``qsd1_w3``, ``qsd2_w3``, ``qst1_w3`` and ``qst2_w3`` inside the folder ``data``.

### Run the main code

To generate the submissions for this week, **2 different executable files** have been created.

1. For the predictions for the images without background (**QST1_W3**), please execute `generate_sumbission_cropped.py` as is indicated next. In this file, the best denoising and best texture methods have been used.

```
python generate_submission_cropped.py \
--queries-dir "./data/qst1_w3/" \
--bbdd-dir "./data/BBDD/"
```

2. For the predictions for the images with background (**QST2_W3**), as well as the corresponding masks, please execute `generate_submission_background.py` as is indicated next. In this file, the best denoising and best texture methods have been used, as well as a method to detect and extract the paintings present in each image.

```
python generate_submission_background.py \
--queries-dir "./data/qst2_w3/" \
--bbdd-dir "./data/BBDD/"
```

For more information on the methods used in these files, see the following sections.


<h2 id="this-weeks-tasks">💡 This week's tasks</h2>

This week we must tackle three distinct challenges:
- Some images might contain noise (random samples or hue changes), so we must filter the noise (**Task 1**).
- We can only use texture descriptors to compare and retrieve images (**Task 2**).
- Given an image, we must detect the paintings present in the image (1 or 2), then remove the background (**Task 3**).

At the end, the different methods to solve these challenges will be combined in a single pipeline, going from a raw, noisy image with one or two paintings, to the predictions corresponding to those paintings (**Task 4**).

### Task 1
In Task 1, we must develop and test one or more methods to filter the noise in the images. Additionally, it is specified that we must do this using linear or non-linear filters.

TO DO
TO DO
TO DO


### Task 2
In Task 2, we must develop and test one or more methods to extract texture descriptors from the images. We should do the feature extraction using strictly texture descriptors only, and it is suggested we use LBP, DCT or wavelet-based methods for this purpose. 

To adress this task, we have decided to study and test four different approaches: **LBP**, **DCT**, **Wavelets** and **Gabor filters**. For each of these approaches, using the clean QSD1_W3 dataset, we have studied different parameter combinations and preprocessing techniques. Then, we have analized the results and selected the best variant of each approach. At the end, we have also tested these variants with the noisy QSD1_W3 dataset, as well as with all the noise-filtered versions of this dataset, corresponding to the 5 denoising methods developed in Task 1. This way, we also have been able to assess how these approaches react to the presence of noise. Next, we explain in more detail each approach.

#### DCT

At the heart of this approach, the `dct()` function from **OpenCV** has been used. As additional preprocessing, we have considered changing the color space, splitting the image into different blocks, and rescaling to 256x256 for computational reasons as well as to simplify the process of splitting into blocks.

A total of **360** different parameter combinations have been tested. They are the following:
- **Color**: gray, L (from CIELAB), V (from HSV)
- **Block size**: 256x256, 128x128, 64x64, 32x32, 16x16, 8x8
- **Number of relevant coefficients**: 64, 32, 16, 8, 4
- **Distance measure**: Cosine similarity, Pearson correlation, L1, L2

Next are the highlights after analyzing the results:
- The V color channel performs better on average than the other color channels.
- The Cosine/Pearson distances perform significantly better on average than L1 and L2
- Very high or very log dimensional feature vectors perform bad on average.
- Out of the 360 different combinations, 30 of them have perfect scores (MAP@1 and MAP@5 both equal to 1).
- The best variant corresponds to: *using V channel, Cosine distance, block size of 64, 64 relevant coefficients* (see DCT results analysis for more information on this decision).

#### LBP

The core of this approach is the `local_binary_pattern()` function from Scikit-image. Just like in the DCT approach, we have changed the color, splitted the image into different blocks and resized the image (in this case, because LBP is strongly dependent on resolution).

A total of **324** different parameter combinations have been tested. They are the following:
- **Color**: gray, L , V
- **Number of blocks**: 64, 16, 4, 1
- **(Radius, Points)**: (1, 8), (2, 10), (3, 12)
- **Calculation method**: "default", "ror", "uniform"
- **Distance measure**: Bhattacharyya, Correlation, Intersection
The multi-resolution LBP approach has also been tested (concatinating all three combinations of raidus/points pairs), but the results are similar to the best results found with the single-resolution approach, although higher dimensional.

Next are the highlights after analyzing the results:
- The L color channel performs better on average, although not by much.
- The more blocks that are used, the better the results (on average).
- The defualt calculation method is much better. We believe this is because the "ror" method (roation invariant) produces sparse vectors (as processed with our functions), and thus distances cannot be properly calculated with them. On the other hand, the uniform method has been designed to be fast to compute, so the information is (in our case) too compressed (the feature vectors are low dimensional).
- The best variant corresponds to: *using the L channel, Bhattacharyya distance, radius of 3 together with 12 neighboring points, and using the default calculation method* (see LBP results analysis for more information on this decision).

#### Wavelets

TO DO
TO DO
TO DO

#### Gabor filters

TO DO
TO DO
TO DO

#### Studying how the presence of noise affects performance

To analyze how noise affects peformance, the best variants of all four methods have been tested not only with the clean QSD1_W3 dataset (which is the one used to perform the studies), but also with the noisy QSD1_W3 dataset, as well as with 5 different versions of denoised datasets (see Task 1 for more info on the denoising methods). The results are shown in the following table:

![results for the performance with and without noise](figs/noise_performance_table.png)

Analyzing the results, the following comments can be made:
- aaa
- aaa
- aaa
- aaa


### Task 3
In Task 3, we must be able to detect all the paintings in an image, then extract those paintings, effectively removing the background. In addition, a binary mask must be created so the developed method can be evaluated.

TO DO
TO DO
TO DO


### Task 4
In Task 4, we must combina all the previous methods, so that we can go from a raw, noisy image to the predictions for the paintings found in the image.

TO DO
TO DO
TO DO



<h2 id="files-in-this-project">📂 Files in this project</h2>

Here you can find a bief description of the most relevant files and functions that we created to solve this week's tasks. They will be divided into different sections, corresponding to the main tasks of this week.

### Noise removal related files

TO DO
TO DO
TO DO


### Texture extraction related files

TO DO
TO DO
TO DO


### Painting detection and background removal related filed

TO DO
TO DO
TO DO


### Full pipeline related files

TO DO
TO DO
TO DO

