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

TO DO
TO DO
TO DO


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

### Texture extraction related files

### Painting detection and background removal related filed

### Full pipeline related files

