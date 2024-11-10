# Master in Computer Vision, Module C1
# Week 4
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
Even though different locations can be specified, in general the scripts assume a specific structure. To make sure everything works as expected, please create a folder named ``data`` at the top level of the repository. Then, place the datasets ``BBDD``, ``qsd1_w4`` and ``qst1_w4`` inside the folder ``data``.

### Run the main code

% TODO EXPLAIN THIS WEEK'S FILES

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

This week, the retrieval of the paintings must be done using keypoint detectors and local descriptors. Our approach to solve this task consists os several studies:
1. **Testing** of the different keypoint detection and local description **methods**.
2. Study of the different **matching methods** for the descriptors.
3. **Optimization** of the **threshold** to detect **unknown paintings** for different keypoint detection and local description methods.
4. Study of how the **noise** can affect the performance of the keypoint detection and local description matching methods.

### Task 1
This task consists of the implementation of the keypoint detection and local description methods.

#### Testing of the different keypoint detection and local description methods and study of the different matching methods for the descriptors

We have suggested two different detectors to solve this problem:
- **Harris corner detector:** identifies corners using local intensity changes, creating stable rotationally invariant points, but not invariant in scale.
- **Harris Laplacian:** adds scale invariance to Harris corner detector by applying a Laplacian of Gaussian filter.

Many options are available to compute local description. In our case, we have used the ones listed below.
- **SIFT:** detects keypoints invariant in scale and rotation, then creates descriptors based on the gradient of the region around the keypoint.
- **ORB:** combines the FAST keypoint detection with the descriptor BRIEF with improvements to get rotational invariance and selecting the most descriptive keypoints.
- **AKAZE:** uses a non-linear scale space and binary descriptors, making it highly efficient and resilient to changes in lighting and contrast.

However, the implementation of SIFT, ORB and AKAZE also allows us to both detect the keypoints and to compute the local descriptors. For this reason, the previous 5 methods result in a total of 9 combinations, as shown in the following figure:

  ![Combinations of keypoint detectors and local descriptors](figs/methods_combinations.png)

 Two different matching methods were used, resulting in a total of 18 combinations:
 - **Brute Force:** A descriptor of the first image is compared with all the descriptors in the other image using a distance. The closest one is returned. Using OpenCV BFMatcher function. To check whether the matches were correct, **Cross Check** method was used (the matches are checked bidirectionally: a match of two descriptors A and B is good only if B is the closest to A, and A is the closest to B).
 - **FLANN:** More efficient than Brute Force for large datasets / high dim descriptors. Finds approximately close matches. Different search methods available. Using OpenCV FlannBasedMatcher. In this case, the goodness of the matches was checked using **Lowe's ratio** (as described in D. Lowe SIFT paper: compares closest (M1) and second-closest (M2) match for each keypoint, calculating the ratio M1/M2 then using a threshold to decide).

The performance of each combination was evaluated using mean average precision at `K=1`and `K=5`. The results are shown below.

  ![Perfomance of each detector-descriptor combination with different matching strategies](figs/methods_results.png)

### Task 2


### Task 3

In this task, we must evaluate our system on this week's dataset. Moreover, we are asked to evaluate last week's query system on this new dataset and compare their performances. First, we decided to do an additional study you will find below.

### Additional study: how does noise affect the different keypoint descriptors methods?

This study investigates how noise affects the performance of two keypoint detection and descriptor methods, ORB and AKAZE. Using the FLANN matching method, we analyze each method’s robustness under both noisy and noise-free conditions, with performance evaluated through MAP@1. Therefore, the experiment setup is the following:

1. **Conditions**: Keypoint detection and descriptor methods: **ORB** and **AKAZE**. We also used **FLANN matching** to assess performance, which was the best matching method. This methods are used for both noisy and _clean_ images.

2. **Noise Type in Dataset**: We used the qsd1_w3 dataset, which contains two types of synthetic noise:
    - **Salt-and-Pepper Noise**: Disrupts pixel intensities randomly, creating sharp, contrasting points.


    ![Example of salt-and-pepper noisy image](figs/sp_noisy.jpg) ![GT for the same image](figs/sp_gt.jpg)


    - **Color Noise**: Introduces subtle color shifts that can affect pixel values across different channels.


    ![Example of color alteration noisy image](figs/color_noisy.jpg) ![GT for the same image](figs/color_gt.jpg)

3. **Performance Evaluation**: We used MAP@1, Mean Average Precision at k=1, scores to evaluate the results.

**Visual Insights**
- Keypoint detection in noisy vs. clean images:
    - Salt-and-Pepper noise:

    ![Example of keypoint detection in salt-and-pepper noisy image](figs/Figure_1query_spnoise.png) ![Example of keypoint detection in GT for the same image](figs/Figure_1query_nonoise.png)


    - Color noise:

    ![Example of keypoint detection in color alteration noisy image](figs/Figure_1query_colornoise.png) ![Example of keypoint detection in GT for the same image](figs/Figure_1query_nonoise_2.png)
    ![Example 2 of keypoint detection in color alteration noisy image](figs/Figure_1query_noise0.png) ![Example 2 of keypoint detection in GT for the same image](figs/Figure_1query_nonoise0.png)


- Bar plot:
The bar plot compares the MAP@1 performance of AKAZE and ORB with the FLANN matching method, with separate bars for "No Noise" and "Noise".


![Bar plot of the results for "No Noise" and "Noise" images](figs/barplot_noise.png)

**Findings on Noise Impact:**
- **Keypoint Detection**:
    - Salt-and-Pepper Noise: This type of noise often introduces false keypoints, as random pixel spikes are detected as features.
    - Color Noise: Minimal impact on keypoint detection unless significant intensity changes occur, which can shift keypoint locations slightly.
- **Descriptors**:
    - Salt-and-Pepper Noise: Creates distortion in descriptors, reducing reliability as it disrupts the local structure needed for accurate matching.
    - Color Noise: No impact on descriptors, as ORB and AKAZE both rely on grayscale intensity rather than color data.
- **Matching Process**:

    Noise disrupts both keypoint detection and descriptor accuracy, leading to a higher rate of mismatches.

In summary, while noise does degrade the retrieval system’s performance, its impact is not severe for either method under the tested conditions, with ORB showing slightly more sensitivity to noise than AKAZE. The effect it has on MAP@1 for each method is:
- **AKAZE: MAP@1 drops by 0.04 with noise.**
- **ORB: MAP@1 sees a more substantial drop of 0.1.**

### Task 4

In Task 4 we must generate and submit the results for a "blind" competition using our final retrieval system.

% TODO EXPLAIN RELEVANT FILES

