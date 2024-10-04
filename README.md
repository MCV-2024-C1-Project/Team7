# Master Computer Vision, Module C1

## Module C1 - Week 1
The goal of this week's task is to match pictures of paintings from a large image database with other image queries that feature those artworks. The resemblance between each query and each image in the database is assessed based on their visual content; specifically, the similarity between their histograms.

### Task 1
In task one, up to two methods could be chosen for computing the image descriptors (histograms). We decided to calculate all the histograms in advance for the following color representations: RGB, CIELAB, HSV, YCrCb, and Grayscale, so we could experiment if we wanted to. After the process of creation, the histograms are saved in pickle files. This is done by executing the ```create_histograms.py``` script and having the databases in the following directory:

```
./data
```
It is required to have downloaded the three DB (BBDD, qsd1_w1, qst1_w1) and to have placed them in ``data`` in order to execute the script. After the execution, the files will be distributed as the following:
```
./data/histograms/<DataBaseFolderName>/<ColorRepresentation>/<OriginalFileName>.pkl
```
 
However, the subsequent tasks require the choice of only two color representations. In our case, the selected ones have been 'Rep1' and 'Rep2'.

% TODO Si una d'aquestes representacions separa en diversos canals estaria superxulo posar tres imatges amb cadascun dels canals! <3

#### Representació 1
% TODO explain the representation and the reason for choosing it

#### Representació 2
% TODO explain the representation and the reason for choosing it

### Task 2
Task two consisted on choosing between different similarity measures that would be used to compute the likeliness of the histograms. The measure we chose is:

#### Similarity measure 1
$Formula   de  mates  superxula$

### Task 3
In task three, the similarity between the queries and all the images in the database is computed according to the described criteria in the two previous tasks. The top ``K`` predictions for each query (i.e., the indices of the ``K`` most similar histograms for each of the queries, sorted) are returned by the function ``generate_results()``. These results allow us to compute the mAP@K. The obtained values for the mAP@K with the pre-selected methods are the following:
|          | mAP@1 | mAP@5 |
|----------|-------|-------|
| Method 1 |       |       |
| Method 2 |       |       |

### Task 4
The creation of the submition for the blind competition is done in the ```main.py``` file. Remember to execute the file ```create_histograms.py``` before. The execution of this script has to be done as follows:
```
main.py argument argument etc.
```

## This week's pipeline
As a summary, for this week's task, the execution pipeline to obtain the submition is the following:

```
create_histograms.py --> main.py
```

## New files
### distance_matrix.py
### histograms.py
### create_histogram.py
### main.py
## Extra functionalities

Aqui posem totes les coses extres que hem fet com lo de visualitzar histogrames i tota la pesca

* ```plot_histograms```: makes possible the visualization of the histogram extracted from an image. If there is more than one chanel the histograms appear in the same plot.

## Required libraries
* Numpy
* OpenCV
* Matplotlib
* Pandas 
* Imageio
* Docopt
* Scikit-image
* Sklearn
* Ml_metrics
* Dataframe_image
### 
This repository contains code for the MCV C1 Project. The code can be used to evaluate your results and to
check that the weekly submissions have the proper format.

You can use the evaluation functions in the ```evaluation/``` folder to compute precision, recall, F1, ...,
both at the pixel level and at the window level. Check the ```score_painting_retrieval.py``` script for
an example on how to use the scoring functions.
## Description of the scripts:

### test_submission.py: 
- This script allows the students to test their submissions to minimize the probability of errors.
  It uses a fake GT that is updated every week because the the size of the images has to agree with the ones
  in the week's test set.

  Example of usage of the submission script:

  Assume that 'fake' is the folder with fake ground-truth and 'submission_folder' is the folder where the
  students put the results in Google Drive. 'submission_folder' must contain the folder structure "week?/QST{1,2}/method{1,2}/", similar to the google drive folder where you submit the results. To check the submission for week3, Team 1), the students have to execute:

```
python test_submission.py 3 1 submission_folder fake # Test submission for both QST1 and QST2
```

  another example, for week 5, Team 1:

```
python test_submission_new.py 5 1 submission_folder fake 
```

### score_painting_retrieval.py
- This script is used by the lecturers to perform the scoring. The usage is:

```
# GT files assumed to be on folder W5/qst1_w5
python score_painting_retrieval.py /home/dlcv01/c1-results/week5/QST1 W5/qst1_w5/gt_corresps.pkl --kVal='1,5' --augList=W5/qst1_w5/augmentations.pkl --imaDir=W5/qst1_w5 --gtPicFrames=2020/W5/qst1_w5/frames.pkl
```

### virtualenvs.txt
- Info to create a virtualenv to run this code.


### utils/plot_results.py
- Plot painting retrieval results tables

### utils/print_dict.py
- Print the contents of a .pkl file
