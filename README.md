# Master Computer Vision, Module C1

## Module C1 - Week 1
The goal of this week task is searching images froma a large image BD based on the visual contents using histograms and similarity metrics.
### Task 1
On task one we had to choose up to two methods for computing the image descriptors as histograms. We have decided to calculate beforehand all the histograms for the next representations: RGB, LAB, HSV, YCrCb, Grayscale so we could experiment a bit if we wanted to. After the proces of creation the script saves them in pickle files. This is done executing the ```create_histograms.py``` script and having the databases in the next directory:

```
./data
```
It is required to have downloaded the three DB (BBDD, qsd1_w1, qst1_w1) and placing them in that directory, after that you can execute the script. After the execution the files will be distributed as the following:
```
./data/histograms/<DataBaseFolderName>/<Representation>/<OriginalFileName>.pkl
```
 

However as the next tasks that require choosing two representations the ones selected are:

Si una d'aquestes representacions separa en diversos canals estaria superxulo posar tres imatges amb cadascun dels canals

#### Representació 1

#### Representació 2

### Task 2
On the second task we had to select between diferent similarity measures to compute how alike are two histograms. The selected measure is:

#### Similarity measure 1
$Formula   de  mates  superxula$

### Task 3
For this task the return of the top k images has been done on the main script with the help of the ``generate_result`` function that gives the sorted index of each histogram by similarity. With the results we select the top K results and then compute de MAP@K. The obtained values for the MAP@K with the pre-selected methods are the following:
|          | MAP@1 | MAP@5 |
|----------|-------|-------|
| Method 1 |       |       |
| Method 2 |       |       |

### Task 4
The creation of the submition for the blind competition is done in the ```main.py``` file, remember executing the ```create_histograms.py``` before. The execution of this script has to be done in the next way:
```
main.py argument argument etc.
```

## This weeks pipeline
As a summary, this weeks task the execution pipeline to obtain the submition are the next:

```
create_histograms.py --> main.py
```

## Extra functionalities

Aqui posem totes les coses extres que hem fet com lo de visualitzar histogrames i tota la pesca

## Required libraries
* Numpy
* OpenCV
* Matplotlib

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
