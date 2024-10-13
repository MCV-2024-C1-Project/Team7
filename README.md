# Master Computer Vision, Module C1
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

### Task 2

### Task 3

### Task 4

### Task 5

### Task 6

<h2 id="files-in-this-project">📂 Files in this project</h2>
Here you can find a bief description of the most relevant files and functions that we created to solve this week's tasks.

#### ``file.py``
Explanation
