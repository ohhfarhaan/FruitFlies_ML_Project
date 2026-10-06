# Real-Time Detailed Video Analysis of Fruit Flies

A machine learning project for real-time analysis of fruit-fly videos. The system detects flies, determines their sex, estimates their orientation, and predicts the wing angle of male flies.

## Team

- Mohammed Farhan Ahmed
- Mancirat Singh

## Problem Statement

Fruit-fly behavioral analysis often requires manually examining video frames to determine the position, sex, orientation, and wing movement of individual flies.

This project develops a machine-learning-based pipeline that automatically performs these tasks from video frames and provides real-time analysis.

## Project Pipeline

The system consists of four main components.

### 1. Fly Detection / Fly Count

- Detects fly contours from video frames.
- Uses contour-based features.
- Uses a Decision Tree classifier to identify flies.

### 2. Male/Female Classification

- Extracts geometric features such as contour area and aspect ratio.
- Uses feature standardization.
- Uses Logistic Regression to distinguish between male and female flies.

### 3. Fly Orientation Detection

- Uses image moments to estimate the fly orientation.
- Extracts HOG features from the fly image.
- Uses PCA for dimensionality reduction.
- Uses Logistic Regression to resolve the remaining orientation ambiguity.

### 4. Male Wing Angle Prediction

- Processes the male fly wing region.
- Extracts HOG features.
- Uses PCA for dimensionality reduction.
- Uses Linear Regression to predict the wing angle.

## Dataset

The project uses the annotated fruit-fly video dataset associated with the original CS229 project.

The dataset contains:

- Annotated images
- Hand-labeled fly information
- Test videos

The dataset is not included in this repository because of its large size.

Place the downloaded dataset in:

input/
├── images/
└── videos/

## Technologies Used

### Programming and Libraries

- Python
- OpenCV
- NumPy
- Scikit-learn
- Matplotlib
- imutils
- Joblib
- tqdm

### Machine Learning Techniques

- Decision Tree
- Logistic Regression
- Linear Regression
- Principal Component Analysis (PCA)
- Histogram of Oriented Gradients (HOG)

## Installation

The project was developed and tested using Python 3.6 in an x86_64 Conda environment on macOS.

### 1. Create the Conda environment

```bash
conda create --platform osx-64 -n cs229-project python=3.6
conda activate cs229-project


2. Install dependencies
pip install opencv-python==4.1.2.30
pip install numpy==1.19.5
pip install scikit-learn==0.24.2
pip install matplotlib==3.3.4
pip install imutils==0.5.4
pip install joblib==1.1.1
pip install tqdm==4.64.1

3. Install the project
pip install -e . --no-deps

4. Add the dataset
Place the dataset in:
cs229-project/
└── input/
    ├── images/
    └── videos/

The dataset is intentionally excluded from GitHub because of its large size.
Running the Project
Train the models
make

Run the real-time demo
make demo

Run the performance test
make profile

Experiments
Orientation PCA Experiment
The orientation pipeline originally uses PCA to reduce the dimensionality of HOG features before Logistic Regression.
We experimentally evaluated different PCA configurations:
- 5 components
- 10 components
- 15 components
- 25 components
- 40 components
The comparison was used to study the effect of PCA dimensionality on male and female orientation classification.
The original baseline configuration uses 15 PCA components.
Orientation Experiment Results
PCA Components	Male Test Accuracy	Female Test Accuracy
5	98.33%	98.85%
10	99.17%	98.85%
15	99.17%	98.85%
25	99.17%	98.85%
40	100.00%	98.85%


The experiment shows that increasing the number of PCA components does not consistently improve classification accuracy. The 10-component configuration achieved the same measured accuracy as the 15-component configuration in the comparison experiment while using fewer components.
For the final integrated pipeline, the original 15-component configuration was retained.
Wing Angle PCA Experiment
The wing-angle regression pipeline was also evaluated with different PCA dimensionalities.
PCA Components	MAE	RMSE	R²
40	2.247°	3.111°	0.982480
60	1.982°	2.802°	0.985787


Increasing PCA components from 40 to 60 resulted in:
- 11.79% lower MAE
- 9.93% lower RMSE
- R² improvement from 0.982480 to 0.985787
The 60-component configuration was therefore selected for the final wing-angle model.
Final Model Results
After integrating the project pipeline, the final training run produced:
Component	Test Result
Fly Count	0.0% error
Male/Female Classification	0.0% error
Male Orientation	0.8% error
Female Orientation	0.0% error
Wing Angle	0.043 rad test σ


The wing-angle error corresponds to approximately 2.46°.
Real-Time Performance
The complete pipeline was tested on a MacBook Air with an Apple M5 processor.
A demo run processed:
- Total frames: 631
- Elapsed time: approximately 20 seconds
- Throughput: 31.56 FPS
The measured per-frame processing stages were approximately:
Stage	Time per frame
I/O	4.7 ms
Fly contour detection	2.4 ms
Male/Female identification	0.6 ms
Orientation	1.8 ms
Wing angle	4.2 ms


The complete system therefore operates at approximately 31.56 FPS on the development MacBook Air M5.
Project Structure
FruitFlies_ML_Project/
│
├── cs229/
│   ├── contour.py
│   ├── demo.py
│   ├── load_data_id.py
│   ├── load_data_is_fly.py
│   ├── load_data_orientation.py
│   ├── load_data_wing.py
│   ├── train_id.py
│   ├── train_is_fly.py
│   ├── train_orientation.py
│   └── train_wing.py
│
├── experiments/
│   ├── compare_wing_pca.py
│   └── pca_comparison.py
│
├── results/
│   └── results.md
│
├── Makefile
├── README.md
├── setup.py
├── LICENSE
└── .gitignore

Output Files
Generated datasets, trained models, graphs, and other large generated files are excluded from the Git repository.
Examples include:
output/data/
output/models/
output/graphs/
output/diagrams/

These files are generated locally when the project is run.
Reproducibility
To reproduce the project:
1. Create the Conda environment.
2. Install the required dependencies.
3. Download and place the dataset in input/.
4. Run:
make

5. Run the real-time demo:
make demo

Conclusion
This project demonstrates how classical computer vision and machine learning techniques can be combined to perform detailed real-time fruit-fly video analysis.
The project was first reproduced as a working baseline and then extended through controlled PCA experiments.
The orientation experiment evaluated the effect of PCA dimensionality on classification performance, while the wing-angle experiment showed that increasing PCA components from 40 to 60 improved regression performance on the evaluated dataset.
The final integrated pipeline achieved approximately 31.56 FPS on the development MacBook Air M5 while performing fly detection, sex classification, orientation estimation, and wing-angle prediction.
Acknowledgement
This project is based on and extends the original Stanford CS229 fruit-fly video analysis project by Steven Herbst.
The original implementation and dataset were used as the starting point for reproducing the baseline pipeline and conducting our experiments.