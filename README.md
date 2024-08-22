# module21challenge
Project Overview
This project aims to build and optimize a neural network model to predict the success of funding applications received by AlphabetSoup, a fictional charity organization. The project involves preprocessing the data, creating and training the model, optimizing the model, and analyzing the results.

Requirements
Python 3.8 or higher
TensorFlow 2.16 or higher
Pandas 1.3.5 or higher
Scikit-learn 0.24.2 or higher
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/yourusername/alphabet-soup-charity.git
cd alphabet-soup-charity
Create and activate a virtual environment:

bash
Copy code
python -m venv env
source env/bin/activate  # On Windows use `env\Scripts\activate`
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Running the Project
Data Preprocessing:

The script preprocesses the data by:

Dropping non-beneficial ID columns (EIN, NAME).
Replacing low-frequency categorical values with "Other".
One-hot encoding the categorical variables.
Splitting the data into features (X) and target (y), and further into training and testing sets.
Model Definition:

A Sequential neural network model is created with:

Input layer matching the number of features.
Two hidden layers with ReLU activation.
Output layer with sigmoid activation for binary classification.
Model Compilation and Training:

The model is compiled using adam optimizer and binary_crossentropy loss function. It is then trained on the training data for 100 epochs.

Model Evaluation:

The model's performance is evaluated using the test data, and the loss and accuracy metrics are printed.

Model Export:

The trained model is saved to an HDF5 file named AlphabetSoupCharity.h5.

Model Optimization:

 Note: This step requires additional work where you need to apply at least three optimization techniques to the model and save the optimized model as AlphabetSoupCharity_Optimization.h5.
Analysis Report:

 Note: An analysis report must be written covering the model's performance, answering key questions, and discussing alternative models.
