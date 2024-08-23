# module21challenge

# Project Overview

  In this project, the goal was to create and refine a neural network model that could predict the success of funding applications submitted to AlphabetSoup, a fictional charity. The project covered all the essential steps: preprocessing the data, building and training the model, optimizing the model for better performance, and analyzing the results.

# Running the Project
  # Data Preprocessing:
  
The data preprocessing script handles several tasks:

- Dropping non-essential ID columns (EIN, NAME).
- Replacing rare categorical values with "Other".
- One-hot encoding the categorical variables.
- Splitting the data into features (X) and target (y), then further splitting it into training and testing sets.
       
  # Model Definition:
    
  The neural network model is built as follows:

- A Sequential model with an input layer matching the number of features.
- Two hidden layers, each using the ReLU activation function.
- An output layer with a sigmoid activation function, since we’re doing binary classification.

  # Model Compilation and Training:

  The model is compiled with the Adam optimizer and the binary_crossentropy loss function. It is then trained on the training data for 100 epochs.

  # Model Evaluation:
  After training, the model is evaluated using the test data, and both the loss and accuracy metrics are printed.

  # Model Export:
  The trained model is saved in an HDF5 file named AlphabetSoupCharity.h5.

  # Model Optimization:
  Several optimization strategies were employed to try and improve the model’s performance:

- Increasing the number of neurons in the hidden layers.
- Adding an extra hidden layer.
- Experimenting with different activation functions.
- Adjusting the number of training epochs.

  Although these optimizations didn’t push the model's accuracy past the desired 75% threshold, they provided valuable insights into the factors that influence the performance of deep learning models.

  # Summary:
  
In the end, the neural network model performed reasonably well in predicting the success of the funding applications, though it didn’t quite hit the accuracy target. The optimization process highlighted how different tweaks could impact model performance. If I were to try this again, I might go with a decision tree. This model might better handle the categorical data and provide more clarity on which features are most influential in predicting success.

