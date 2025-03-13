# neural-network-challenge-2

# Neural Network Challenge 2 - Employee Attrition Prediction

## Assignment Overview

In this assignment, I created a neural network that HR can use to predict whether employees are likely to leave the company (attrition). Additionally, I also predicted which department might be a better fit for each employee. The task involves implementing a branched neural network to handle two separate target predictions: **attrition** and **department**.

## Steps Involved

### Part 1: Preprocessing
1. **Import Data**: Loaded the employee data using `pandas.read_csv()`.
2. **Create `y_df`**: The target columns `Attrition` and `Department` were selected and stored in `y_df`.
3. **Create `X_df`**: Chose 10 columns from the data to use as input features for the model (excluding `Attrition` and `Department`).
4. **Convert Data to Numeric**: Non-numeric columns in the dataset were converted into numeric types using techniques such as `OneHotEncoder`, `LabelEncoder`, or `.astype()`.
5. **Split Data**: The data was split into training and testing sets using `train_test_split()`.
6. **Standard Scaling**: Applied `StandardScaler` to scale the features to ensure the model performs efficiently.
7. **One-Hot Encoding for Target Variables**: Applied `OneHotEncoder` to encode the `Attrition` and `Department` target columns.

### Part 2: Model Creation, Compilation, and Training
1. **Model Architecture**: 
    - Created a **non-sequential** model (branched structure) with two output branches.
    - Defined an **input layer**.
    - Included at least two shared hidden layers.
    - Created a branch for predicting `Department` and another for predicting `Attrition`.
2. **Model Compilation**: Compiled the model with an appropriate loss function and optimizer.
3. **Model Training**: Trained the model on the preprocessed training data.
4. **Model Evaluation**: Evaluated the model on the test data and printed the accuracy for both the `Department` and `Attrition` predictions.

### Part 3: Summary Questions
1. **Accuracy as a Metric**: Answered whether accuracy is the best metric to use for this data and provided reasoning.
2. **Activation Functions**: Explained the choice of activation functions used in the output layers.
3. **Model Improvements**: Discussed possible ways to improve the model, such as adding more hidden layers or experimenting with different activation functions.

## Key Fixes and Improvements
- **Categorical Data Encoding**: Ensured all categorical data was converted to numeric types for compatibility with neural networks.
- **Model Evaluation**: Added the evaluation step to measure the model's accuracy on both target columns (`Department` and `Attrition`).
- **Model Improvement Suggestions**: Suggested several ways to improve the model, such as adding more layers, optimizing hyperparameters, and using additional features.

## Technologies Used
- Python
- Pandas
- Scikit-learn
- Keras/TensorFlow

## How to Run the Code
1. Clone the repository to your local machine.
2. Ensure you have the required libraries installed:
    ```bash
    pip install pandas scikit-learn tensorflow
    ```
3. Run the Jupyter notebook `attrition.ipynb` to see the results and outputs.

