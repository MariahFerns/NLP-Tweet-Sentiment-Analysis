# #️⃣Tweet Sentiment Analysis

## Introduction
This project aims to analyze tweets and predict their sentiment using Natural Language Processing (NLP) techniques and machine learning models. The goal is to classify tweets into three categories: Neutral, Negative, and Positive.

## Objectives
- Perform data preprocessing and text cleaning.
- Use NLP techniques to prepare the text data for model training.
- Build and train an LSTM model for sentiment analysis.
- Evaluate the model and make predictions on sample text.

## Dataset Description
The dataset contains the following columns:

| Column Name | Description                             |
|-------------|-----------------------------------------|
| Tweets      | Data in the form of sentences written by individuals |
| category    | Sentiment category: 0 (Neutral), -1 (Negative), 1 (Positive) |

## Methodology
### Data Preparation
1. **Import Libraries and Load Dataset**:
   - Import necessary Python libraries such as pandas, seaborn, matplotlib, sklearn, nltk, and tensorflow.
   - Load the dataset from the provided Excel file into a pandas DataFrame.

2. **Change Dependent Variable to Categorical**:
   - Convert the numerical categories (0, -1, 1) to categorical labels ("Neutral," "Negative," "Positive").

3. **Missing Value Analysis**:
   - Check for missing values and drop any null/missing values from the dataset.

### Text Preprocessing
1. **Clean Text Data**:
   - Remove all symbols except alphanumeric characters.
   - Transform all words to lowercase.
   - Remove punctuation, stopwords, and numbers.
   - Perform tokenization, lemmatization, and expand contractions.

### Data Splitting
1. **Split Data into Dependent and Independent DataFrames**:
   - Separate the tweets (X) from the sentiment categories (y).

### Text Data Operations
1. **One-Hot Encoding and Padding**:
   - Perform one-hot encoding for each sentence using TensorFlow.
   - Add padding to the sequences from the front side using TensorFlow.

### Model Building and Training
1. **Build and Compile LSTM Model**:
   - Define the model architecture including input length, vocabulary size, dropout layer, and activation function.
   - Compile the LSTM model.

2. **Dummy Variable Creation**:
   - Create dummy variables for the dependent variable categories.

3. **Split Data into Training and Test Sets**:
   - Split the data into training and testing sets.

### Model Training
1. **Train the Model**:
   - Train the LSTM model on the training data.

### Model Evaluation
1. **Normalize Predictions**:
   - Normalize the prediction results to match the original categories (nearest to 1 is predicted as yes, others as 0).

2. **Measure Performance Metrics**:
   - Calculate accuracy, print the classification report, and plot the confusion matrix.

### Sample Text Inferences
1. **Make Inferences**:
   - Pass sample text through the model and make predictions.

## Folder Structure
- `data/`: Contains the dataset files.
- `notebooks/`: Jupyter notebooks for data analysis, text preprocessing, and model training.
- `src/`: Python scripts for data processing, text preprocessing, and model training.
- `api/`: Api code (if any).
- `models/`: Trained models and saved results.
- `results/`: Output files including visualizations, model evaluation metrics, and plots.

## Installation and Usage
1. **Clone the Repository**:
   ```bash
   git clone https://github.com/MariahFerns/Tweet-Sentiment-Analysis.git
   cd Tweet-Sentiment-Analysis

2. Install the required libraries:
   ```bash
   pip install -r requirements.txt

3. Run Jupyter Notebooks:
   Navigate to the notebooks/ folder and open the notebooks to explore data analysis and model development.

## Results and Findings
**EDA Insights:**
- Identified the distribution of tweet sentiments.
- Found patterns in text data that contribute to sentiment classification.

**Model Evaluation:**
- Evaluated the LSTM model and found it to be effective in classifying tweet sentiments.
- Plotted ROC curves and confusion matrices to compare model performance.

## Conclusion
This project successfully analyzed and classified tweet sentiments using NLP techniques and an LSTM model. The findings can help in understanding public opinion and sentiment on various topics.
