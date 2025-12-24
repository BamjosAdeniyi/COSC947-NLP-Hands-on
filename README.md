# COSC947 Natural Language Processing: Hands-on Project - Sentiment Analysis

## 1. Introduction

This repository contains the hands-on project for the COSC947 Natural Language Processing course in Computer Science (PhD), at Babcock University. The project focuses on exploring various sentiment analysis techniques, ranging from lexicon-based methods to classical machine learning and neural networks, applied to different text datasets. This work fulfills a core requirement of the course, demonstrating practical application and understanding of NLP concepts.

## 2. Project Overview

The primary goal of this project is to implement and compare different approaches to sentiment analysis. We analyze sentiment in two main types of textual data: AI-related tweets and product reviews. Additionally, we leverage a larger IMDB movie review dataset for neural network training and comparison.

### Key Objectives:
*   Implement a lexicon-based sentiment analysis model (VADER).
*   Explore the impact of emoji sentiment on predictions.
*   Apply classical machine learning models (Logistic Regression, Naïve Bayes).
*   Develop a basic Feedforward Neural Network (FFNN) using PyTorch for sentiment classification.
*   Evaluate and compare the performance of these models across different datasets.
*   Interpret the results using accuracy scores and confusion matrices.

## 3. Covered Topics & Techniques

This project covers a wide array of NLP topics and techniques essential for sentiment analysis:

*   **Text Preprocessing**: Implied through the use of pre-cleaned datasets.
*   **Lexicon-based Sentiment Analysis**: Utilization of NLTK's VADER (Valence Aware Dictionary and sEntiment Reasoner) to assign sentiment scores and labels.
*   **Emoji Sentiment Handling**: A custom function to detect and incorporate emoji presence into sentiment analysis.
*   **Feature Extraction**: TF-IDF (Term Frequency-Inverse Document Frequency) vectorization for converting text into numerical features suitable for machine learning models.
*   **Classical Machine Learning Models**: 
    *   **Logistic Regression**: A linear model used for binary or multiclass classification.
    *   **Multinomial Naïve Bayes**: A probabilistic classifier often used for text classification tasks due to its simplicity and effectiveness with sparse features.
*   **Neural Networks (Deep Learning)**:
    *   **Feedforward Neural Network (FFNN)**: A basic neural network architecture implemented with PyTorch, demonstrating fundamental deep learning concepts for NLP.
    *   **PyTorch**: An open-source machine learning framework for building and training neural networks.
*   **Model Evaluation**: Use of standard metrics like accuracy, precision, recall, F1-score, and confusion matrices to assess model performance.

## 4. Datasets Used

*   **`cleaned_top100_tweets.csv`**: A small dataset of 100 AI-related tweets, used for initial VADER and Logistic Regression experiments.
*   **`cleaned_top200_reviews.csv`**: A dataset of 200 product reviews, used for evaluating Naïve Bayes and Neural Network performance on a different domain.
*   **NLTK's IMDB Movie Reviews**: A larger dataset (2000 samples) of movie reviews, commonly used for binary sentiment classification, primarily utilized for training and evaluating the Neural Network and Naïve Bayes models.

## 5. Models and Experiments

### 5.1 Lexicon-Based Analysis (VADER)

*   **Dataset**: `cleaned_top100_tweets.csv`
*   **Method**: VADER assigned sentiment labels (`positive`, `negative`, `neutral`) based on compound scores. An additional custom function `emoji_sentiment` was created to count emojis, though not explicitly integrated into the final VADER prediction accuracy for `sentiment_label` comparison.
*   **Outcome Interpretation**: VADER achieved an accuracy of `0.44` when compared against the `sentiment_label` column of the tweets dataset. The classification report showed high precision for 'neutral' but very low recall, indicating it struggled to correctly identify neutral sentiments, often misclassifying them. Conversely, it had high recall for 'positive' but low precision. This suggests VADER's rule-based nature might not perfectly align with the `sentiment_label` annotation in this specific dataset, especially for nuanced or domain-specific language.

### 5.2 Classical Machine Learning Models

#### 5.2.1 Logistic Regression

*   **Dataset**: `cleaned_top100_tweets.csv` (using VADER-generated `label` as target).
*   **Feature Extraction**: TF-IDF (3000 max features, ngram_range=(1,2)).
*   **Outcome Interpretation**: Logistic Regression achieved an accuracy of `0.70` on the tweets dataset. While appearing higher than VADER, the classification report showed `0.00` precision, recall, and F1-score for 'negative' and 'neutral' classes, indicating that the model did not predict these classes at all and likely defaulted to 'positive' predictions for most samples. This behavior is common with highly imbalanced datasets or when the model struggles to differentiate minority classes with limited features, as seen in the confusion matrix.

#### 5.2.2 Naïve Bayes

*   **Dataset**: NLTK's IMDB Movie Reviews.
*   **Feature Extraction**: TF-IDF (4000 max features, stop words='english').
*   **Outcome Interpretation**: The Multinomial Naïve Bayes model performed well on the IMDB dataset, achieving an accuracy of `0.8075`. It showed balanced precision, recall, and F1-scores for both 'neg' and 'pos' classes (`0.79-0.83`), indicating its effectiveness in handling text classification, especially with its probabilistic approach that is robust to feature independence assumptions often suitable for bag-of-words representations.

### 5.3 Neural Network for NLP (PyTorch)

#### 5.3.1 IMDB Dataset

*   **Dataset**: NLTK's IMDB Movie Reviews (processed into numerical labels: 0 for 'neg', 1 for 'pos').
*   **Feature Extraction**: TF-IDF (5000 max features, stop words='english').
*   **Model Architecture**: A Feedforward Neural Network (FFNN) with two hidden layers (256, 128 neurons), ReLU activations, and Dropout layers (0.4) for regularization. Output layer with 2 neurons for binary classification.
*   **Training**: 10 epochs, Adam optimizer, CrossEntropyLoss.
*   **Outcome Interpretation**: The Neural Network achieved an accuracy of `0.8325` on the IMDB dataset, slightly outperforming the Naïve Bayes model. It showed strong and balanced performance across both classes, with precision, recall, and F1-scores around `0.82-0.84`. This suggests that the FFNN, with sufficient data and appropriate architecture, can learn more complex patterns and achieve better generalization than simpler models for this task.

#### 5.3.2 200 Product Reviews Dataset

*   **Dataset**: `cleaned_top200_reviews.csv` (processed into numerical labels: 0 for 'negative', 1 for 'neutral', 2 for 'positive').
*   **Feature Extraction**: TF-IDF (3000 max features).
*   **Model Architecture**: The same FFNN architecture as above, but with the output layer adjusted to 3 neurons for multiclass classification.
*   **Training**: 10 epochs, Adam optimizer, CrossEntropyLoss.
*   **Outcome Interpretation**: Both the Neural Network and Naïve Bayes models achieved a high accuracy of `0.975` on the product reviews dataset. However, the classification report reveals a critical insight: the dataset's test split had only one 'negative' sample and 39 'positive' samples, with no 'neutral' samples. Both models failed to predict the single 'negative' instance (0 precision, recall, f1 for class 0), while performing perfectly on the 'positive' class. This extremely high accuracy is misleading due to severe class imbalance in the test set. It highlights the importance of robust evaluation metrics beyond raw accuracy and understanding the data distribution during analysis, particularly for datasets with skewed class distributions.

## 6. Key Findings and Interpretations

*   **VADER's Limitations**: While useful for quick, general sentiment analysis, VADER's rule-based nature can struggle with specific datasets where its lexicon or rules don't perfectly align with the ground truth labels or domain-specific language. Its `0.44` accuracy on the tweets indicates this limitation.
*   **Classical ML on Small/Imbalanced Data**: Logistic Regression on the small tweets dataset, particularly with potential class imbalance, showed very poor performance for minority classes, indicating its sensitivity to data characteristics. Naïve Bayes, however, proved to be a strong baseline, performing commendably on the IMDB dataset.
*   **Neural Network Strength**: The FFNN demonstrated a slight but noticeable improvement over Naïve Bayes on the larger, balanced IMDB dataset (`0.8325` vs `0.8075`), suggesting its ability to capture more intricate feature relationships given sufficient data.
*   **Impact of Data Imbalance**: The results on the 200 product reviews dataset are a crucial learning point. The `0.975` accuracy for both Naïve Bayes and the Neural Network is artificially inflated due to a highly imbalanced test set (predominantly positive samples). This emphasizes the need to inspect confusion matrices and classification reports carefully, especially the per-class metrics, to avoid misinterpreting model performance. For highly imbalanced datasets, metrics like F1-score for minority classes or weighted averages become more informative than overall accuracy.
*   **Model Choice**: The choice of sentiment analysis model depends heavily on the dataset's size, balance, domain, and the desired level of interpretability. For general-purpose tasks, VADER can be a quick start. For more complex patterns and larger datasets, neural networks can offer superior performance, while Naïve Bayes provides a robust and often competitive baseline, particularly when features are well-defined (e.g., TF-IDF).

## 7. Setup and Usage

To run this project, ensure you have a Python environment (preferably Google Colab for direct execution). The following steps outline the setup:

1.  **Install Dependencies**: The first code cell (`R2QkoEgsyA3s`) installs all necessary libraries:
    ```bash
    !pip install torch torchvision torchaudio --quiet
    !pip install emoji scikit-learn --quiet
    !pip install emoji # Re-installation just in case
    nltk.download('vader_lexicon')
    nltk.download('movie_reviews')
    ```
2.  **Load Datasets**: Ensure `cleaned_top100_tweets.csv` and `cleaned_top200_reviews.csv` are available in your working directory. The notebook loads these files directly.
3.  **Execute Cells**: Run all cells sequentially, from Section 0 to Section 8. The notebook is structured to demonstrate each analysis step.

## 8. Conclusion

This project provides a practical journey through various sentiment analysis techniques, highlighting their implementation, evaluation, and the critical importance of understanding dataset characteristics. It underscores that while advanced models like neural networks often offer state-of-the-art performance, simpler models like Naïve Bayes remain highly effective, and robust data understanding is paramount for accurate model interpretation and avoiding misleading conclusions from metrics like overall accuracy.
