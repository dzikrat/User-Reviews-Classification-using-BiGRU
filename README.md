# Sentiment Analysis of User Reviews Application using BiGRU
## Overview
This project focuses on sentiment analysis of user reviews from an application. Using a Bidirectional Gated Recurrent Unit (BiGRU) model, the goal is to classify the sentiment of each review as either positive or negative. The model leverages deep learning techniques to analyze the textual data and provides an evaluation based on accuracy and F1-score.

*This research was carried out as a requirement for completing the bachelor's degree in Statistics at the Faculty of Mathematics and Natural Sciences, Universitas Padjadjaran.*
## Data
- **Source**: User reviews from the Mobile BeaCukai application on the Google Play Store.
- **Period of Time**: September 13, 2017, to November 18, 2023.
- **Volume**: 1,189 reviews written in Indonesian.
- **Method**: Data collected using scraping techniques.
## Features
- **BiGRU**: Implemented a Bidirectional GRU model for sentiment analysis.
- **Sentiment Classification**: Classifies user reviews into positive or negative sentiments.
## Tools and Libraries
- **Python**: Programming language used for the entire project.
- **TensorFlow and Keras**: Used for implementing the BiGRU model.
- **Scikit-learn**: Used for data splitting, generating classification reports, and evaluating the model using a confusion matrix.
- **Pandas and NumPy**: Utilized for data manipulation and numerical operations.
- **Google Play Scraper Package**: Used for scraping user reviews from the Google Play Store.
- **FastText Word Embedding**: Employed for converting words into vector representations to improve the model's understanding of semantic relationships in the text.
## Steps :
![image](https://github.com/user-attachments/assets/6fef20ad-cb4b-4966-bf07-5627a1618ab0)
## Results
- **Accuracy**: 81.51%
- **F1-Score**: 80.51%

The research results show that a model with 1 embedding layer with FastText word embedding, 1 Bi-GRU layer with 32 neurons, 1 dense layer with 32 neurons, 1 dropout layer with a dropout rate of 0.2, a batch size of 32, Adam optimizer with a learning rate of 0.001, and a maximum of 100 epochs is able to classify reviews with an accuracy of 81.51% and an F1-Score of 80.51%. Therefore, this model is good enough to automatically predict positive and negative sentiments in the reviews of the application in the future.
