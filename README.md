implemented a hybrid model for aspect-based sentiment analysis on Yelp reviews. It combines a BERT model for contextual embeddings and a GRU layer for sequential processing.

Here's a breakdown of the code and why each step was performed:

Import Libraries: Essential libraries like pandas for data handling, torch for model building, transformers for pre-trained models, and scikit-learn for data splitting are imported.
Load Data: The Yelp reviews dataset is loaded using kagglehub.
Balance Data: The dataset is balanced by sampling 100 reviews for each star rating to ensure the model doesn't get biased towards more frequent ratings.
Tokenization and Dataset Preparation:
The bert-base-uncased tokenizer is loaded to convert text into numerical input suitable for BERT.
A custom YelpDataset class is created to handle the data loading and tokenization for training and testing.
Star ratings are converted into three sentiment classes: negative (1-2 stars), neutral (3 stars), and positive (4-5 stars).
The data is split into training and testing sets.
DataLoader instances are created to efficiently load data in batches during training.
Hybrid Model Definition:
A HybridAspectModel is defined, inheriting from nn.Module.
It incorporates a pre-trained bert-base-uncased model for generating contextual embeddings.
A bidirectional GRU layer processes the BERT output sequence to capture dependencies.
Two linear layers (heads) are added: one for predicting aspects and one for predicting sentiment.
An Adam optimizer is used for updating model weights.
Cross-entropy loss is used as the loss function for both aspect and sentiment prediction.
The model is trained for a specified number of epochs, iterating through the training data in batches.
