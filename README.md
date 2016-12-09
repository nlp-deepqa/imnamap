# IMNAMAP - Iterative Multi-document Neural Attention for Multiple Answer Prediction
Code for the paper "Iterative Multi-document Neural Attention for Multiple Answer Prediction".

## Description
People have information needs of varying complexity, which can be solved by an intelligent agent able to answer questions formulated in a proper way, eventually considering user context and preferences. In a scenario in which the user profile can be considered as a question, intelligent agents able to answer questions can be used to find the most relevant answers for a given user.

In this work we propose a novel model based on Artificial Neural Networks to answer questions with multiple answers by exploiting multiple facts retrieved from a knowledge base. The model is evaluated on the factoid Question Answering and top-n recommendation tasks of the [*bAbI Movie Dialog dataset*](https://research.fb.com/projects/babi/).

After assessing the performance of the model on both tasks, we try to define the long-term goal of a conversational recommender system able to interact using natural language and supporting users in their information seeking processes in a personalized way.

## Requirements
- Python >= 3.4
- TensorFlow >= 0.11.0
- NLTK >= 3.2.1
- Elasticsearch (Python API) >= 2.3.0

## Usage
1. Create pickle file for movie dialog dataset using `build_movie_dialog.py`
2. Create Elasticsearch index from movie dialog knowledge base using `index_movie_dialog.py`
3. Train IMNAMAP models for movie dialog (tasks 1 or 2) using `train_movie_dialog.py` (default command-line parameters are the ones used in the paper)
4. Evaluate the trained models using `eval_movie_dialog.py`

## Authors
All the following authors have equally contributed to this project (listed in alphabetical order by surname):

- Claudio Greco ([github](https://github.com/claudiogreco))
- Alessandro Suglia ([github](https://github.com/aleSuglia))
