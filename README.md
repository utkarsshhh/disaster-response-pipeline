# Disaster Response Pipeline

This project is a web app with a model for an API to classify disaster messages. It is built using HTML, Flask Framework, Python
and Javascript.


## Introduction

I have developed this project using the disaster data provided by [Appen](https://appen.com/). The application of the project is to classify any messages related to the disaster into one or multiple of the 36 disaster related categories (Eg. Shelter,Fire etc.) provided for better aid. 


## Usage

### Installation

The code is present in python and HTML scripts and requires a Python IDE (I used PyCharm) to the run the project locally.
The code is divided into 3 sections

#### 1.  app

The [app](https://github.com/utkarsshhh/disaster-response-pipeline/tree/main/app) folder contains the HTML, Javascript and Flask code. The application can be run locally from this directory by running the parent file run.py using the command 

> python run.py

#### 2.  data

The [data](https://github.com/utkarsshhh/disaster-response-pipeline/tree/main/data) folder contains the process_data.py script containing the ETL pipeline code storing the raw data files in CSV format to SQLite Database file as "disasters.db".

#### 3. model

The [models](https://github.com/utkarsshhh/disaster-response-pipeline/tree/main/models) folder contains the train_classifier script that uses the Pipeline for NLP transformations and MutiOutputClassifier to prepare and train the model and stores the trained model "classifier.pkl" to be used for making predictions in the app.


### Libraries

#### 1. [pandas](https://pandas.pydata.org)
#### 2. [numpy](https://numpy.org/)
#### 3. [sqlite3](https://docs.python.org/3/library/sqlite3.html)
#### 4. [scikit-learn](https://scikit-learn.org/stable/)
#### 5. [pickle](https://docs.python.org/3/library/pickle.html)
#### 6. [nltk](https://www.nltk.org)
#### 7. [sqlalchemy](https://www.sqlalchemy.org)
#### 8. [plotly](https://plotly.com/python/getting-started/)


### Web Application

The application can be used [here](#)



## Development

The following modules have been applied on the data for Data Wrangling and Modelling

#### 1. [WordNet](https://www.nltk.org/_modules/nltk/stem/wordnet.html)
#### 2. [CountVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html)
#### 3. [TF-IDF](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html)
#### 4. [Pipeline](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html)
#### 5. [MultiOutputClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html)
#### 6. [GridSearchCV](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)


## Result

The detailed results of analysis can be found [here](#)


## Licensing

[Appen](https://appen.com/) has the credit for data. I obtained the data from [Udacity](https://www.udacity.com/) along with the template for the web application as part of Udacity Data Scientist Nanodegree program. 

