#Importing libraries required for building ML pipeline
import sys
import pandas as pd
from sklearn.model_selection import train_test_split,GridSearchCV
import pickle
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger','omw-1.4'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine


#Importing the clean and processed data
if(len(sys.argv)==3):
    db_path = sys.argv[1]
    model_path = sys.argv[2]
else:
    db_path = "../data/disasters.db"
    model_path = "classifier.pkl"
conn = create_engine('sqlite:///'+db_path)
categorized_messages = pd.read_sql_table('categorized_messages',conn)
conn.dispose()


#Split the dataframe to feature and target variables
X = categorized_messages['message']
Y = categorized_messages.iloc[:,5:]



#Split data into train and test dataset
x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size = 0.3,random_state = 1)



def tokenize(text):
    '''
    This function tokenizes and lemmatizes text to make it usable for further transformation

    Input:
    text

    Output:
    returns the input text as clean tokens in a list


    '''
    tokens = word_tokenize(text.lower())
    lemmatizer = WordNetLemmatizer()
    tokenized_text = [lemmatizer.lemmatize(token).strip() for token in tokens]
    return tokenized_text



#Defining the pipeline for tranforming the data and making predictions using the classifier
pipeline  = Pipeline([('vect',CountVectorizer(tokenizer = tokenize)),
                     ('tran',TfidfTransformer()),
                     ('mlf',MultiOutputClassifier(estimator = RandomForestClassifier()))])
pipeline.fit(x_train,y_train)
#Predicting using the pipeline
y_pred = pipeline.predict(x_test)


def check_accuracy(pred_values, real_values):
    '''
    This function prints the comparison of real target values with predicted target values for individual columns

    Input:
    pred_values,real_values


    '''
    pred_values = pred_values.transpose()
    for i in range(pred_values.shape[0]):
        print(classification_report(real_values.iloc[:, i], pred_values[i]))


#Checking the accuracy of "pipeline"
check_accuracy(y_pred,y_test)


#Apply GridSearch on the pipeline to tune the hyperparameters
parameters = {
    'mlf__estimator__max_features':['sqrt','log2']
}
pipeline  = Pipeline([('vect',CountVectorizer(tokenizer = tokenize)),
                     ('tran',TfidfTransformer()),
                     ('mlf',MultiOutputClassifier(estimator = RandomForestClassifier()))])
tuned_model = GridSearchCV(pipeline,parameters,n_jobs=1)
tuned_model.fit(x_train,y_train)
y_pred_tuned = tuned_model.predict(x_test)

#checking the accuracy of tuned_model
check_accuracy(y_pred_tuned,y_test)


#Storing the tuned model in a pickle format
pickle.dump(tuned_model, open(model_path, 'wb'))







