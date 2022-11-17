# importing the libraries required for the ETL pipeline
import pandas as pd
import numpy as np
import sqlite3 as sql



#Extracting the data

messages = pd.read_csv("messages.csv")
disaster_category = pd.read_csv("categories.csv")
print (messages.head())
print (disaster_category.head())



#Transform the categories to usable columns


category_column = disaster_category['categories']
#Fetch the list of categories from the first row
category_list = [category[:-2] for category in category_column[0].split(";")]

#Split the categories to individual columns
df_category = pd.Series(category_column).str.split(";",expand = True)
df_category.columns = category_list
df_category = df_category.apply(np.vectorize(lambda x: int(x[-1])))

#Add the categories columns to the categories dataframe
categories = pd.concat([disaster_category,df_category],axis =1)
categories.drop(labels = 'categories',axis = 1, inplace = True)

#Merge categories and messages dataframes
categorized_messages = messages.merge(categories,on = 'id')

# Drop the duplicates in dataframe
print (categorized_messages['id'].value_counts())
categorized_messages.drop_duplicates(inplace = True)
print (categorized_messages['id'].value_counts())




#Load the clean data to SQLite

conn = sql.connect('disasters.db')
categorized_messages.to_sql('categorized_messages',conn)
conn.close()









