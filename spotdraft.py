import pandas as pd 
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
import numpy as np
import re

import string

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import nltk
from nltk.corpus import stopwords

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.svm import SVC


noise_list = ["is","@", "a", "this", "...","ourselves", "hers", "between", "yourself", "but", "again", "there", "about", "once", "during", "out", "very", "having", "with", "they", "own", "an", "be", "some", "for", "do", "its", "yours", "such", "into", "of", "most", "itself", "other", "off", "is", "s", "am", "or", "who", "as", "from", "him", "each", "the", "themselves", "until", "below", "are", "we", "these", "your", "his", "through", "don", "nor", "me", "were", "her", "more", "himself", "this", "down", "should", "our", "their", "while", "above", "both", "up", "to", "ours", "had", "she", "all", "no", "when", "at", "any", "before", "them", "same", "and", "been", "have", "in", "will", "on", "does", "yourselves", "then", "that", "because", "what", "over", "why", "so", "can", "did", "not", "now", "under", "he", "you", "herself", "has", "just", "where", "too", "only", "myself", "which", "those", "i", "after", "few", "whom", "t", "being", "if", "theirs", "my", "against", "a", "by", "doing", "it", "how", "further", "was", "here", "than"] 

def remove_noise(input_text):
    words = input_text.split() 
    noise_free_words = [word for word in words if word not in noise_list] 
    noise_free_text = " ".join(noise_free_words) 
    return noise_free_text

def cleanhtml(raw_html):
  cleanr = re.compile('<.*?>')
  cleantext = re.sub(cleanr, '', raw_html)
  return cleantext  


#def text_process(text):

#    '''
#    Takes in a string of text, then performs the following:
#    1. Remove all punctuation
#    2. Remove all stopwords
#    3. Return the cleaned text as a list of words
#    '''

#    nopunc = [char for char in text if char not in string.punctuation]
#    nopunc = ''.join(nopunc)    
#    return [word for word in nopunc.split() if word.lower() not in stopwords.words('english')]


data1 = pd.read_csv('Black-Female-Names.csv')
data2 = pd.read_csv('Black-Male-Names.csv')
data3 = pd.read_csv('Hispanic-Female-Names.csv')
data4 = pd.read_csv('Hispanic-Male-Names.csv')
data5 = pd.read_csv('Indian-Female-Names.csv')
data6 = pd.read_csv('Indian-Male-Names.csv')
data7 = pd.read_csv('White-Female-Names.csv')
data8 = pd.read_csv('White-Male-Names.csv')



#datafn = pd.concat([data1["first_name"],data2["first_name"],data3,data4,data5,data6,data7,data8])
#dataln = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8])
#data = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8])


cols = data1.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x)
data1.columns = cols
data1.to_csv("data1.csv", index=False)

cols = data2.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x)
data2.columns = cols
data2.to_csv("data2.csv", index=False)

cols = data3.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x)
data3.columns = cols
data3.to_csv("data3.csv", index=False)

cols = data4.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x)
data4.columns = cols
data4.to_csv("data4.csv", index=False)

cols = data5.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x)
data5.columns = cols
data5.to_csv("data5.csv", index=False)

cols = data6.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x)
data6.columns = cols
data6.to_csv("data6.csv", index=False)

cols = data7.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x)
data7.columns = cols
data7.to_csv("data7.csv", index=False)

cols = data8.columns
cols = cols.map(lambda x: x.replace(' ', '_') if isinstance(x, (str, unicode)) else x)
data8.columns = cols
data8.to_csv("data8.csv", index=False)

##############DATA CLEANING DONE####################


#+++++++==CONCATENATION ON DATA++++++++++++++++++++++++++++++++

#SPLIT THE NAMES IN 2 COLUMNS- INDIAN CSVS
#X = 

data1 = pd.read_csv('data1.csv')
data2 = pd.read_csv('data2.csv')
data3 = pd.read_csv('data3.csv')
data4 = pd.read_csv('data4.csv')
data5 = pd.read_csv('data5.csv')
data6 = pd.read_csv('data6.csv')
data7 = pd.read_csv('data7.csv')
data8 = pd.read_csv('data8.csv')

dataind = pd.concat([data5,data6])
#print(dataind)

df = dataind
df = pd.read_csv('dataind.csv')
#df1 = df['name']
t1 = df['gender']
t2 = df['race']
X = df['name']
X = X.replace(np.nan, 0)
print(type(X))
X = X.astype(str)

for i in range(len(X)):        
    X[i] = remove_noise(X[i])        #remove stopwords

for i in range(len(X)):
    X[i] = cleanhtml(X[i])   	           #remove html chars with regex

for i in range(len(X)):             #remove puncutation marks
    X[i] = X[i].translate(None, string.punctuation)   


for i in range(len(X)):	                                            
    X[i] = X[i].lower()    #set all to lowercase


df8 = pd.DataFrame({"name" :X})
df8.to_csv("cleanedind.csv", index=True)

#dataind.to_csv('dataind.csv', index=False)
#df['A'], df['B'] = X.str.split(' ', 1).str[1]
#df = pd.DataFrame(df8.row.str.split(' ',1).tolist(), columns = ['first_name','last_name'])
df['A'], df['B'] = df8['name'].str.split(' ', 1).str

df9 = pd.DataFrame({"first_name" :df['A'], "last_name" :df['B'] , "gender" : t1, "race": t2  })
df9.to_csv("finalcleanedind.csv", index=True)



###########NOW WE CAN CONCANATE THINGS
datafn1 = pd.concat([data1,data2,data3,data4,data7,data8, df9])  #df9 the indian one required most processsing

#datafn = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8])
datafn1.to_csv("datafn1.csv", index =False)

dftemp = pd.read_csv('datafn1.csv')
tt1 = dftemp['gender']
tt1 = tt1.replace(np.nan, 0)
tt2 = dftemp['race']
tt2 = tt2.replace(np.nan, 0)
#print(tt2)

#now clean the entire data

X2 = dftemp['first_name']
X2 = X2.astype(str)
X2 = X2.replace(np.nan, 0)
print("yoyoyo")
print(type(X2))


for i in range(len(X2)):        
    X2[i] = remove_noise(X2[i])        #remove stopwords

for i in range(len(X)):
    X2[i] = cleanhtml(X2[i])   	           #remove html chars with regex

for i in range(len(X2)):             #remove puncutation marks
    X2[i] = X2[i].translate(None, string.punctuation)   


for i in range(len(X2)):	                                            
    X2[i] = X2[i].lower()    #set all to lowercase


X3 = dftemp['last_name']
X3 = X3.astype(str)
X3 = X3.replace(np.nan, 0)

for i in range(len(X3)):        
    X3[i] = remove_noise(X3[i])        #remove stopwords

for i in range(len(X3)):
    X3[i] = cleanhtml(X3[i])   	           #remove html chars with regex

for i in range(len(X3)):             #remove puncutation marks
    X3[i] = X3[i].translate(None, string.punctuation)   


for i in range(len(X3)):	                                            
    X3[i] = X3[i].lower()    #set all to lowercase

df10 = pd.DataFrame({"first_name" :X2, "last_name" :X3 , "gender" : tt1, "race": tt2  })
df10.to_csv("finalcleanedall.csv", index=True)


#######NOW REMOVE THE MIDDLE WORDS########################

s = X2
#print(s)

for i in range(len(s)) :
    #print(s[i])
    s[i] = s[i].split(" ")[0]
    #s[i] = s[i][0:s[i].index(" ")]


df11 = pd.DataFrame({"first_name" :s, "last_name" :X3 , "gender" : tt1, "race": tt2  })
df11.to_csv("final123cleanedall.csv", index=True)




###############NOW THE CODE CONVERSION TO VECTORS STARTS#####################################

X = s
y = tt1
print(np.shape(X))

#datacat = pd.concat([s, X3],axis=1)
#datacat.to_csv("firstlastcontd.csv", index =False)
#print(np.shape(datacat))
#print("bb")
#X1 = datacat
#y1 = y
#X1 = X3
y1 = y


#####NOW I SHALL JOIN ALL THE FIRST_NAME AND LAST_NAME 

#df['names'] = df.Year.str.cat(df.Quarter)
df11['names'] = s.str.cat(X3, sep=' ')
df15 = pd.DataFrame({"names" : df11['names'] , "gender" : tt1, "race": tt2  })
df15.to_csv("firstlastjoined.csv", index=True)

X1 = df11['names']
y1 = y


#print(y[9])
print(np.shape(X))
print(np.shape(y))

print(type(y))
print("gazwa-e-sindh")
print(type(X))

bow_transformer = CountVectorizer().fit(X) #analyzer=text_process #assigna vector id to each word and count its occurrence frequency
X = bow_transformer.transform(X)    #after fitting the data, transform it for some readable format 
bow_transformer1 = CountVectorizer().fit(X1) #analyzer=text_process #assigna vector id to each word and count its occurrence frequency
X1 = bow_transformer.transform(X1)    #after fitting the data, transform it for some readable format 


tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X)
tfidf_transformer1 = TfidfTransformer()
X1 = tfidf_transformer1.fit_transform(X1)
print("tf-df bhi ho gaya!")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101) #split into training and test data
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=101) #split into training and test data

print("vectorize, split ho gaya!!!!!!!")

#######WITH THE DATA CONVERTED TO FEATURES, NOW WE SHALL APPLY THW ALGORITHMS@@@@@@@@@@@@@@@@@@@@@@@@@@@

nb = BernoulliNB()
#nb = SVC()
nb.fit(X_train, y_train)
preds = nb.predict(X_test)
predsprob = nb.predict_proba(X_test)
print("NAIVE BAYES ho gaya!!")


nb = BernoulliNB()
#nb = SVC()
nb.fit(X_train1, y_train1)
preds1 = nb.predict(X_test1)
preds1prob = nb.predict_proba(X_test1)
print("NAIVE BAYES ho gaya!!")
#print(preds1prob)
#print(type(preds1prob))
print(np.shape(preds1prob))

probm , probf = preds1prob[:,:1] , preds1prob[:,1:2]
probm = probm.ravel() #to convert from ND to 1D)
probf = probf.ravel() 
print(np.shape(probm))
print(np.shape(probf))
print(np.shape(preds1))
print(np.shape(X_test1))
print(np.shape(y_test1))
#for i in (X_test):
#    print( X_test[i] )
#print(type(X))
#sprint(type(X_test))

#clf = SVC()
#clf.fit(X_train, y_train) 
#preds1 = clf.predict(X_test)
#print("SVC ho hogaya!!!!!!!!")

print('first_name Accuracy score:')
print(accuracy_score(y_test, preds))

print('first_name- precision score :')
print(precision_score(y_test, preds, average='micro') )

print('names joined Accuracy score:')
print(accuracy_score(y_test1, preds1))

print('names joined- precision score :')
print(precision_score(y_test1, preds1, average='micro') )

#print(np.shape(X_test))

#df13 = pd.DataFrame({"names-X_test" :X_test, "Y-Y_test" :y_test , "prediction": preds  }) #"gender" : tt1,
#df13.to_csv("predictionfirstname.csv", index=True)

#df16 = pd.DataFrame({"names-X_test" :X_test1, "Y-Y_test" : y_test1  , "prediction": preds1  }) #"gender" : tt1,
#df16.to_csv("predictionjoined.csv", index=True)

df17 = pd.DataFrame({ "Y-Y_test" : y_test1  , "prediction": preds1 , "prob-F":probm, "prob-M":probf  }) #"gender" : tt1,
df17.to_csv("prediction11joinedprob.csv", index=False )



#######PREDICT WHAT YOU WANT#######################
#a = raw_input("Enter the name to guess the gender: ")

'''
df = pd.DataFrame(X2)
df2 = pd.DataFrame([[a]])
#a = df2['a']
print("yoyo2")

df.append(df2, ignore_index=True)
print(df2)


bow_transformer = CountVectorizer().fit(X) #analyzer=text_process #assigna vector id to each word and count its occurrence frequency
X = bow_transformer.transform(X)    #after fitting the data, transform it for some readable format 

tfidf_transformer = TfidfTransformer()
X = tfidf_transformer.fit_transform(X)



print(nb.predict(a))

'''

#print('Accuracy score SVC:')
#print(accuracy_score(y_test, preds1))

#print('precision score SVC:')
#print(precision_score(y_test, preds1, average='micro') )




#########CODE CONVERSION TO VECTORS STOPS###################################################



#s="Hello World<Goodbye World"
#print s.split("<")[0]
#Hello World
#s[:s.index("<")]
#'Hello World'



#df.join(df['name'].str.split('-', 1, expand=True).rename(columns={0:'A', 1:'B'}))


#dataind1 = pd.DataFrame(dataind.row.str.split(' ',1).tolist(), columns = ['first_name','last_name'])
#dataind['first_name'], dataind['last_name'] = X.str.split(' ', 1).str

#print(first_name)



#datafn = pd.concat([data1,data2,data3,data4,data5,data6,data7,data8])
#datafn.to_csv("datafn.csv", index =False)


#REMOVE MIDDLENAMES' WORDS
#concantenate data row wise
