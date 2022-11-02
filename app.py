import pickle
import requests
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd
from sklearn import preprocessing




import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier


from sklearn import metrics
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
import re
import warnings
from wordcloud import WordCloud

from sklearn.ensemble import RandomForestRegressor
warnings.filterwarnings("ignore")


app=Flask(__name__)
df=pd.read_csv("dataset-seek.csv")

df1=df.iloc[:520,1:]
df1['salary']=df1['salary'].fillna(method='ffill')
df1['sal']=df1.astype({'salary': 'str'}).dtypes
df1['sal1']=df1['salary'].astype(str)
df1['sal1']=df1['sal1'].apply(lambda x: x.replace('k','').replace('$',''))
df1['Salary Estimate']=(df1['sal1'].apply(lambda x:x.split('+')))  
df1['Salary Estimate']=(df1['sal1'].apply(lambda x:x.strip('$')))
df1['Salary-Estimate1']=df1['sal1'].apply(lambda x:x.split(' '))
df1['salary_estimate']=df1['sal1'].apply(lambda x:x.split('+')[0])   
df1['salary_min']=(df1['Salary-Estimate1'].apply(lambda x:x[:][0]))
df1['min_salary'] =  df1['salary_min']
df1['Salary-max']=(df1['Salary-Estimate1'].apply(lambda x:x[:][2:3]))
df1['value1'] =  df1['Salary-max'].str[0]
df1=df1.drop(['Salary Estimate', 'Salary-Estimate1','sal1','sal','salary_estimate','salary_min','Salary-max'], axis=1)


df1['min_salary']=df1['min_salary'].apply(lambda x:x.split(',')[0])
df1['value2']=df1['value1'].astype(str).apply(lambda x:x.split(',')[0])
df1['value2']=df1['value1'].astype(str).apply(lambda x:x.split(',')[0])
df1=df1.drop(['value1'],axis=1)
df1['min_salary1']=df1['min_salary'].apply(lambda x:x.split('-')[0])
min_salary1=df1['min_salary1']
df1['min_salary1'].replace(to_replace = 'Base',value='103',inplace=True)
df1['min_salary1'].replace(to_replace = 'Attractive',value='190',inplace=True)
df1['min_salary1'].replace(to_replace = 'Up',value='100',inplace=True)
df1=df1.drop(['salary','min_salary'],axis=1)
df1.rename(columns = {'value2':'max_salary','min_salary1':'min_salary','class':'Industry'}, inplace = True)
df1['max_salary']=pd.to_numeric(df1['max_salary'], errors='coerce')
df1['min_salary']=pd.to_numeric(df1['min_salary'], errors='coerce')
df1['max_salary']=df1['max_salary'].fillna(method='bfill')
df1['min_salary1']=df1['min_salary'].fillna(method='ffill')
df1['min_salary']=pd.to_numeric(df1['min_salary'], errors='coerce')
df1['max_salary']=df1['max_salary'].fillna(method='bfill')
df1['min_salary1']=df1['min_salary'].fillna(method='ffill')
df1['min_salary']=df1['min_salary'].astype(int)
df1['avg_salary'] = (df1.min_salary+df1.max_salary)//2
df1=df1.drop(['min_salary1'],axis=1)
dummy = pd.get_dummies(df1, columns=['Industry', 'location','title','company'], prefix=['I', 'l','t','c'], drop_first=True)
dummy.head()
dummy=dummy.drop(['max_salary','min_salary','days_before'],axis=1)

df1=dummy



df1=df1.drop(['description'],axis=1)
y = df1['avg_salary']
X = df1.drop(columns=['avg_salary'])


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, stratify=y)



model=RandomForestRegressor(max_depth=50,n_estimators=100, n_jobs=-1)
model.fit(X_train, y_train)

   

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    #request.get_json(force=True)
    title_arr=np.zeros(22)
    company_arr=np.zeros(30)
    location_arr=np.zeros(7)
    Industry_arr=np.zeros(6)

    if request.method=="POST":
   
            title=request.form["title"]
            if title=='Data Scientist':
                    title_arr[0]=1
            elif title=='Machine Learning Engineer':
                    title_arr[1]=1
            elif title=='Data Engineer':
                    title_arr[2]=1
            elif title=='Senior Data Engineer':
                    title_arr[3]=1
            elif title=='Principal Data Engineer - Data & Analytics':
                    title_arr[4]=1
            elif title=='Junior Machine Learning Engineer':
                    title_arr[5]=1
            elif title=='Principle Data Scientist':
                    title_arr[6]=1
            elif title=='Engineer- Big Data Technologies':
                    title_arr[7]=1
            elif title=='Data Scientist (Knowledge Graph)':
                    title_arr[8]=1
            elif title=='Data Engineer - Python (High Frequency Trading)':
                    title_arr[9]=1
            elif title=='Senior AI Systems Engineer':
                    title_arr[10]=1
            elif title=='Machine Learning Lead':
                    title_arr[11]=1
            elif title=='Senior Data Scientist':
                    title_arr[12]=1
            elif title=='AI Engineer':
                    title_arr[13]=1
            elif title=='Analytics Engineer':
                    title_arr[14]=1
            elif title=='Principal Data Consultant':
                    title_arr[15]=1
            elif title=="Data Engineers - Big Data ":
                    title_arr[16]=1
            elif title=='Senior Data Scientist - Macquarie Park/WfH':
                    title_arr[17]=1
            elif title=='Senior Data Engineer/Machine Learning  ':
                    title_arr[18]=1
            elif title=='Machine Learning Data Scientist ':
                    title_arr[19]=1
            elif title=='PhD Studentships in Automated and Transparent Machine Learning':
                    title_arr[20]=1
            elif title=='Data Scientist - Life and Investments  ':
                    title_arr[21]=1
            elif title=='Data Engineer | BIG W':
                    title_arr[22]=1
            elif title=='Data Scientist (AIPS - Recs)':
                    title_arr[23]=1
            

            company=request.form["company"]
            if company=='Bluefin Resources Pty Limited':
                    company_arr[0]=1
            elif company=='Correlate Resources':
                    company_arr[1]=1
            elif company=='The Argyle Network':
                    company_arr[2]=1
            elif company=='ASIC':
                    company_arr[3]=1
            elif company=='Westpac Group':
                    company_arr[4]=1
            elif company=='Pearson Australia':
                    company_arr[5]=1
            elif company=='Talent Insights Group Pty Ltd':
                    company_arr[6]=1
            elif company=='SEEK Limited':
                    company_arr[7]=1
            elif company=='The Onset':
                    company_arr[8]=1
            elif company=='TheDriveGroup':
                    company_arr[9]=1
            elif company=='CatapultBI':
                    company_arr[10]=1
            elif company=='Intelligen Pty Ltd':
                    company_arr[11]=1
            elif company=='Charterhouse':
                    company_arr[12]=1
            elif company=='Susquehanna Pacific Pty Ltd':
                    company_arr[13]=1
            elif company=='Aldi Stores':
                    company_arr[14]=1
            elif company=='Big W':
                    company_arr[15]=1
            elif company=='PRA':
                    company_arr[16]=1
            elif company=='Westbury Partners':
                    company_arr[17]=1
            elif company=='Versent Pty Ltd':
                    company_arr[18]=1
            elif company=='NewyTechPeople':
                    company_arr[19]=1
            elif company=='Precision Sourcing ':
                    company_arr[20]=1
            elif company=='SustainAbility Consulting ':
                    company_arr[21]=1
            elif company=='Robert Walters ':
                    company_arr[22]=1
            elif company=='Alloc8':
                    company_arr[23]=1
            elif company=="'Laing O'Rourke Australia Construction Pty Limited":
                    company_arr[24]=1
            elif company=='AC3 Pty Limited  ':
                    company_arr[25]=1
            elif company=='CPB Contractors Pty Limited':
                    company_arr[26]=1
            elif company=='University of Technology Sydney':
                    company_arr[27]=1
            elif company=='Stockland':
                    company_arr[28]=1
            elif company=='Bet Technology':
                    company_arr[29]=1
            elif company=='Zurich Financial Services Australia':
                    company_arr[30]=1
            elif company=='EY ':
                    company_arr[31]=1
            
            location=request.form["location"]
            if location=='Sydney NSW':
                    location_arr[0]=1
            elif location=='Parliament House':
                    location_arr[1]=1
            elif location=='North Sydney':
                    location_arr[2]=1
            elif location=='North Shore & Northern Beaches':
                    location_arr[3]=1
            elif location=='Minchinbury':
                    location_arr[4]=1
            elif location=='Macquarie Park':
                    location_arr[5]=1
            elif location=='Bella Vista':
                    location_arr[6]=1
            
            Industry=request.form["Industry"]
            if Industry=='Information & Communication Technology':
                    Industry_arr[0]=1
            elif Industry=='Science & Technology':
                    Industry_arr[1]=1
            elif Industry=='Banking & Financial Services':
                    Industry_arr[2]=1
            elif Industry=='Education & Training':
                    Industry_arr[3]=1
            elif Industry=='Government & Defence':
                    Industry_arr[4]=1
            elif Industry=='Consulting & Strategy':
                    Industry_arr[5]=1

    data=np.concatenate([title_arr,company_arr,location_arr,Industry_arr])
    prediction=model.predict([data])[0]
    
    return render_template('home.html',prediction_text = "the salary is ${}".format(round(prediction)))

if __name__=="__main__":
    app.run(debug=True)



