'''
mjob mother job ;fjob father job.Famsize le3- <=3; Gt3 > 3. PSTatus A-otdelno jivut, T-vmeste jivut 
School: MS-Mousinho da Silveira , GP-Gabriel Pereira
Address: U(urban-gorodkoy) , R(rural-derevenskiy)
MEdu mamina obrozovanie ; FEdu father obrazovanie 
traverl time vrema puti doma shkola doma  
absences otsutsvie
Student Alcohol Consumption - Effect of alcohol on Grades
'''
import pandas as pd 
import seaborn as sns 
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve 
from sklearn.metrics import confusion_matrix
from sklearn.linear_model import LogisticRegression 
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split #traning test
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.ensemble import RandomForestClassifier

#read data
df = pd.read_csv("student-mat.csv",sep=';')
df.info() 
#obyem data
df.describe()
sns.countplot(df['Dalc'],label='count')
sns.countplot(df['Walc'],label='count')
sns.countplot(df['G3'],label='count')

#yes/no 1/0 deyishmek
df['schoolsup'].replace({'yes': 1, 'no': 0}, inplace=True)
df['famsup'].replace({'yes': 1, 'no': 0}, inplace=True)
df['paid'].replace({'yes': 1, 'no': 0}, inplace=True)
df['activities'].replace({'yes': 1, 'no': 0}, inplace=True)
df['nursery'].replace({'yes': 1, 'no': 0}, inplace=True)
df['higher'].replace({'yes': 1, 'no': 0}, inplace=True)
df['internet'].replace({'yes': 1, 'no': 0}, inplace=True)
df['romantic'].replace({'yes': 1, 'no': 0}, inplace=True)
df['sex'].replace({'F': 1, 'M': 0}, inplace=True)#female1 male0
df['school'].replace({'GP': 1, 'MS': 0}, inplace=True)
df['address'].replace({'U': 1, 'R':0}, inplace=True)
df['Pstatus'].replace({'A': 0, 'T': 1}, inplace=True)
df['famsize'].replace({'GT3': 1, 'LE3': 0}, inplace=True)
df['Mjob'].replace({'health':1,'at_home':2, 
                    'teacher':3, 'services':4, 'other':5}, inplace=True) 
df['Fjob'].replace({'health':1,'at_home':2, 
                    'teacher':3, 'services':4, 'other':5}, inplace=True)
df['reason'].replace({'other':0, 'reputation':1, 
                      'home':2, 'course':3}, inplace=True)
df['guardian'].replace({'other':0, 'father':1, 'mother':2}, inplace=True)
df['G3'].replace({1:0, 2:0, 3:0, 4:0,
                  5:1, 6:1, 7:1, 8:1, 9:1, 10:1,
                  11:2, 12:2, 13:2, 14:2, 15:2, 
                  16:3, 17:3, 18:3, 19:3, 20:3}, inplace=True)

#data type deyishmek 
df = df.astype(str) 

#pivot table
pt = pd.pivot_table(df, index = 'G3',
                     columns= 'Walc',
                     values= 'Dalc', aggfunc= 'mean')

pt2 = pd.pivot_table(df, index = 'absences',
                     columns= 'Dalc',
                     values= 'G3', aggfunc= 'mean')

# Crosstable - saymaq üçün
crs = pd.crosstab(index = df['Dalc'],
                  columns= df['G3'])


'''Modeling'''

# Data X Y bolmek 
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

# Training Test set 
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state = 1)

# Feature Scaling odin raz xv
st = StandardScaler()
X_train = st.fit_transform(X_train)
X_test = st.transform(X_test)

'''Logistic Regression Model''' 
lt = LogisticRegression()
lt.fit(X_train, Y_train)
Y_pred = lt.predict(X_test)

# Confusion Matrix 
con = confusion_matrix(Y_test, Y_pred)

# accuracy_score
acc = accuracy_score(Y_test, Y_pred) 


classification_report(Y_test, y_pred)
roc = roc_auc_score(Y_test, y_pred)
f12 = f1_score(Y_test, y_pred)
''' Decision Tree Classifier '''
cl = DecisionTreeClassifier(criterion= "entropy", random_state=0)
cl.fit(X_train, Y_train)

# Test Model
y_pred = cl.predict(X_test)

# Confusion Matrix
con2 = confusion_matrix(Y_test, y_pred)
 
acc2 = accuracy_score(Y_test, y_pred)

# Precision
pr = precision_score(Y_test, y_pred)

# Recall
rc = recall_score(Y_test, y_pred)

f1 = f1_score(Y_test, y_pred)
roc = roc_auc_score(Y_test, y_pred)
classification_report(Y_test, y_pred)

'''Random forest'''

cl = RandomForestClassifier(n_estimators= 5, criterion= 'entropy', random_state = 0)
cl.fit(X_train, Y_train)
y_pred = cl.predict(X_test)

confusion_matrix(Y_test, y_pred)

acc3 = accuracy_score(Y_test, y_pred)

classification_report(Y_test, y_pred)
roc = roc_auc_score(Y_test, y_pred)
f1 = f1_score(Y_test, y_pred)