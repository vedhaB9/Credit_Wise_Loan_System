import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.impute import SimpleImputer

import matplotlib.pyplot as plt
import seaborn as sns


loan=pd.read_csv(r"C:\Users\Tenakani Vedha Shree\Downloads\loan_approval_data.csv")
loan=loan.drop(columns='Gender')
#print(loan.isnull().sum())
#print(loan.info())


categorial_cols=loan.select_dtypes(include=['str']).columns
num_cols=loan.select_dtypes(include=['float64']).columns

num_imp=SimpleImputer(missing_values=np.nan,strategy="mean")
cat_imp=SimpleImputer(missing_values=np.nan,strategy="most_frequent")
loan[num_cols]=num_imp.fit_transform(loan[num_cols])
loan[categorial_cols]=cat_imp.fit_transform(loan[categorial_cols])

#EDA
'''classes=loan['Loan_Approved'].value_counts()
plt.style.use('dark_background')
plt.pie(classes,autopct='%1.1f%%',labels=['no','yes'])
plt.show()
loanpurposes=loan['Loan_Purpose'].value_counts()
plt.bar(loanpurposes.index,loanpurposes.values)
plt.title('purpose of loans')
plt.show()
'''
                                            #   FEATURE ENCODING

pd.set_option('display.max_columns', None)
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
le=LabelEncoder()
loan['Education_Level']=le.fit_transform(loan['Education_Level'])
loan['Loan_Approved']=le.fit_transform(loan['Loan_Approved'])

cols=['Employer_Category','Loan_Purpose','Marital_Status','Employment_Status','Property_Area']
ohe=OneHotEncoder(drop='first',sparse_output=False,handle_unknown='ignore')
encoded=ohe.fit_transform(loan[cols])
encoded_df=pd.DataFrame(encoded,columns=ohe.get_feature_names_out(cols),index=loan.index)
loan=pd.concat([loan.drop(columns=cols),encoded_df],axis=1)

                                        # CORRELATION HEATMAP(to detect multi-collinearity)
num_cols=loan.select_dtypes(include=['number'])
corr_matrix=num_cols.corr()
#print(num_cols.corr()['Loan_Approved'].sort_values(ascending=False))
plt.figure(figsize=(15,8))
sns.heatmap(
    corr_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"

)
plt.show()
                        # TRAIN_TEST_SPLIT + FEATURE SCALING
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X=loan.drop(columns=['Loan_Approved','Applicant_ID'])
y=loan['Loan_Approved']
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
X_train_scaled=scaler.fit_transform(X_train)
X_test_scaled=scaler.transform(X_test)
                        # TRAIN AND EVALUATE MODELS
from  sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score,recall_score,accuracy_score,confusion_matrix
log_model=LogisticRegression(max_iter=100004)
log_model.fit(X_train_scaled,y_train)
y_pred=log_model.predict(X_test_scaled)

print("precision score: ",precision_score(y_test,y_pred))
print("accuracy score: ",precision_score(y_test,y_pred))
print("recall score: ",precision_score(y_test,y_pred))










