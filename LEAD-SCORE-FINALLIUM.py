#!/usr/bin/env python
# coding: utf-8

# # X EDUCATION-LEAD SCORE

# X-Education company wants to judge the customers (students) whether they pay for education or not. Educational companies normally work on these situations and call them as 'Leads'. 

# ### The step by step procedure to analyse business data as follows:
# 
# 
# #### Step 1:  Read Data
# 
# #### Step 2:  Inspecting Data
# 
# #### Step 3:  Data Preparation
# 
# #### Step 4:  Test-Train Split
# 
# #### Step 5:  Feature Scaling
# 
# #### Step 6:  Looking at Correlations
# 
# #### Step 7:  Model Building
# 
# #### Step 8:  Feature Selection using RFE
# 
# #### Step 9:  Plotting the ROC Curve
# 
# #### Step 10: Finding the Optimal Cutoff point
# 
# #### Step 11: Making the predictions on the Testset

# ## Step 1: Read Data

# In[1]:


# Suppressing Warnings
import warnings
warnings.filterwarnings('ignore')


# In[2]:


# Import necessary libraries

import pandas as pd
import numpy as np
from numpy import argmax


# In[3]:


# Importing dataset

x_lead = pd.read_csv('Leads.csv')
x_lead.head()


# ## Step 2: Inspecting Data

# In[4]:


x_lead.shape


# In[5]:


x_lead.describe()


# In[6]:


x_lead.info()


# ## Step 3: Data Preparation

# #### 3.a Converting lengthy column names to small column names

# In[7]:


x_lead = x_lead.rename(columns={'Lead Number':'Lead_Number',
                                'Lead Quality':'Lead_Quality','Last Activity':'Last_Activity',
                                'Page Views Per Visit':'Page_Views_Per_Visit',
                                'Do Not Email':'Do_Not_Email',
                                'Do Not Call':'Do_Not_Call',
                                'Lead Origin':'Lead_Origin',
                                'Lead Source':'Lead_Source',
                                'Update me on Supply Chain Content':'Supply_Chain_Updates',
                                'Get updates on DM Content':'DM_Updates',
                                'How did you hear about X Education':'X_Education_Promotion',
                                'Lead Profile':'Lead_Profile',
                                'Total Time Spent on Website':'Time_Spent_on_Website',
                                'What is your current occupation':'Current_Occupation',
                                'I agree to pay the amount through cheque':'Cheque_Payment',
                                'What matters most to you in choosing a course':'Reason_for_Course',
                                'Asymmetrique Activity Index':'Asymmetric_Activity_Index',
                                'Asymmetrique Profile Index':'Asymmetric_Profile_Index',
                                'Asymmetrique Activity Score':'Asymmetric_Activity_Score',
                                'Asymmetrique Profile Score':'Asymmetric_Profile_Score',
                                'A free copy of Mastering The Interview':'Copy_Request'})


# In[8]:


x_lead.head(2)


# #### 3.b Drop unnecessary columns

# In[9]:


x_lead.columns


# In[10]:


x_lead_o=x_lead.select_dtypes(include='object')


# In[11]:


x_lead_n=x_lead.select_dtypes(exclude='object')


# In[12]:


x_lead_o.dtypes


# In[13]:


x_lead_n.dtypes


# #### 3.c Dropping unnecessary columns 

# In[14]:


# Let us look in to the object dataframe 

#Initially we remove 'Prospect ID' column from x_lead_o dataframe

x_lead_o=x_lead_o.drop(columns=['Prospect ID'],axis=1)


# In[15]:


for col in x_lead_o.columns:
    print(x_lead_o[col].value_counts())
    print('*'*60)    


# In[16]:


# Dropping columns which have disproportional values in it

x_lead_o=x_lead_o.drop(columns=['Magazine','Newspaper','X Education Forums',
                                'Digital Advertisement','Through Recommendations',
                               'Receive More Updates About Our Courses','Supply_Chain_Updates',
                               'DM_Updates','Cheque_Payment','Do_Not_Call',
                               ],axis=1)


# In[17]:


x_lead_o.isna().sum()


# In[18]:


x_lead_o.isna().sum()


# In[19]:


# Imputing the null values with mode values
for col in x_lead_o.columns:
    x_lead_o[col]=x_lead_o[col].fillna(x_lead_o[col].mode()[0])


# In[20]:


x_lead_o.isna().sum()


# In[21]:


x_lead_o.head()


# #### 3.d Aggregating the values of Columns to classification reduction

# In[22]:


# Lets convert 'Select' to null value because 'Select' is the value use to choose the value for query

for col in x_lead_o.columns:
    print(x_lead_o[col].value_counts())
    print('*'*50)


# In[23]:


origin_list=[]
for row in x_lead_o.Lead_Origin:
    if row!='Landing Page Submission' and row!='API':
        origin_list.append(row)
set_origin=set(origin_list)
uniq_origin_list=list(set_origin)
uniq_origin_list


# In[24]:


x_lead_o.Lead_Origin=x_lead_o.Lead_Origin.replace(uniq_origin_list,'Other_online_gateways')


# In[25]:


x_lead_o['Lead_Origin'].value_counts()


# In[26]:


engine=['Google','Direct Traffic','Olark Chat','Organic Search']
source_list=[]
for row in x_lead_o.Lead_Source:
    if row not in engine:
        source_list.append(row)
set_source=set(source_list)
uniq_source_list=list(set_source)
uniq_source_list


# In[27]:


x_lead_o.Lead_Source=x_lead_o.Lead_Source.replace(uniq_source_list,'Other_References')


# In[28]:


x_lead_o['Lead_Source'].value_counts()


# In[29]:


c_list=[]
for row in x_lead_o.Country:
    if row!='India':
        c_list.append(row)
set_country=set(c_list)
uniq_list=list(set_country)
uniq_list


# In[30]:


#for i in range(len(uniq_list)):
x_lead_o.Country=x_lead_o.Country.replace(uniq_list,'Other_Countries')


# In[31]:


x_lead_o.Country


# In[32]:


x_lead_o['Country'].value_counts()


# In[33]:


tags_list=[]
for row in x_lead_o.Tags:
    if row!='Will revert after reading the email' and row!='Ringing':
        tags_list.append(row)
tags_set=set(tags_list)
uniq_tag_list=list(tags_set)


# In[34]:


x_lead_o['Tags']=x_lead_o['Tags'].replace(uniq_tag_list,'Other_than_email_and_phonecall')


# In[35]:


x_lead_o.Tags.value_counts()


# In[36]:


q_list=[]
for row in x_lead_o.Lead_Quality:
    if row!='Might be' and row!='Not Sure':
        q_list.append(row)
q_set=set(q_list)
uniq_q_list=list(q_set)


# In[37]:


x_lead_o['Lead_Quality']=x_lead_o['Lead_Quality'].replace(uniq_q_list,'Other_than_Might_be_and_Not_Sure')


# In[38]:


x_lead_o.Lead_Quality.value_counts()


# In[39]:


o_list=[]
for row in x_lead_o.Current_Occupation:
    if row!='Unemployed' and row!='Working Professional':
        o_list.append(row)
o_set=set(o_list)
uniq_o_list=list(o_set)


# In[40]:


x_lead_o['Current_Occupation']=x_lead_o['Current_Occupation'].replace(uniq_o_list,'Others')


# In[41]:


x_lead_o.Current_Occupation.value_counts()


# In[42]:


p_list=[]
for row in x_lead_o.Lead_Profile:
    if row!='Potential Lead' and row!='Select':
        p_list.append(row)
p_set=set(p_list)
uniq_p_list=list(p_set)


# In[43]:


x_lead_o['Lead_Profile']=x_lead_o['Lead_Profile'].replace(uniq_p_list,'Other Leads')


# In[44]:


x_lead_o.Lead_Profile.value_counts()


# In[45]:


promo_list=[]
for row in x_lead_o.X_Education_Promotion:
    if row!='Select' and row!='Online Search':
        promo_list.append(row)
promo_set=set(promo_list)
uniq_promo_list=list(promo_set)


# In[46]:


x_lead_o['X_Education_Promotion']=x_lead_o['X_Education_Promotion'].replace(uniq_promo_list,'Offline_Search')


# In[47]:


x_lead_o.X_Education_Promotion.value_counts()


# #### 3.e Imputing 'Select' values as null and with mean and mode values

# In[48]:


cols=['City','Specialization','X_Education_Promotion','Lead_Profile']

for col in cols:
    x_lead_o[col]=x_lead_o[col].replace('Select',np.nan)
    x_lead_o[col]=x_lead_o[col].replace(' ',np.nan)


# In[49]:


x_lead_o.isna().sum()


# In[50]:


x_lead_o['X_Education_Promotion'].mode()[0]


# In[51]:


x_lead_o['Lead_Profile'].mode()[0]


# In[52]:


x_lead_o['Specialization'].mode()[0]


# In[53]:


x_lead_o['Lead_Quality'].value_counts()


# In[54]:


# x_lead_o.replace(['Might be','Not Sure'],'Not_Confident',inplace=True)
# x_lead_o.replace(['Other_than_Might_be_and_Not_Sure'],'Confident',inplace=True)


# In[55]:


for col in cols:
    x_lead_o[col]=x_lead_o[col].fillna(x_lead_o[col].mode()[0])


# In[56]:


x_lead_o.isna().sum()


# In[57]:


x_lead_o['Asymmetric_Activity_Index'].replace({"01.High": "HIGH",
                       "02.Medium": "MEDIUM",
                       "03.Low": "LOW"},inplace = True) 


# In[58]:


x_lead_o['Asymmetric_Profile_Index'].replace({"01.High": "HIGH",
                       "02.Medium": "MEDIUM",
                       "03.Low": "LOW"},inplace = True) 


# In[59]:


x_lead_o['Asymmetric_Profile_Index'].head(5)


# In[60]:


x_lead_o.isna().sum()


# In[61]:


# Now let us consider numeric dataframe

x_lead_n.dtypes


# In[62]:


x_lead_n.isna().sum()


# In[63]:


x_lead_n.Asymmetric_Activity_Score.head()


# In[64]:


x_lead_n.Asymmetric_Activity_Score.mean()


# In[65]:


x_lead_n.Asymmetric_Profile_Score.mean()


# In[66]:


x_lead_n['Asymmetric_Activity_Score'].fillna(round(x_lead_n['Asymmetric_Activity_Score'].mean(),1), inplace=True)

x_lead_n['Asymmetric_Profile_Score'].fillna(round(x_lead_n['Asymmetric_Profile_Score'].mean(),1), inplace=True)


# In[67]:


x_lead_n.head(2)


# In[68]:


# Lets check whether dataframe possess null values
x_lead_n.isnull().values.sum()


# In[69]:


# Lets remove those null values from the dataframe
x_lead_n=x_lead_n.dropna()


# In[70]:


# Now check again for null values
x_lead_n.isnull().values.sum()


# In[71]:


x_lead_n.drop(['Lead_Number'],axis=1,inplace=True)


# In[72]:


x_lead_n.head()


# In[73]:


x_lead_n.shape


# In[74]:


x_lead_n['Prospect ID']=x_lead['Prospect ID']


# In[75]:


x_lead_n.isna().sum()


# In[76]:


x_lead_o.shape


# In[77]:


x_lead_o['Prospect ID']=x_lead['Prospect ID']


# In[78]:


x_lead_o.isna().sum()


# In[79]:


# Merge the two dataframes 

x_lead_m=x_lead_o.merge(x_lead_n,how='left')


# In[80]:


x_lead_m.isna().sum()


# In[81]:


x_lead_m=x_lead_m.dropna()


# In[82]:


x_lead_m.isna().sum()


# In[83]:


x_lead_m.head()


# In[84]:


# Let us remove the unnecessary variables 

x_lead_m.drop(['Prospect ID'], axis = 1, inplace = True) 


# #### 3.f For categorical variables with multiple levels, create dummy features

# In[85]:


o_list=[]
for col in x_lead_m.columns:
    if x_lead_m[col].dtype=='object':
        o_list.append(col)
o_list


# In[86]:


# Creating a dummy variable for some of the categorical variables and dropping the first one.
dummy_x = pd.get_dummies(x_lead_m[o_list], drop_first=True)

# Adding the results to the master dataframe
x_lead_dum = pd.concat([x_lead_m, dummy_x], axis=1)


# In[87]:


x_lead_dum.head()


# #### Dropping the repeated variables

# In[88]:


x_lead_dum = x_lead_dum.drop(columns=o_list,axis=1)


# In[89]:


x_lead_dum.columns


# In[90]:


x_lead_dum .shape


# In[91]:


x_lead_dum.info()


# #### 3.g Checking for Outliers

# In[92]:


x_lead_dum.select_dtypes(include='float').columns


# In[93]:


num_col_list=list(x_lead_dum.select_dtypes(include='float').columns)


# In[94]:


# Checking for outliers in the continuous variables
num_x_lead = x_lead_dum[num_col_list]


# In[95]:


# Checking outliers at 25%, 50%, 75%, 90%, 95% and 99%
num_x_lead.describe(percentiles=[.25, .5, .75, .90, .95, .99])


# In[96]:


# We observe outliers in the column 'Time_Spent_on_Website'

low_qua=x_lead_dum['Time_Spent_on_Website'].quantile(0.20)
hi_qua=x_lead_dum['Time_Spent_on_Website'].quantile(0.95)


# In[97]:


low_qua


# In[98]:


hi_qua


# In[99]:


x_lead_dum=x_lead_dum[(x_lead_dum['Time_Spent_on_Website']>low_qua) & (x_lead_dum['Time_Spent_on_Website']<hi_qua)]


# In[100]:


x_lead_dum.isna().sum()


# In[101]:


x_lead_dum['Time_Spent_on_Website']=x_lead_dum['Time_Spent_on_Website'].fillna(x_lead_dum['Time_Spent_on_Website'].mean())


# Lets check whether there is null values present in the dataframe

# In[102]:


x_lead_dum.isna().values.sum()


# We don't have any missing values

# In[103]:


x_lead_dum['Time_Spent_on_Website']=x_lead_dum['Time_Spent_on_Website'].round(2)


# In[104]:


x_lead_dum['Time_Spent_on_Website'].head()


# In[105]:


### Checking the Convert Rate
convert = (sum(x_lead_dum['Converted'])/len(x_lead_dum['Converted'].index))*100
convert


# We have almost 34.6% convert rate

# ## Step 4: Looking at Correlations

# In[106]:


# Importing matplotlib and seaborn
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[107]:


# Let's see the correlation matrix 
plt.figure(figsize = (30,50))        # Size of the figure
sns.heatmap(x_lead_dum.corr(),annot = True)
plt.show()


# In[108]:


X=x_lead_dum.drop(['Converted'],axis=1)
y=x_lead_dum['Converted']


# #### 4.a Dropping highly correlated dummy variables

# In[109]:


cor_matrix = X.corr().abs()


# In[110]:


upper_tri = cor_matrix.where(np.triu(np.ones(cor_matrix.shape),k=1).astype(np.bool))


# In[111]:


to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.85)]


# In[112]:


to_drop


# In[113]:


X_no_corr = X.drop(columns=to_drop, axis=1)
X_no_corr.head()


# In[114]:


X_no_corr.head()


# In[115]:


X=X_no_corr


# ## Step 5: Feature Selection Using RFE

# In[116]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier()


# In[117]:


from sklearn.feature_selection import RFE
rfe = RFE(estimator=rfc, n_features_to_select=10)             # running RFE with 15 variables as output
_ = rfe.fit(X, y)


# In[118]:


rfe.support_


# In[119]:


list(zip(X.columns, rfe.support_, rfe.ranking_))


# In[120]:


col = X.columns[rfe.support_]


# In[121]:


col


# In[122]:


X=X.loc[:,rfe.support_]


# In[123]:


X=X.rename(columns={'Last_Activity_SMS Sent':'Last_Activity_SMS_Sent',
                    'Specialization_Finance Management':'Specialization_Finance_Management',
                  'Current_Occupation_Working Professional':'Current_Occupation_Working_Professional',
                    'Tags_Will revert after reading the email':'Tags_Will_revert_after_reading_the_email',
                    'Last Notable Activity_Modified':'Last_Notable_Activity_Modified',
                    'Last Notable Activity_SMS Sent':'Last_Notable_Activity_SMS_Sent'})


# In[124]:


X.columns


# In[125]:


X['Time_Spent_on_Website']=X['Time_Spent_on_Website'].apply(float)


# In[126]:


X['Time_Spent_on_Website'].dtype


# In[127]:


X.head()


# ## Step 6: Test-Train Split

# In[128]:


from sklearn.model_selection import train_test_split


# In[129]:


# Putting feature variable to X


# In[130]:


X.head()


# In[131]:


X.dtypes


# In[132]:


sns.set(style="whitegrid")
plt.figure(figsize=(8,5))
total = float(len(x_lead_dum))
ax = sns.countplot(x=x_lead_dum.Converted, data=x_lead_dum)
plt.title('Converted percentages', fontsize=20)
for p in ax.patches:
    percentage = '{:.1f}%'.format(100 * p.get_height()/total)
    x = p.get_x() + p.get_width()
    y = p.get_height()
    ax.annotate(percentage, (x, y),ha='center')
plt.show()


# In[133]:


# Putting response variable to y
y = x_lead_dum['Converted']

y.head()

y.value_counts()


# In[134]:


# Splitting the data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)


# ## Step 7: Model Building

# In[135]:


# Let's start by splitting our data into a training set and a test set.


# #### Running our First Training Model

# In[136]:


import statsmodels.api as sm


# ### Logistic regression model

# #### 7.a Assessing the model with StatsModels

# In[137]:


#sm.families.Binomial() represents Logistic Regression of Generalized Linear Model
X_train_sm = sm.add_constant(X_train)
logm1 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial()) 
res = logm1.fit()
res.summary()


# In[138]:


X_train = X_train.drop(['Page_Views_Per_Visit'], axis=1)


# In[139]:


X_train_sm = sm.add_constant(X_train)
logm2 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm2.fit()
res.summary()


# In[140]:


X_train = X_train.drop(['TotalVisits'], axis=1)


# In[141]:


X_train_sm = sm.add_constant(X_train)
logm3 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm3.fit()
res.summary()


# In[142]:


len(X_train.columns)


# In[143]:


len(X_test.columns)


# In[144]:


X_train_sm = sm.add_constant(X_train)


# In[145]:


# Getting the predicted values on the train set
y_train_pred = res.predict(X_train_sm)
y_train_pred[:10]


# In[146]:


y_train_pred = np.array(y_train_pred).reshape(-1)
y_train_pred[:10]


# #### 7.b Creating a dataframe with the actual convert flag and the predicted probabilities

# In[147]:


y_train_pred_final = pd.DataFrame({'Converted':y_train.values, 'Converted_Prob':y_train_pred})
y_train_pred_final['Lead_Number'] = y_train.index
y_train_pred_final.head()


# #### 7.c Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0

# In[148]:


y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)

# Let's see the head
y_train_pred_final.head()


# In[149]:


from sklearn import metrics


# In[150]:


# Confusion matrix 
confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
print(confusion)


# In[151]:


# Predicted          not_converted    converted
# Actual
# not_converted        2690            267
# converted            377            1183


# In[152]:


# Let's check the overall accuracy.
print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# #### 7.d Checking VIFs

# In[153]:


# Check for the VIF values of the feature variables. 
from statsmodels.stats.outliers_influence import variance_inflation_factor


# In[154]:


# Create a dataframe that will contain the names of all the feature variables and their respective VIFs
vif = pd.DataFrame()
vif['Features'] = X_train.columns
vif['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# In[155]:


# As VIF values of top two columns are very high they must be removed

X_train=X_train.drop(['Asymmetric_Activity_Score'],axis=1)


# In[156]:


len(X_train.values)


# In[157]:


X_train.shape[1]


# In[158]:


vifu = pd.DataFrame()
vifu['Features'] = X_train.columns
#vif['Features'] = X_train.columns
vifu['VIF'] = [variance_inflation_factor(X_train.values, i) for i in range(X_train.shape[1])]
vifu['VIF'] = round(vifu['VIF'], 2)
vifu = vifu.sort_values(by = "VIF", ascending = False)
vifu


# In[159]:


colm_vifu=X_train.columns.to_list()
colm_vifu


# In[160]:


# Finalized Columns of X_train depending on p-values and VIFs
X_train_sm = sm.add_constant(X_train)


# In[161]:


#X_train_sm = sm.add_constant(X_train)
logm4 = sm.GLM(y_train,X_train_sm, family = sm.families.Binomial())
res = logm4.fit()
res.summary()


# In[162]:


y_train_pred = res.predict(X_train_sm)


# In[163]:


print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[ ]:





# In[164]:


X_test.head()


# In[165]:


X_test=X_test[colm_vifu]


# In[166]:


# Standardization

from sklearn.preprocessing import StandardScaler

scaler=StandardScaler()

X_train_sc=scaler.fit_transform(X_train)


# In[167]:


# Let's re-run the model using the selected variables
X_train_sm_vif = sm.add_constant(X_train_sc)
logm_vif = sm.GLM(y_train,X_train_sm_vif, family = sm.families.Binomial())
res = logm_vif.fit()
res.summary()


# In[168]:


y_train_pred = np.array(res.predict(X_train_sm_vif)).reshape(-1)


# In[169]:


y_train_pred[:10]


# In[170]:


y_train_pred_final['Converted_Prob'] = y_train_pred


# In[171]:


# Creating new column 'predicted' with 1 if Converted_Prob > 0.5 else 0
y_train_pred_final['predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > 0.5 else 0)
y_train_pred_final.head()


# In[172]:


# Let's take a look at the confusion matrix again 

confusion = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion


# In[173]:


# Predicted          not_converted    converted
# Actual
# not_converted        2814            143
# converted            526            1034 


# In[174]:


# Let's check the overall accuracy.

print(metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.predicted))


# In[ ]:





# #### 7.e Metrics beyond simply accuracy

# In[175]:


TP = confusion[1,1] # true positive 
TN = confusion[0,0] # true negatives
FP = confusion[0,1] # false positives
FN = confusion[1,0] # false negatives


# In[176]:


# Let's see the sensitivity of our logistic regression model
print(TP / float(TP+FN))


# In[177]:


# Let us calculate specificity
print(TN / float(TN+FP))


# In[178]:


# Calculate false postive rate - predicting converted when customer does not have converted
print(FP/ float(TN+FP))


# In[179]:


# positive predictive value 
print (TP / float(TP+FP))


# In[180]:


# Negative predictive value
print (TN / float(TN+ FN))


# ## Step 8: Plotting the ROC Curve

# In[181]:


def draw_roc( actual, probs ):
    fpr, tpr, thresholds = metrics.roc_curve( actual, probs,
                                              drop_intermediate = False )
    auc_score = metrics.roc_auc_score( actual, probs )
    plt.figure(figsize=(5, 5))
    plt.plot( fpr, tpr, label='ROC curve (area = %0.2f)' % auc_score )
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate or [1 - True Negative Rate]')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic Curve')
    plt.legend(loc="lower right")
    plt.show()

    return None


# In[182]:


fpr, tpr, thresholds = metrics.roc_curve( y_train_pred_final.Converted, 
                                         y_train_pred_final.Converted_Prob, drop_intermediate = False )


# In[183]:


draw_roc(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)


# ## Step 9: Finding Optimal Cutoff Point

# #### 9.a Optimal cutoff probability is that prob where we get balanced sensitivity and specificity

# In[184]:


# Let's create columns with different lead score probability cutoffs 
lead_score_prob = [float(x)/10 for x in range(10)] 
for i in lead_score_prob:
    y_train_pred_final[i]= y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > i else 0)
y_train_pred_final.head()


# In[185]:


# Now let's calculate accuracy sensitivity and specificity for various lead score probability cutoffs.
cutoff_df = pd.DataFrame( columns = ['Lead_Score_prob','accuracy','sensi','speci'])
from sklearn.metrics import confusion_matrix

# TP = confusion[1,1] # true positive 
# TN = confusion[0,0] # true negatives
# FP = confusion[0,1] # false positives
# FN = confusion[1,0] # false negatives


num = [0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for i in num:
    cm1 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final[i])
    #print('ConfusionMatrix for {}',cm1.format(num))
    total1=sum(sum(cm1))
    #print('Total {}',total1.format(num))
    accuracy = (cm1[0,0]+cm1[1,1])/total1
    
    speci = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    sensi = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    cutoff_df.loc[i] =[ i ,accuracy,sensi,speci]

print(cutoff_df)


# In[186]:


# Let's plot accuracy sensitivity and specificity for various probabilities.
cutoff_df.plot.line(x='Lead_Score_prob', y=['accuracy','sensi','speci'])
plt.show()


# In[187]:


def Find_Optimal_Cutoff(target, predicted):
    false_positive_rate, true_positive_rate, threshold_cutoff = metrics.roc_curve(target, predicted)
    i = np.arange(len(true_positive_rate))
    
    # true_negative_rate = 1-false_positive_rate
    
    roc = pd.DataFrame({'tf' : pd.Series(true_positive_rate-(1-false_positive_rate), index=i),
                        'threshold_cutoff' : pd.Series(threshold_cutoff, index=i)})
    print(roc)
    roc_t = roc.iloc[(roc.tf).abs().argsort()[:1]]
    print(roc_t)

    return list(roc_t['threshold_cutoff'])

threshold_cutoff = Find_Optimal_Cutoff(y_train_pred_final.Converted, 
                                         y_train_pred_final.Converted_Prob)
threshold_cutoff


# In[188]:


print("From the curve above,", round(threshold_cutoff[0],2)," is the optimum point to take it as a cutoff probability.")


# In[189]:


y_train_pred_final['final_predicted'] = y_train_pred_final.Converted_Prob.map(lambda x: 1 if x > round(threshold_cutoff[0],2) else 0)

y_train_pred_final.head()


# In[190]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_train_pred_final.Converted, y_train_pred_final.final_predicted)


# In[191]:


confusion2 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.final_predicted )
confusion2


# In[192]:


TP = confusion2[1,1] # true positive 
TN = confusion2[0,0] # true negatives
FP = confusion2[0,1] # false positives
FN = confusion2[1,0] # false negatives


# In[193]:


# Let's see the sensitivity of our logistic regression model
print(TP / float(TP+FN))


# In[194]:


# Let us calculate specificity
print(TN / float(TN+FP))


# In[195]:


# Calculate false postive rate - predicting converted when customer does not have converted
print(FP/ float(TN+FP))


# In[196]:


# Positive predictive value 
print (TP / float(TP+FP))


# In[197]:


# Negative predictive value
print (TN / float(TN+ FN))


# #### 9.b Precision and Recall

# In[198]:


#Looking at the confusion matrix again


# In[199]:


confusion3 = metrics.confusion_matrix(y_train_pred_final.Converted, y_train_pred_final.predicted )
confusion3


# #### Precision

# TP / TP + FP

# In[200]:


confusion3[1,1]/(confusion3[0,1]+confusion3[1,1])


# #### Recall

# TP / TP + FN

# In[201]:


confusion3[1,1]/(confusion3[1,0]+confusion3[1,1])


# ###### Using sklearn utilities for the same

# In[202]:


from sklearn.metrics import precision_score, recall_score


# In[203]:


precision_score(y_train_pred_final.Converted, y_train_pred_final.predicted)


# In[204]:


recall_score(y_train_pred_final.Converted, y_train_pred_final.predicted)


# #### 9.c Precision and recall tradeoff

# In[205]:


from sklearn.metrics import precision_recall_curve


# In[206]:


y_train_pred_final.Converted, y_train_pred_final.predicted


# In[207]:


precision, recall, thresholds_prc = precision_recall_curve(y_train_pred_final.Converted, y_train_pred_final.Converted_Prob)


# In[208]:


# convert to f score
fscore = (2 * precision * recall) / (precision + recall)
print(fscore)


# In[209]:


# locate the index of the largest f score
ix = argmax(fscore)
print(ix)


# In[210]:


print('Best Threshold=%f, F-Score=%.3f' % (thresholds_prc[ix], fscore[ix]))


# In[211]:


# plot the roc curve for the model
no_skill = len(y_train_pred_final.Converted[y_train_pred_final.Converted==1.0]) / len(y_train_pred_final.Converted)
plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
plt.plot(recall, precision, marker='.', label='Logistic')
plt.scatter(recall[ix], precision[ix], marker='o', color='black', label='Best')

# axis labels
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.legend()

# show the plot
plt.show()


# In[212]:


y_train_pred_final.head()


# In[213]:


y_train=y_train_pred_final[['Converted','Lead_Number','final_predicted']]


# In[214]:


y_train.set_index('Lead_Number')


# In[215]:


X_train.head()


# In[216]:


X_train.index.name = 'Lead_Number'


# In[217]:


Lead_train = pd.merge(X_train, y_train, left_index=True, right_index=True)


# In[218]:


Lead_train.head()


# In[219]:


Lead_train.isna().sum()


# In[220]:


Lead_train.shape


# In[221]:


Lead=Lead_train


# In[222]:


colm_vifu


# In[223]:


ab=[]
for c in Lead.columns:
    if c not in colm_vifu:
        ab.append(c)
ab


# In[224]:


Lead.drop(['Converted'],axis=1,inplace=True)


# In[225]:


Lead.columns


# ## Step 10: Making predictions on the test set

# In[226]:


X_test


# In[227]:


X_test_sc=scaler.transform(X_test)


# In[228]:


colm_vifu


# In[229]:


X_test_sm = sm.add_constant(X_test_sc)


# Making predictions on the test set

# In[230]:


y_test_pred = res.predict(X_test_sm)


# In[231]:


y_test_pred[:10]


# In[232]:


# Converting y_pred to a dataframe which is an array
y_pred_df = pd.DataFrame(y_test_pred)


# In[233]:


# Let's see the head
y_pred_df.head()


# In[234]:


# Converting y_test to dataframe
y_test_df = pd.DataFrame(y_test)


# In[235]:


# Putting Lead_Number to index
y_test_df['Lead_Number'] = y_test_df.index


# In[236]:


# Removing index for both dataframes to append them side by side 
y_pred_df.reset_index(drop=True, inplace=True)
y_test_df.reset_index(drop=True, inplace=True)


# In[237]:


# Appending y_test_df and y_pred_1
y_pred_final = pd.concat([y_test_df, y_pred_df],axis=1)


# In[238]:


y_pred_final.head()


# In[239]:


# Renaming the column 
y_pred_final= y_pred_final.rename(columns={ 0 : 'Converted_Prob'})


# In[240]:


# Rearranging the columns
y_pred_final = y_pred_final.reindex(['Lead_Number','Converted','Converted_Prob'], axis=1)


# In[241]:


# Let's see the head of y_pred_final
y_pred_final.head()


# In[242]:


# Best Threshold = 0.32

y_pred_final['final_predicted'] = y_pred_final.Converted_Prob.map(lambda x: 1 if x > thresholds_prc[ix] else 0)


# In[243]:


y_pred_final.head(10)


# In[244]:


y_pred_final['Converted_Prob_Rounded']=y_pred_final['Converted_Prob'].round(2)


# In[245]:


y_pred_final.head()


# In[246]:


y_pred_final['LEAD_SCORE'] = pd.qcut(y_pred_final['Converted_Prob_Rounded'].rank(method='first'), 101, labels=False)

y_pred_final.head(10)


# In[247]:


## Above Lead Score value that leads to productive calls to students

#Best Threshold is 0.32

lead_score_set=set(y_pred_final['Converted_Prob_Rounded'].to_list())
lead_score_set


# In[248]:


Optimal_Lead_Score=y_pred_final.loc[y_pred_final['Converted_Prob_Rounded'] == 0.32, 'LEAD_SCORE'].iloc[0] 
Optimal_Lead_Score


# In[249]:


# Let's check the overall accuracy.
metrics.accuracy_score(y_pred_final.Converted, y_pred_final.final_predicted)


# In[250]:


confusion4 = metrics.confusion_matrix(y_pred_final.Converted, y_pred_final.final_predicted )
confusion4


# In[251]:


TP = confusion4[1,1] # true positive 
TN = confusion4[0,0] # true negatives
FP = confusion4[0,1] # false positives
FN = confusion4[1,0] # false negatives


# In[252]:


# Let's see the sensitivity of our logistic regression model
print(TP / float(TP+FN))


# In[253]:


# Let us calculate specificity
print(TN / float(TN+FP))


# #### SUBJECTIVE QUESTION AND ANSWERS

# In[254]:


y_subj=y_pred_final[['Lead_Number','final_predicted']]


# In[255]:


X_test.head()


# In[256]:


X_subj=X_test.copy()


# In[257]:


X_subj.head()


# In[258]:


X_subj['Lead_Number']=X_test.index
X_subj = X_subj.reset_index(drop=True)


# In[259]:


x_y_merge=y_subj.merge(X_subj, how = 'inner')


# In[260]:


x_y_merge.head()


# In[261]:


#x_y_merge=x_y_merge.rename(columns={'Last Notable Activity_Email Bounced':'Last_Notable_Activity_Email_Bounced'})


# In[262]:


x_y_merge.head()


# In[263]:


# plotting heatmap to understand exact correlation between variables

plt.figure(figsize=(12,12))
sns.heatmap(x_y_merge.corr(),annot=True);


# In[264]:


y_subj=y_pred_final[['Lead_Number','LEAD_SCORE']]


# In[265]:


Useful_calls = y_subj.loc[y_subj['LEAD_SCORE'] > Optimal_Lead_Score]


# In[266]:


Useful_calls


# In[267]:


print("The team should call to those leads who hold rank above",Optimal_Lead_Score," because they have more probability of getting converted.")


# ### Strategy

# Initially, team should mainly focus on the variable 'final_predicted' along with 'LEAD_SCORE'. That lead should be contacted who have lead score greater than 61.That lead will have higher probability of becoming 'HOT LEAD'. So in this way, team can ring more productive calls rather than wasting valuable time.

# In[ ]:





# In[268]:


Lead.columns


# In[269]:


X=Lead.drop(['Lead_Number', 'final_predicted'],axis=1)


# In[270]:


X.shape


# In[271]:


y=Lead['final_predicted']


# In[272]:


y.shape


# In[273]:


# train-test-split

# X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)


# In[274]:


X_train_sc.shape


# In[275]:


X_test_sc.shape


# In[276]:


y_train.shape


# In[277]:


y_test.shape


# In[278]:


y_train=y_train.drop(['Converted','Lead_Number'],axis=1)


# In[279]:


from sklearn.metrics import accuracy_score


# #### Decision Tree Classifier

# In[280]:


from sklearn.tree import DecisionTreeClassifier


# In[281]:


dt_classifier=DecisionTreeClassifier()
_=dt_classifier.fit(X_train_sc,y_train)


# In[282]:


y_pred_dt=dt_classifier.predict(X_test_sc)
dt_accuracy=round(accuracy_score(y_test,y_pred_dt)*100,2)


# In[283]:


dt_accuracy


# #### Random Forest Classifier

# In[284]:


from sklearn.ensemble import RandomForestClassifier


# In[285]:


rfc_classifier=RandomForestClassifier(n_estimators=20,criterion='entropy',random_state=51)
rfc_classifier.fit(X_train_sc,y_train)


# In[286]:


y_pred_rfc=rfc_classifier.predict(X_test_sc)
rfc_accuracy=round(accuracy_score(y_test,y_pred_rfc)*100,2)


# In[287]:


rfc_accuracy


# #### AdaBoost Classifier

# In[288]:


from sklearn.ensemble import AdaBoostClassifier


# In[289]:


adb_classifier=AdaBoostClassifier(DecisionTreeClassifier(criterion='entropy',random_state=21),
                                 n_estimators=2000,
                                 learning_rate=0.1,
                                 algorithm='SAMME.R',
                                 random_state=1)
adb_classifier.fit(X_train_sc,y_train)


# In[290]:


y_pred_adb=adb_classifier.predict(X_test_sc)
adb_accuracy=round(accuracy_score(y_test,y_pred_adb)*100,2)


# In[291]:


adb_accuracy


# #### XGBoost Classifier

# In[292]:


import xgboost as xgb


# In[293]:


xgb_classifier=xgb.XGBClassifier()
xgb_classifier.fit(X_train_sc,y_train)


# In[294]:


y_pred_xgb=xgb_classifier.predict(X_test_sc)
xgb_accuracy=round(accuracy_score(y_test,y_pred_xgb)*100,2)


# In[295]:


xgb_accuracy


# ### Saving the Model

# In[297]:


import pickle

#dump information to that file
pickle.dump(logm4,open('logmodel.pkl','wb'))

#load a model
pickle.load(open('logmodel.pkl','rb'))


# In[ ]:




