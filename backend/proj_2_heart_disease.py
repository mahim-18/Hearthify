#!/usr/bin/env python
# coding: utf-8

# # Importing Dataset

# In[2]:


import pandas as pd
heart = pd.read_csv("Heart_Disease_Prediction.csv")
# Sex: 1:male, 0:female
# Target: 1:Presence, 0:Absence


# In[3]:


heart.describe()


# In[4]:


heart.info()


# In[5]:


heart.head()


# ## Converting text to numbers

# In[7]:


heart['Target'] = heart['Heart Disease'].apply(lambda x: 1 if x == 'Presence' else 0)


# In[8]:


heart.drop(['Heart Disease'], axis=1, inplace=True)


# In[9]:


heart.head()


# # Splitting data into Train and Test

# In[11]:


from sklearn.model_selection import train_test_split
heart_train, heart_test = train_test_split(heart, test_size=0.2, random_state=42)


# In[12]:


heart['Sex'].value_counts() 


# ## Equally dividing male and females into Train and Test data

# In[14]:


from sklearn.model_selection import StratifiedShuffleSplit
strat_split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for heart_train_index, heart_test_index in strat_split.split(heart, heart['Sex']):
    strat_heart_train = heart.loc[heart_train_index] 
    strat_heart_test = heart.loc[heart_test_index] 


# In[15]:


heart = strat_heart_train.copy()
heart.describe()


# # Finding Correlations

# In[17]:


corr_matrix = heart.corr()


# In[18]:


corr_matrix['Target'].sort_values(ascending=False)


# In[19]:


from pandas.plotting import scatter_matrix
attributes=['Target', 'Exercise angina', 'Number of vessels fluro', 'Thallium']

scatter_matrix(heart[attributes], figsize=(20,15))


# In[20]:


train_features = strat_heart_train.drop('Target', axis=1)
test_features = strat_heart_test.drop('Target', axis=1)


# In[21]:


train_labels = strat_heart_train['Target']
test_labels = strat_heart_test['Target']


# In[22]:


from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
my_pip = Pipeline([
    ('std_scaler', StandardScaler()),
])


# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
# model = LogisticRegression()
model = SVC(probability=True)

heart_train_features = my_pip.fit_transform(train_features)
heart_test_features = my_pip.fit_transform(test_features)


# In[24]:


heart_train_features


# In[25]:


model.fit(heart_train_features, train_labels)


# In[26]:


predicted_on_train = model.predict(heart_train_features)


# In[27]:


predicted_on_test = model.predict(heart_test_features)


# In[28]:


from sklearn.metrics import accuracy_score


# In[29]:


print(f"Accuracy on train data : {accuracy_score(predicted_on_train, train_labels)}")


# In[30]:


print(f"Accuracy on test data : {accuracy_score(predicted_on_test, test_labels)}")


# In[31]:


heart_test_features


# In[32]:


# from sklearn.linear_model import LogisticRegression
# from sklearn.tree import DecisionTreeClassifier
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# from sklearn.neighbors import KNeighborsClassifier
# from sklearn.ensemble import GradientBoostingClassifier
# from sklearn.metrics import accuracy_score

# # Define models
# models = {
#     "Logistic Regression": LogisticRegression(),
#     "Decision Tree": DecisionTreeClassifier(),
#     "Random Forest": RandomForestClassifier(),
#     "SVM": SVC(),
#     "KNN": KNeighborsClassifier(),
#     "Gradient Boosting": GradientBoostingClassifier()
# }

# # Train and evaluate models
# for name, model in models.items():
#     model.fit(heart_test_features, test_labels)  # Training
#     # score = model.score(heart_test_features, heart_test_labels)  # Accuracy
#     predicted_on_test = model.predict(heart_test_features)
#     print(f"Accuracy on test data using {name} : {accuracy_score(predicted_on_test, test_labels)}")
#     # print(f"{name}: {score:.4f}")


# In[33]:


# Accuracy on train data using Logistic Regression : 0.8657407407407407
# Accuracy on train data using Decision Tree : 1.0
# Accuracy on train data using Random Forest : 1.0
# Accuracy on train data using SVM : 0.9398148148148148
# Accuracy on train data using KNN : 0.8888888888888888
# Accuracy on train data using Gradient Boosting : 1.0


# Accuracy on test data using Logistic Regression : 0.8703703703703703
# Accuracy on test data using Decision Tree : 1.0
# Accuracy on test data using Random Forest : 1.0
# Accuracy on test data using SVM : 0.9259259259259259
# Accuracy on test data using KNN : 0.7777777777777778
# Accuracy on test data using Gradient Boosting : 1.0


# In[34]:


heart_train_features[0]


# In[35]:


train_features.head()


# In[36]:


train_labels.head()


# In[37]:


# model.predict([[ 0.99042285, -1.44420022,  0.90628393, -1.24010722,  0.40612837,
#        -0.42465029, -0.99540267,  0.77010209,  1.5411035 ,  0.67817444,
#         0.67293042,  1.56790588, -0.81749667]])

model.predict(heart_train_features[3].reshape(1, -1))


# In[38]:


# heart_test_features


# In[39]:


test_features.head()


# In[40]:


test_labels.value_counts()


# In[41]:


final_prediction = model.predict([[ 0.65511222,  0.67783439, -0.36927447, -0.11532929, -0.4770329 ,
            -0.38592249, -1.16095912,  0.20759172, -0.92847669,  0.56935897,
            0.69652603,  1.90559695,  0.95221954]])


# In[42]:


if final_prediction[0] == 1 :
    print("The Person has Heart Disease")
else :
    print("The Person DOES NOT have Heart Disease")


# In[43]:


import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Predict on test data
y_pred = model.predict(heart_test_features)

# Compute the confusion matrix
cm = confusion_matrix(test_labels, y_pred)

# Display the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")


# In[44]:


if hasattr(model, "predict_proba"):
    probabilities = model.predict_proba([heart_train_features[0]])[0]  # Get probabilities
    classes = model.classes_  # Get class labels

    # Plot bar chart
    plt.figure(figsize=(6, 4))
    plt.bar(classes, probabilities, color=['blue', 'red'])
    plt.xlabel("Class")
    plt.ylabel("Probability")
    plt.title("Prediction Confidence for a Single Sample")
    plt.ylim(0, 1)  # Set y-axis limit between 0 and 1
    plt.xticks(classes)  # Ensure correct class labels
    plt.show()
else:
    print("Model does not support probability predictions. Re-train with probability=True in SVC().")


# In[85]:




# In[ ]:




import joblib

joblib.dump(model, "heart_disease_model.pkl")