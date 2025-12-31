import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier #random forest model
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression #logistic regression model

df=pd.read_csv('Iris.csv')
print(df.head())
print(df.tail())
print(df.info()) #shows the data types and non-null counts of each column

print(df.describe()) #shows statistical summary of numerical columns

print(df.shape)
print(df.columns)
print(df.dtypes)

print(df.isnull().sum())

# Fill NaNs with the mean of the column(SepalLengthCm)
df['SepalLengthCm'] = df['SepalLengthCm'].fillna(df['SepalLengthCm'].mean())
# Fill NaNs with the mean of the column(PetalLengthCm)
df['PetalLengthCm'] = df['PetalLengthCm'].fillna(df['PetalLengthCm'].mean())

# Fill NaNs with the median of the column(SepalWidthCm)
df['SepalWidthCm'] = df['SepalWidthCm'].fillna(df['SepalWidthCm'].median())
# Fill NaNs with the median of the column(PetalWidthCm)
df['PetalWidthCm'] = df['PetalWidthCm'].fillna(df['PetalWidthCm'].median())

# Fill NaNs with the mode of the column(id)
df['Id'] = df['Id'].fillna(df['Id'].mode()[0])


print(df.isnull().sum())

print(df["Species"].unique())
le = LabelEncoder()
y = le.fit_transform(df["Species"])

#Iris-setosa → 0 Iris-versicolor → 1 Iris-virginica → 2


df = df.drop("Id", axis=1)
print(df.head())
x=df.drop("Species", axis=1)

#data splitting
X_train, X_test, y_train, y_test = train_test_split(
    x, y,
    test_size=0.2,      # 80% for training, 20% for testing
    random_state=42,    # reproducibility
    stratify=y          # IMPORTANT for classification
)

print(X_train.shape, X_test.shape)
print(y_train.shape, y_test.shape)

#random forest model
rf = RandomForestClassifier(
    n_estimators=100,    # number of trees
    random_state=42      # reproducibility
)

rf.fit(X_train, y_train) #train the model
y_pred = rf.predict(X_test) #test the model      

accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Accuracy:", accuracy)

cm = confusion_matrix(y_test, y_pred)
print(cm)

feature_importance = pd.Series(
    rf.feature_importances_,
    index=x.columns
).sort_values(ascending=False)

print(feature_importance)

#logistic regression model

reg = LogisticRegression(
    max_iter=10000, #ensures convergence
    random_state=0
)
reg.fit(X_train, y_train) #train the model

y_pred = reg.predict(X_test) #test the model

print("Logistic Regression Accuracy:",
      accuracy_score(y_test, y_pred))

import pickle
#dumping the model to a pkl file

#saving the label encoder
with open("label_encoder.pkl", "wb") as file:
    pickle.dump(le, file)

#saved the trained random forest model
with open("random_forest_model.pkl", "wb") as file:
    pickle.dump(rf, file)

#saved the trained logistic regression model
with open("logistic_regression_model.pkl", "wb") as file:
    pickle.dump(reg, file)  

#unloading the label encoder from the pkl file
with open("label_encoder.pkl", "rb") as file:
    le_loaded = pickle.load(file)

#unloading the random forest model from the pkl file
with open("random_forest_model.pkl", "rb") as file:
    rf_loaded = pickle.load(file)

#unloading the logistic regression model from the pkl file
with open("logistic_regression_model.pkl", "rb") as file:
    reg_loaded = pickle.load(file)

#testing the loaded random forest model
y_pred_loaded = rf_loaded.predict(X_test)
print("Loaded Model Accuracy(RFM):", accuracy_score(y_test, y_pred_loaded))

#testing the loaded logistic regression model
y_pred_loaded_reg = reg_loaded.predict(X_test)
print("Loaded Model Accuracy(LRM):", accuracy_score(y_test, y_pred_loaded_reg)) 


#getting user input for predictions
sepal_length = float(input("Enter sepal length: "))
sepal_width  = float(input("Enter sepal width: "))
petal_length = float(input("Enter petal length: "))
petal_width  = float(input("Enter petal width: "))

user_input = [[
    sepal_length,
    sepal_width,
    petal_length,
    petal_width
]]

#prediction using loaded random forest model
pred_encoded = rf_loaded.predict(user_input)
pred_label = le.inverse_transform(pred_encoded)

print("\n--- User Input Summary ---")
print(f"Sepal Length : {sepal_length}")
print(f"Sepal Width  : {sepal_width}")
print(f"Petal Length : {petal_length}")
print(f"Petal Width  : {petal_width}")
print("--------------------------\n")
print("Random Forest Model Prediction:", pred_label)
