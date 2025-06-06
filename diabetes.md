```python
# 🩺 Diabetes Prediction using Pima Indians Dataset

# 📦 Step 1: Import Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve

# 📥 Step 2: Load Dataset
url = "https://raw.githubusercontent.com/plotly/datasets/master/diabetes.csv"
df = pd.read_csv(url)

# 👁️ Step 3: Inspect the Data
print("Shape:", df.shape)
print(df.head())
print(df.info())
print(df.describe())
print("\nTarget distribution:\n", df['Outcome'].value_counts())

# 🔍 Step 4: Exploratory Data Analysis
plt.figure(figsize=(6,4))
sns.countplot(x='Outcome', data=df)
plt.title("Diabetes Outcome Distribution")
plt.xticks([0,1], ['No Diabetes', 'Diabetes'])
plt.show()

# Correlation heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Feature Correlation Heatmap")
plt.show()

# ⚙️ Step 5: Data Cleaning (replace 0 with NaN where inappropriate)
cols_with_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
df[cols_with_zeros] = df[cols_with_zeros].replace(0, np.nan)
df.fillna(df.median(), inplace=True)

# 📚 Step 6: Train-Test Split
X = df.drop('Outcome', axis=1)
y = df['Outcome']

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 🧠 Step 7: Train Random Forest Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 📈 Step 8: Evaluation
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("📊 Classification Report:")
print(classification_report(y_test, y_pred))

# Confusion Matrix
plt.figure(figsize=(6,4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# ROC Curve
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = roc_auc_score(y_test, y_proba)

plt.figure(figsize=(6,6))
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid()
plt.show()

# 🔍 Step 9: Feature Importance
feature_importance = pd.DataFrame({
    'Feature': df.columns[:-1],
    'Importance': model.feature_importances_
}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(8,6))
sns.barplot(data=feature_importance, x='Importance', y='Feature')
plt.title("Feature Importance (Random Forest)")
plt.show()

```

    Shape: (768, 9)
       Pregnancies  Glucose  BloodPressure  SkinThickness  Insulin   BMI  \
    0            6      148             72             35        0  33.6   
    1            1       85             66             29        0  26.6   
    2            8      183             64              0        0  23.3   
    3            1       89             66             23       94  28.1   
    4            0      137             40             35      168  43.1   
    
       DiabetesPedigreeFunction  Age  Outcome  
    0                     0.627   50        1  
    1                     0.351   31        0  
    2                     0.672   32        1  
    3                     0.167   21        0  
    4                     2.288   33        1  
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 768 entries, 0 to 767
    Data columns (total 9 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   Pregnancies               768 non-null    int64  
     1   Glucose                   768 non-null    int64  
     2   BloodPressure             768 non-null    int64  
     3   SkinThickness             768 non-null    int64  
     4   Insulin                   768 non-null    int64  
     5   BMI                       768 non-null    float64
     6   DiabetesPedigreeFunction  768 non-null    float64
     7   Age                       768 non-null    int64  
     8   Outcome                   768 non-null    int64  
    dtypes: float64(2), int64(7)
    memory usage: 54.1 KB
    None
           Pregnancies     Glucose  BloodPressure  SkinThickness     Insulin  \
    count   768.000000  768.000000     768.000000     768.000000  768.000000   
    mean      3.845052  120.894531      69.105469      20.536458   79.799479   
    std       3.369578   31.972618      19.355807      15.952218  115.244002   
    min       0.000000    0.000000       0.000000       0.000000    0.000000   
    25%       1.000000   99.000000      62.000000       0.000000    0.000000   
    50%       3.000000  117.000000      72.000000      23.000000   30.500000   
    75%       6.000000  140.250000      80.000000      32.000000  127.250000   
    max      17.000000  199.000000     122.000000      99.000000  846.000000   
    
                  BMI  DiabetesPedigreeFunction         Age     Outcome  
    count  768.000000                768.000000  768.000000  768.000000  
    mean    31.992578                  0.471876   33.240885    0.348958  
    std      7.884160                  0.331329   11.760232    0.476951  
    min      0.000000                  0.078000   21.000000    0.000000  
    25%     27.300000                  0.243750   24.000000    0.000000  
    50%     32.000000                  0.372500   29.000000    0.000000  
    75%     36.600000                  0.626250   41.000000    1.000000  
    max     67.100000                  2.420000   81.000000    1.000000  
    
    Target distribution:
     Outcome
    0    500
    1    268
    Name: count, dtype: int64
    


    
![png](output_0_1.png)
    



    
![png](output_0_2.png)
    


    📊 Classification Report:
                  precision    recall  f1-score   support
    
               0       0.80      0.88      0.84       100
               1       0.73      0.59      0.65        54
    
        accuracy                           0.78       154
       macro avg       0.76      0.74      0.75       154
    weighted avg       0.77      0.78      0.77       154
    
    


    
![png](output_0_4.png)
    



    
![png](output_0_5.png)
    



    
![png](output_0_6.png)
    



```python

```
