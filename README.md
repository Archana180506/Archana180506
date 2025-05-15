
# MODEL BUILDING
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
X = data.drop('Class', axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,
test_size=0.2)
lr = LogisticRegression(max_iter=1000)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Logistic Regression Report:\n", classification_report(y_test,
y_pred_lr))
print("Random Forest Report:\n", classification_report(y_test,
y_pred_rf))
# VISUALIZATION OF RESULTS
from sklearn.metrics import ConfusionMatrixDisplay, roc_curve
ConfusionMatrixDisplay.from_estimator(rf, X_test, y_test)
plt.title("Confusion Matrix - Random Forest")
plt.show()
fpr, tpr, _ = roc_curve(y_test, rf.predict_proba(X_test)[:, 1])
plt.plot(fpr, tpr, label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
from google.colab import files
uploaded = files.upload()
import pandas as pd
df = pd.read_csv("creditcard.csv")
print(df.head())
import pandas as pd
# Load the CSV file
df = pd.read_csv('creditcard.csv')
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="whitegrid")
plt.figure(figsize=(20, 15))
for i, col in enumerate([f'V{i}' for i in range(1, 7)]):
plt.subplot(2, 3, i + 1)
sns.histplot(data=df, x=col, hue='Class', bins=50, element='step', palette='Set1')
plt.title(f'Histogram of {col}')
plt.tight_layout()
plt.show()
sns.boxplot(data=df, x='Class', y='V1') # replace 'V1' with the variable you want
plt.show()
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df_scaled = pd.DataFrame(scaler.fit_transform(df.drop('Class', axis=1)),columns=df.columns[:-1])
df_scaled['Class'] = df['Class']
plt.figure(figsize=(12,8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()
df.isnull().sum()
# Check duplicates
duplicates = df.duplicated().sum()
print("Duplicate rows:"
, duplicates)
# Remove duplicates (if any)
df = df.drop_duplicates()
