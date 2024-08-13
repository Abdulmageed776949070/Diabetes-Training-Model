import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, roc_curve, auc

# تحميل البيانات
df = pd.read_csv('C:/Users/lenovo/Downloads/Homework 5/diabetes.csv')  #  المسار  لملف البيانات

# استكشاف البيانات
print(df.describe())

# رسم توزيع المتغير المستهدف (Outcome)
plt.figure(figsize=(6, 4))
sns.countplot(df['Outcome'], palette='coolwarm')


# رسم مصفوفة الارتباط
plt.figure(figsize=(10, 8))
corr_matrix = df.corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')


# تقسيم البيانات إلى ميزات (X) والعمود المستهدف (y)
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# تقسيم البيانات إلى مجموعات تدريب واختبار (80% تدريب، 20% اختبار)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# تدريب نموذج Random Forest مع GridSearchCV لتحسين المعلمات
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
rf_model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(rf_model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

# أفضل المعلمات من GridSearch
best_params = grid_search.best_params_
print(f'Best Parameters: {best_params}')

# إجراء التنبؤات على مجموعة الاختبار
y_pred = grid_search.predict(X_test)

# تقييم النموذج
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

print(f'Accuracy: {accuracy}')
print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')

# رسم منحنى ROC-AUC
y_pred_prob = grid_search.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.title('Distribution of Outcome Variable')
plt.title('Correlation Matrix')
plt.show()