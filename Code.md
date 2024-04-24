```python
pip install imbalanced-learn
```
### Importing necessary libraries
```python
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix,accuracy_score,precision_score
from sklearn import metrics
```
```python

# Load the credit card fraud dataset
df = pd.read_csv('creditcard.csv')
```
### Data Preprocessing
```python
X = df.drop('Class', axis=1)
y = df['Class']
```
```python
nan_indices = y_train.index[y_train.isnull()]
X_train_cleaned = X_train.drop(index=nan_indices)
y_train_cleaned = y_train.drop(index=nan_indices)
```

```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
```python
print("Class distribution before SMOTE:")
print(y_train.value_counts())
```
### Balancing the Dataset

```pyton
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
```
```pyton
print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())
```
```pyton
print("\nClass distribution after SMOTE:")
print(pd.Series(y_resampled).value_counts())
```
### Training model using logistic regression
```pyton
from sklearn.linear_model import LogisticRegression
```
```pyton
logreg = LogisticRegression(max_iter=1000, random_state=42)
logreg.fit(X_resampled, y_resampled)
```
```python
y_pred = logreg.predict(X_test)
```
```python
logistic_accuracy = accuracy_score(y_test, y_pred)
print("Logistic Regression Accuracy:", logistic_accuracy)
```
```python
logistic_precision = precision_score(y_test, y_pred)
print("Logistic Regression Precision:", logistic_precision)
```
```python

cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
cnf_matrix
```
```python
logistic_conf_matrix = confusion_matrix(y_test, y_pred)
print("Logistic Regression Confusion Matrix:")
print(logistic_conf_matrix)
```
```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
```
### Calssification report of Log_regression
```python
```
### training using Rf classifier
```python
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train_balanced, y_train_balanced)
```
```python
y_pred = rf_classifier.predict(X_test)
```python
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
### Classification report of rf calssifier
```python
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```




### training using decision tree
```python
clf = DecisionTreeClassifier(criterion="entropy")
```
```python
clf.fit(X_train_balanced, y_train_balanced)
```
```python
y_pred1 = clf.predict(X_test)
```
```python
print('Training set score: {:.4f}'.format(clf.score(X_train_balanced, y_train_balanced)))
print('Test set score: {:.4f}'.format(clf.score(X_test, y_test)))
```
```python
plt.figure(figsize=(12,8))
from sklearn import tree
tree.plot_tree(clf.fit(X_train_balanced, y_train_balanced))
```
```python
cnf_matrix=confusion_matrix(y_test, y_pred1)
```
```python
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import pandas as pd
%matplotlib inline
class_names=[0,1] # name  of classes
fig, ax = plt.subplots()
tick_marks = np.arange(len(class_names))
plt.xticks(tick_marks, class_names)
plt.yticks(tick_marks, class_names)
# create heatmap
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g')
ax.xaxis.set_label_position("top")
plt.tight_layout()
plt.title('Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
```
```python
print("\nClassification Report of decision tree(entropy):")
print(classification_report(y_test, y_pred1))
```
```python
d_accuracy = accuracy_score(y_test, y_pred1)
print("rf_classifier Accuracy:", d_accuracy)
```
