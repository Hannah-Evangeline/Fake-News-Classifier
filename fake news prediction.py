import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load datasets
fake = pd.read_csv("Fake.csv")
real = pd.read_csv("True.csv")

# Step 2: Add labels
fake["label"] = 0  # Fake news
real["label"] = 1  # Real news

# Step 3: Combine
data = pd.concat([fake, real], axis=0)
data = data[["text", "label"]]  # We’ll use only the "text" column

# Step 4: Split
X_train, X_test, y_train, y_test = train_test_split(
    data["text"], data["label"], test_size=0.2, random_state=42
)

# Step 5: TF-IDF Vectorizer
tfidf = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_vec = tfidf.fit_transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Step 6: Logistic Regression
model = LogisticRegression()
model.fit(X_train_vec, y_train)
y_pred = model.predict(X_test_vec)

# Step 7: Report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Step 8: Confusion Matrix Plot
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=["Fake", "Real"], yticklabels=["Fake", "Real"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()


from sklearn.metrics import accuracy_score

acc = accuracy_score(y_test, y_pred)
plt.bar(['Accuracy'], [acc], color='lightpink')
plt.title('Model Accuracy')
plt.ylim(0, 1)
plt.ylabel('Accuracy')
plt.show()


import seaborn as sns
import matplotlib.pyplot as plt

sns.countplot(x="label", data=data)
plt.title("Fake (0) vs Real (1) News Count")
plt.xlabel("News Type")
plt.ylabel("Count")
plt.xticks([0, 1], ['Fake', 'Real'])
plt.show()
