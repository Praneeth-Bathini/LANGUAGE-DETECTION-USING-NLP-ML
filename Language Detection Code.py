import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from tabulate import tabulate
# Load dataset
data = pd.read_csv("https://raw.githubusercontent.com/amankharwal/Website-data/master/dataset.csv")
# Drop missing values
if data.isnull().sum().sum() > 0:
    print("Warning: Dataset contains missing values. Dropping them...\n")
    data.dropna(inplace=True)
# Dataset summary
print("Dataset Summary:")
print(f"Number of rows: {data.shape[0]}")
print(f"Number of columns: {data.shape[1]}")
print(f"Columns: {list(data.columns)}")
print(f"Dataset size (in memory): {data.memory_usage(deep=True).sum() / 1024:.2f} KB\n")
# Show sample data
def truncate_text(text, length=40):
    return text if len(text) <= length else text[:length] + "..."
preview_data = data.copy()
preview_data['Text'] = preview_data['Text'].apply(lambda x: truncate_text(x))
print("Dataset Preview:")
print(tabulate(preview_data.head(), headers='keys', tablefmt='fancy_grid'))
# Language distribution
print("\nLanguage Distribution:")
print(tabulate(data["language"].value_counts().reset_index(), headers=["Language", "Count"], tablefmt="fancy_grid"))
# Prepare text and labels
x = np.array(data["Text"])
y = np.array(data["language"])
# TF-IDF vectorization
cv = TfidfVectorizer()
X = cv.fit_transform(x)
# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# Initialize models
model1 = MultinomialNB()
model2 = LogisticRegression(max_iter=1000)
model3 = RandomForestClassifier(n_estimators=100, random_state=42)
model4 = KNeighborsClassifier()
model5 = SVC()
# Train and evaluate models
models = [model1, model2, model3, model4, model5]
accuracies = []
for idx, model in enumerate(models, start=1):
    model.fit(X_train, y_train)
    accuracy = model.score(X_test, y_test)
    accuracies.append(accuracy)
    print(f"Accuracy of model{idx} ({model.__class__.__name__}): {accuracy * 100:.2f}%")
# Select best model
best_model_index = np.argmax(accuracies)
best_model = models[best_model_index]
print(f"\nBest Model: model{best_model_index + 1} ({best_model.__class__.__name__}) with accuracy: {accuracies[best_model_index] * 100:.2f}%\n")
# Continuous language detection using best model
while True:
    user_input = input("Enter a text to detect its language (or type 'exit' to quit): ")
    if user_input.lower() == 'exit':
        print("\nExiting language detection. Goodbye!\n")
        break
    input_data = cv.transform([user_input]).toarray()
    prediction = best_model.predict(input_data)[0]
    print("\nPrediction Result:")
    print(tabulate([[prediction]], headers=["Detected Language"], tablefmt="fancy_grid"))
    print("\nEnter another text or type 'exit' to stop.\n")