import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import stopwords
import nltk
import joblib


nltk.download('stopwords')


df = pd.read_csv('sentiment_analysis.csv')


stop_words = set(stopwords.words('english'))
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))


X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['label'], test_size=0.2, random_state=42)


output_dir = 'output'
os.makedirs(output_dir, exist_ok=True)


models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(probability=True),  
    "Random Forest": RandomForestClassifier(),
    "Naive Bayes": MultinomialNB(),
    "Gradient Boosting": GradientBoostingClassifier()
}


results = {}
for model_name, model in models.items():
    pipeline = make_pipeline(CountVectorizer(), model)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    results[model_name] = accuracy
    print(f'{model_name} Accuracy: {accuracy}')

   
    cm = confusion_matrix(y_test, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'{model_name} Confusion Matrix')
    plt.savefig(os.path.join(output_dir, f'{model_name}_confusion_matrix.png'))  
    plt.close()

    
    report = classification_report(y_test, predictions)
    with open(os.path.join(output_dir, f'{model_name}_classification_report.txt'), 'w') as f:
        f.write(f'{model_name} Classification Report:\n{report}\n')


best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_pipeline = make_pipeline(CountVectorizer(), best_model)


best_pipeline.fit(df['tweet'], df['label'])


joblib.dump(best_pipeline, os.path.join(output_dir, 'best_model.pkl'))
print(f'The best model is {best_model_name} with accuracy {results[best_model_name]}')
