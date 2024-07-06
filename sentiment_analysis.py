import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from sklearn.metrics import accuracy_score
import nltk
from nltk.corpus import stopwords
nltk.download('stopwords')

# Lets do it

df = pd.read_csv('sentiment_analysis.csv')


stop_words = set(stopwords.words('english'))
df['tweet'] = df['tweet'].apply(lambda x: ' '.join([word for word in x.split() if word.lower() not in stop_words]))


X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['label'], test_size=0.2, random_state=42)


models = {
    "Logistic Regression": LogisticRegression(),
    "SVM": SVC(),
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


best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
best_pipeline = make_pipeline(CountVectorizer(), best_model)

best_pipeline.fit(df['tweet'], df['label'])


import joblib
joblib.dump(best_pipeline, 'best_model.pkl')
print(f'The best model is {best_model_name} with accuracy {results[best_model_name]}')
