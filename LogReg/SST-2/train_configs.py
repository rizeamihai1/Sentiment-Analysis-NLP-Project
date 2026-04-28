import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

train_df = pd.read_json(r"c:\Research\NLP_FAC\SST-2\train.json")
test_df = pd.read_json(r"c:\Research\NLP_FAC\SST-2\test.json")

X_train_text = train_df["sentence_preprocessed"].fillna("").values
y_train = train_df["label"].values
X_test_text = test_df["sentence_preprocessed"].fillna("").values

configs = [
    ("Config1", 10.0, 'lbfgs', (1, 2), 50000),
    ("Config2", 1.0, 'lbfgs', (1, 1), 50000),
    ("Config3", 1.0, 'lbfgs', (1, 1), 100000),
    ("Config7", 10.0, 'liblinear', (1, 2), 50000),
    ("Config8", 10.0, 'liblinear', (1, 3), 200000)
]

vectorizers = {}
out_dir = r"c:\Research\NLP_FAC\SST-2-Glue"
os.makedirs(out_dir, exist_ok=True)

for config_name, C, solver, ngram_range, max_features in configs:
    vec_key = (ngram_range, max_features)
    if vec_key not in vectorizers:
        print(f"Fitting vectorizer {vec_key}...")
        tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            sublinear_tf=True,
            strip_accents="unicode"
        )
        X_train_tfidf = tfidf.fit_transform(X_train_text)
        X_test_tfidf = tfidf.transform(X_test_text)
        vectorizers[vec_key] = (X_train_tfidf, X_test_tfidf)
    else:
        X_train_tfidf, X_test_tfidf = vectorizers[vec_key]
        
    print(f"Training {config_name} with C={C}, solver={solver}...")
    model = LogisticRegression(
        C=C, 
        solver=solver, 
        max_iter=500, 
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    predictions = model.predict(X_test_tfidf)
    
    out_df = pd.DataFrame({
        "index": range(len(predictions)),
        "prediction": predictions
    })
    
    out_path = os.path.join(out_dir, f"SST-2-{config_name}.tsv")
    out_df.to_csv(out_path, sep="\t", index=False)
    print(f"Saved {out_path}")
