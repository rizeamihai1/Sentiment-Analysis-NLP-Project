import json
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score

# Configuration
DATASET_PATH = r"C:\Research\NLP_FAC\Datasets\IMDB"
TEXT_COLUMN = "text_preprocessed"
LABEL_COLUMN = "label"
SEED = 42

def load_json_data(split_name):
    path = os.path.join(DATASET_PATH, f"{split_name}.json")
    with open(path, "r", encoding="utf-8") as f:
        return pd.DataFrame(json.load(f))

def main():
    print("Loading data...")
    df_train = load_json_data("train")
    df_test = load_json_data("test")
    
    # Top 5 configurations from grid_search_results.txt
    top_configs = [
        {"C": 10.0, "solver": "lbfgs",     "ngram_range": (1, 2), "max_features": 50000},
        {"C": 10.0, "solver": "lbfgs",     "ngram_range": (1, 2), "max_features": 100000},
        {"C": 10.0, "solver": "liblinear", "ngram_range": (1, 2), "max_features": 50000},
        {"C": 10.0, "solver": "lbfgs",     "ngram_range": (1, 3), "max_features": 200000},
        {"C": 10.0, "solver": "liblinear", "ngram_range": (1, 2), "max_features": 100000},
    ]
    
    test_accuracies = []
    
    print("\nEvaluating top 5 configurations on test data...")
    for i, config in enumerate(top_configs, 1):
        print(f"Running Model {i} with config: {config}")
        
        tfidf = TfidfVectorizer(
            ngram_range=config["ngram_range"],
            max_features=config["max_features"],
            sublinear_tf=True,
            strip_accents="unicode"
        )
        
        X_train = tfidf.fit_transform(df_train[TEXT_COLUMN].fillna("").values)
        X_test = tfidf.transform(df_test[TEXT_COLUMN].fillna("").values)
        
        y_train = df_train[LABEL_COLUMN].values
        y_test = df_test[LABEL_COLUMN].values
        
        model = LogisticRegression(
            C=config["C"],
            solver=config["solver"],
            max_iter=500,
            random_state=SEED
        )
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred)
        test_accuracies.append(acc)
        print(f"Model {i} Test Accuracy: {acc:.4f}")

    # Plotting
    models = [f"Model{i}" for i in range(1, 6)]
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    plt.figure(figsize=(10, 8))
    bars = plt.bar(models, test_accuracies, color=colors)
    
    plt.title("Comparatie acuratete modele", fontsize=16)
    plt.xlabel("Modele", fontsize=12)
    plt.ylabel("Acuratete", fontsize=12)
    plt.ylim(0, 1.0) # Accuracy is between 0 and 1
    
    # Add accuracy values on top of bars
    for bar, acc in zip(bars, test_accuracies):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                 f'{acc:.4f}', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plot_path = os.path.join(os.getcwd(), "comparatie_acuratete_modele.png")
    plt.savefig(plot_path)
    print(f"\nPlot saved to: {plot_path}")
    plt.show()

if __name__ == "__main__":
    main()
