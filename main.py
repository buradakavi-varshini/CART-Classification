import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.metrics import accuracy_score, classification_report

def main():
    print("--- CART (Classification and Regression Trees) Implementation ---")
    
    # 1. Load Dataset from local CSV file
    print("Loading dataset from dataset.csv...")
    df = pd.read_csv("dataset.csv")
    
    # Separate features and target
    X = df[['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']]
    y = df['target']
    target_names = ['setosa', 'versicolor', 'virginica']
    
    print(f"Dataset Shape: {X.shape}")
    print(f"Classes: {target_names}")
    
    # 2. Split Dataset into Training and Testing Sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # 3. Initialize and Train CART Model (Decision Tree Classifier using Gini)
    print("\nTraining Decision Tree (CART) Classifier...")
    clf = DecisionTreeClassifier(criterion='gini', max_depth=3, random_state=42)
    clf.fit(X_train, y_train)
    
    # 4. Evaluate Model
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy: {acc * 100:.2f}%")
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))
    
    # 5. Display Tree Rules
    print("\nDecision Tree Rules (Text Representation):")
    tree_rules = export_text(clf, feature_names=list(X.columns))
    print(tree_rules)

if __name__ == "__main__":
    main()
