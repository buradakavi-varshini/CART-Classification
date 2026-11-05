# CART (Classification and Regression Trees) Implementation in Python

This repository contains a complete Python implementation of a **CART (Classification and Regression Trees)** decision tree model using **scikit-learn**, trained directly on a local CSV dataset file.

## Dataset Used (`dataset.csv`)
* **Dataset Name:** **Iris Dataset**
* **Description:** A classic multivariate dataset consisting of 150 samples of iris flowers from three species (*Iris setosa*, *Iris versicolor*, and *Iris virginica*).
* **Features (4):** 
  - `sepal length (cm)`
  - `sepal width (cm)`
  - `petal length (cm)`
  - `petal width (cm)`
* **Target:** `target` (0: Setosa, 1: Versicolor, 2: Virginica) along with `target_name`.

---

## Project Structure
```text
cart_project/
│
├── dataset.csv       # The Iris dataset in CSV format
├── main.py           # Main Python script loading dataset, training & evaluating CART
├── requirements.txt  # Python package dependencies
└── README.md         # Project documentation
```

---

## Getting Started & Running Locally

1. **Clone the repository:**
   ```bash
   git clone https://github.com/your-username/cart-decision-tree-python.git
   cd cart-decision-tree-python
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the script:**
   ```bash
   python main.py
   ```

---

## Submitting to GitHub

1. Create a new repository on GitHub.
2. Initialize git and push your project:
   ```bash
   git init
   git add .
   git commit -m "Initial commit: CART decision tree implementation with dataset.csv"
   git branch -M main
   git remote add origin https://github.com/your-username/your-repo-name.git
   git push -u origin main
   ```
