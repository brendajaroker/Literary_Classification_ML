"""
train_models.py

This script trains and evaluates machine learning models (K-Means, KNN, SVM, Random Forest, Logistic Regression)
to classify literary works into movements based on stylistic features.

The script loads feature data from a CSV file, trains supervised models, and performs unsupervised K-Means clustering.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import argparse
import os
import warnings
import joblib
from datetime import datetime

# Suppress warnings
warnings.filterwarnings('ignore')

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train models for literary movement classification')
    parser.add_argument('--data', type=str, default='text_features.csv', 
                        help='Path to the features CSV file')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results and models')
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.15,
                        help='Proportion of training data to use for validation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--pca_components', type=int, default=30,
                        help='Number of PCA components to use')
    parser.add_argument('--k_clusters', type=int, default=7,
                        help='Number of clusters for K-Means')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    return parser.parse_args()

def load_and_preprocess_data(data_path, random_state):
    """
    Load and preprocess the feature data.
    
    Args:
        data_path: Path to the CSV file containing features
        random_state: Random seed for reproducibility
        
    Returns:
        Preprocessed data as pandas DataFrames
    """
    print("Loading data from:", data_path)
    df = pd.read_csv(data_path)
    
    print(df['Movement'].value_counts())
    
    X = df.drop(['GutenbergID', 'Author', 'Movement'], axis=1)
    y = df['Movement']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    feature_names = X.columns
    movement_names = le.classes_
    
    X = X.fillna(0)
    
    return X, y_encoded, feature_names, movement_names, df

def create_data_splits(X, y, test_size, val_size, random_state):
    """
    Create train/validation/test splits.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data for testing
        val_size: Proportion of training data for validation
        random_state: Random seed
    
    Returns:
        Data splits as numpy arrays
    """
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=val_size/(1-test_size), 
        random_state=random_state, stratify=y_trainval
    )
    
    print(f"Training set: {X_train.shape[0]} samples")
    print(f"Validation set: {X_val.shape[0]} samples")
    print(f"Test set: {X_test.shape[0]} samples")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def scale_features(X_train, X_val, X_test):
    """
    Scale features using StandardScaler.
    
    Args:
        X_train, X_val, X_test: Data splits
    
    Returns:
        Scaled data
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def reduce_dimensionality(X_train_scaled, X_val_scaled, X_test_scaled, n_components, random_state):
    """
    Reduce dimensionality using PCA.
    
    Args:
        X_train_scaled, X_val_scaled, X_test_scaled: Scaled data
        n_components: Number of PCA components
        random_state: Random seed
    
    Returns:
        Reduced data and PCA object
    """
    pca = PCA(n_components=n_components, random_state=random_state)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_val_pca = pca.transform(X_val_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"PCA with {n_components} components explains {explained_variance:.2%} of variance")
    
    return X_train_pca, X_val_pca, X_test_pca, pca

def train_supervised_models(X_train, y_train, X_val, y_val, random_state):
    """
    Train supervised models with GridSearchCV.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        random_state: Random seed
    
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    model_params = {
        'logistic': {
            'model': LogisticRegression(max_iter=10000, random_state=random_state),
            'params': {'C': [0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=random_state),
            'params': {'n_estimators': [100, 200], 'max_depth': [None, 10, 20, 30]}
        },
        'svm': {
            'model': SVC(probability=True, random_state=random_state),
            'params': {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
        },
        'knn': {
            'model': KNeighborsClassifier(),
            'params': {'n_neighbors': [3, 5, 7, 11], 'weights': ['uniform', 'distance']}
        }
    }
    
    for name, model_info in model_params.items():
        print(f"\nTraining {name}...")
        model = model_info['model']
        params = model_info['params']
        
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        
        grid_search = GridSearchCV(
            model, params, cv=cv, scoring='accuracy', n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        val_accuracy = best_model.score(X_val, y_val)
        
        print(f"{name} best parameters: {best_params}")
        print(f"{name} validation accuracy: {val_accuracy:.4f}")
        
        models[name] = {
            'model': best_model,
            'params': best_params,
            'val_accuracy': val_accuracy
        }
    
    return models

def perform_unsupervised_clustering(X_train, y_train, X_val, y_val, n_clusters, random_state):
    """
    Perform unsupervised clustering with KMeans.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        Trained KMeans model
    """
    print("\nPerforming K-means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_train)
    
    silhouette_avg = silhouette_score(X_train, kmeans.labels_)
    print(f"K-means silhouette score: {silhouette_avg:.4f}")
    
    contingency_table = pd.crosstab(y_train, kmeans.labels_)
    print("\nContingency table (true labels vs. cluster assignments):")
    print(contingency_table)
    
    return kmeans

def evaluate_models_on_test_set(models, kmeans, X_test, y_test, movement_names):
    """
    Evaluate all models on the test set.
    
    Args:
        models: Dictionary of supervised models
        kmeans: KMeans clustering model
        X_test, y_test: Test data
        movement_names: Names of literary movements
    
    Returns:
        Dictionary of test results
    """
    results = {}
    
    print("\n=== Test Set Evaluation ===")
    for name, model_dict in models.items():
        model = model_dict['model']
        y_pred = model.predict(X_test)
        test_accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{name} test accuracy: {test_accuracy:.4f}")
        print("Classification report:")
        print(classification_report(y_test, y_pred, target_names=movement_names))
        
        results[name] = {
            'accuracy': test_accuracy,
            'predictions': y_pred,
            'report': classification_report(y_test, y_pred, target_names=movement_names, output_dict=True)
        }
    
    cluster_to_label = {}
    for cluster in range(kmeans.n_clusters):
        cluster_mask = (kmeans.labels_ == cluster)
        if np.any(cluster_mask):
            most_common_label = np.bincount(y_test[kmeans.predict(X_test) == cluster]).argmax()
            cluster_to_label[cluster] = most_common_label
    
    kmeans_preds = np.array([cluster_to_label.get(label, -1) for label in kmeans.predict(X_test)])
    kmeans_accuracy = accuracy_score(y_test, kmeans_preds)
    
    print(f"\nK-means (mapped to movements) test accuracy: {kmeans_accuracy:.4f}")
    print("Classification report:")
    print(classification_report(y_test, kmeans_preds, target_names=movement_names))
    
    results['kmeans'] = {
        'accuracy': kmeans_accuracy,
        'predictions': kmeans_preds,
        'report': classification_report(y_test, kmeans_preds, target_names=movement_names, output_dict=True),
        'cluster_mapping': cluster_to_label
    }
    
    return results

def create_visualizations(X_train_pca, y_train, kmeans, X_test_pca, y_test, movement_names, 
                         results, output_dir, random_state):
    """
    Create visualizations of the results.
    
    Args:
        X_train_pca, y_train: Training data
        kmeans: KMeans clustering model
        X_test_pca, y_test: Test data
        movement_names: Names of literary movements
        results: Dictionary of test results
        output_dir: Directory to save plots
        random_state: Random seed
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    print("\nGenerating t-SNE visualization...")
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=30)
    X_tsne = tsne.fit_transform(np.vstack([X_train_pca, X_test_pca]))
    X_train_tsne = X_tsne[:len(X_train_pca)]
    X_test_tsne = X_tsne[len(X_train_pca):]
    
    plt.figure(figsize=(12, 10))
    for i, movement in enumerate(movement_names):
        plt.scatter(X_train_tsne[y_train == i, 0], X_train_tsne[y_train == i, 1], 
                   label=movement, alpha=0.7)
    plt.title('t-SNE Visualization of Literary Movements')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tsne_movements.png'), dpi=300)
    
    plt.figure(figsize=(12, 10))
    for i in range(kmeans.n_clusters):
        plt.scatter(X_train_tsne[kmeans.labels_ == i, 0], X_train_tsne[kmeans.labels_ == i, 1], 
                   label=f'Cluster {i}', alpha=0.7)
    plt.title('t-SNE Visualization of K-means Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tsne_kmeans_clusters.png'), dpi=300)
    
    best_model_name = max(results, key=lambda k: results[k]['accuracy'] if k != 'kmeans' else 0)
    best_model_preds = results[best_model_name]['predictions']
    
    plt.figure(figsize=(10, 8))
    cm = confusion_matrix(y_test, best_model_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=movement_names, yticklabels=movement_names)
    plt.title(f'Confusion Matrix - {best_model_name.replace("_", " ").title()}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'confusion_matrix_{best_model_name}.png'), dpi=300)
    
    if 'random_forest' in results:
        rf_model = models['random_forest']['model']
        feature_importances = rf_model.feature_importances_
        
        top_indices = np.argsort(feature_importances)[-20:]
        top_features = feature_names[top_indices]
        top_importances = feature_importances[top_indices]
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(len(top_indices)), top_importances, align='center')
        plt.yticks(range(len(top_indices)), top_features)
        plt.xlabel('Feature Importance')
        plt.title('Top 20 Most Important Features (Random Forest)')
        plt.tight_layout()
        plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300)
    
    accuracies = {name: result['accuracy'] for name, result in results.items()}
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(accuracies.keys()), y=list(accuracies.values()))
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'model_accuracy_comparison.png'), dpi=300)


   
    
    

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    X, y, feature_names, movement_names, df = load_and_preprocess_data(args.data, args.random_state)
    
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(
        X, y, args.test_size, args.val_size, args.random_state
    )
    
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    X_train_pca, X_val_pca, X_test_pca, pca = reduce_dimensionality(
        X_train_scaled, X_val_scaled, X_test_scaled, args.pca_components, args.random_state
    )
    
    models = train_supervised_models(X_train_pca, y_train, X_val_pca, y_val, args.random_state)
    
    kmeans = perform_unsupervised_clustering(X_train_pca, y_train, X_val_pca, y_val, args.k_clusters, args.random_state)
    
    results = evaluate_models_on_test_set(models, kmeans, X_test_pca, y_test, movement_names)
    
    if args.visualize:
        create_visualizations(X_train_pca, y_train, kmeans, X_test_pca, y_test, movement_names, 
                             results, args.output_dir, args.random_state)
    
    
    best_model = max(results.items(), key=lambda x: x[1]['accuracy'])
    print(f"\nBest model: {best_model[0]} with accuracy {best_model[1]['accuracy']:.4f}")
    
    print("\nTraining and evaluation complete!")

if __name__ == "__main__":
    main()