import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.cluster import KMeans
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.metrics.pairwise import pairwise_distances
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
    parser.add_argument('--test_size', type=float, default=0.1,
                        help='Proportion of data to use for testing')
    parser.add_argument('--val_size', type=float, default=0.1111111111111111,
                        help='Proportion of training data to use for validation')
    parser.add_argument('--random_state', type=int, default=42,
                        help='Random seed for reproducibility')
    parser.add_argument('--pca_components', type=int, default=50,
                        help='Number of PCA components to use')
    parser.add_argument('--k_clusters', type=int, default=7,
                        help='Number of clusters for K-Means')
    parser.add_argument('--visualize', action='store_true',
                        help='Generate visualization plots')
    parser.add_argument('--feature_selection', type=str, default='selectkbest',
                        choices=['pca', 'selectkbest', 'model_based', 'none'],
                        help='Feature selection method to use')
    parser.add_argument('--n_features', type=int, default=50,
                        help='Number of features to select with SelectKBest')
    parser.add_argument('--use_ensemble', action='store_true', default=True,
                        help='Use ensemble modeling for better performance')
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
    
    # Handle outliers, extreme values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    numeric_cols = [col for col in numeric_cols if col not in ['GutenbergID']]
    
    # Detect and handle outliers using IQR method
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
        df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
    
    X = df.drop(['GutenbergID', 'Author', 'Movement'], axis=1)
    y = df['Movement']
    
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    feature_names = X.columns.values
    movement_names = le.classes_
    
    X = X.fillna(0)
    
    # Create additional higher-order features
    if X.shape[1] < 100:  # Only create interaction terms if we don't have too many features
        X_interact = create_interaction_terms(X)
        X = pd.concat([X, X_interact], axis=1)
        feature_names = X.columns.values
    
    return X, y_encoded, feature_names, movement_names, df

def create_interaction_terms(X, top_n=10):
    """Create interaction terms for the top features"""
    # For simplicity, let's create interactions between all features
    X_new = pd.DataFrame()
    
    # Variance as a simple feature importance metric
    variances = X.var().sort_values(ascending=False)
    top_features = variances.index[:top_n]
    
    # Create interactions between top features
    for i, feat1 in enumerate(top_features):
        for feat2 in top_features[i+1:]:
            X_new[f'{feat1}_x_{feat2}'] = X[feat1] * X[feat2]
    
    return X_new

def create_data_splits(X, y, test_size, val_size, random_state):
    """
    Create train/validation/test splits.
    
    Args:
        X: Features
        y: Target
        test_size: Proportion of data for testing
        val_size: Proportion of training data to use for validation
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
    Scale features using StandardScaler and clip extreme values.
    
    Args:
        X_train, X_val, X_test: Data splits
    
    Returns:
        Scaled and clipped data
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Clip extreme values to ensure stable distances for KNN
    clip_bounds = (-5, 5)
    X_train_scaled = np.clip(X_train_scaled, clip_bounds[0], clip_bounds[1])
    X_val_scaled = np.clip(X_val_scaled, clip_bounds[0], clip_bounds[1])
    X_test_scaled = np.clip(X_test_scaled, clip_bounds[0], clip_bounds[1])
    
    return X_train_scaled, X_val_scaled, X_test_scaled, scaler

def apply_feature_selection(X_train_scaled, X_val_scaled, X_test_scaled, y_train, 
                           method, n_components=None, n_features=None, random_state=42):
    """
    Apply feature selection based on the specified method.
    
    Args:
        X_train_scaled, X_val_scaled, X_test_scaled: Scaled data
        y_train: Training labels
        method: Feature selection method ('pca', 'selectkbest', 'model_based', 'none')
        n_components: Number of PCA components (if method is 'pca')
        n_features: Number of features to select (if method is 'selectkbest')
        random_state: Random seed
    
    Returns:
        Reduced/selected data and the feature selection object
    """
    if method == 'none':
        return X_train_scaled, X_val_scaled, X_test_scaled, None
        
    if method == 'pca':
        n_components = min(n_components, X_train_scaled.shape[1], X_train_scaled.shape[0])
        pca = PCA(n_components=n_components, random_state=random_state)
        X_train_reduced = pca.fit_transform(X_train_scaled)
        X_val_reduced = pca.transform(X_val_scaled)
        X_test_reduced = pca.transform(X_test_scaled)
        
        explained_variance = np.sum(pca.explained_variance_ratio_)
        print(f"PCA with {n_components} components explains {explained_variance:.2%} of variance")
        
        return X_train_reduced, X_val_reduced, X_test_reduced, pca
        
    elif method == 'selectkbest':
        n_features = min(n_features, X_train_scaled.shape[1])
        selector = SelectKBest(f_classif, k=n_features)
        X_train_reduced = selector.fit_transform(X_train_scaled, y_train)
        X_val_reduced = selector.transform(X_val_scaled)
        X_test_reduced = selector.transform(X_test_scaled)
        
        print(f"Selected {n_features} best features using ANOVA F-value")
        
        return X_train_reduced, X_val_reduced, X_test_reduced, selector
        
    elif method == 'model_based':
        selector = SelectFromModel(
            RandomForestClassifier(n_estimators=200, random_state=random_state),
            threshold='median'
        )
        X_train_reduced = selector.fit_transform(X_train_scaled, y_train)
        X_val_reduced = selector.transform(X_val_scaled)
        X_test_reduced = selector.transform(X_test_scaled)
        
        print(f"Selected {X_train_reduced.shape[1]} features using model-based selection")
        
        return X_train_reduced, X_val_reduced, X_test_reduced, selector
    
    else:
        raise ValueError(f"Unknown feature selection method: {method}")

def train_supervised_models(X_train, y_train, X_val, y_val, random_state, use_ensemble=False):
    """
    Train supervised models with GridSearchCV.
    
    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        random_state: Random seed
        use_ensemble: Whether to use ensemble modeling
    
    Returns:
        Dictionary of trained models
    """
    models = {}
    
    # Custom weights function for KNN to handle class imbalance
    def class_weighted_distance(X, y):
        def _weight_func(distances):
            weights = np.ones_like(distances)
            # Compute class frequencies
            class_counts = np.bincount(y, minlength=len(np.unique(y)))
            class_weights = 1.0 / (class_counts + 1e-10)  # Inverse frequency
            for i, dist in enumerate(distances):
                # Get neighbor indices
                neighbor_indices = np.argsort(dist)[:min(len(dist), 31)]  # Use up to 31 neighbors
                neighbor_classes = y[neighbor_indices]
                # Assign weights based on class frequency
                for j, idx in enumerate(neighbor_indices):
                    weights[i, j] *= class_weights[neighbor_classes[j]]
            return weights
        return _weight_func
    
    model_params = {
        'logistic': {
            'model': LogisticRegression(max_iter=10000, random_state=random_state),
            'params': {
                'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],
                'solver': ['liblinear', 'lbfgs', 'saga'],
                'class_weight': [None, 'balanced']
            }
        },
        'random_forest': {
            'model': RandomForestClassifier(random_state=random_state),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [None, 10, 20, 30, 50],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': [None, 'balanced', 'balanced_subsample']
            }
        },
        'knn': {
            'model': KNeighborsClassifier(),
            'params': {
                'n_neighbors': [3, 5, 7, 11, 15, 21, 25, 31],
                'weights': ['uniform', 'distance', class_weighted_distance(X_train, y_train)],
                'metric': ['euclidean', 'manhattan', 'cosine'],
                'p': [1, 2]  # Only used for minkowski, included for completeness
            }
        },
        'gradient_boosting': {
            'model': GradientBoostingClassifier(random_state=random_state),
            'params': {
                'n_estimators': [100, 200, 300, 500],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.8, 0.9, 1.0]
            }
        },
        'svm': {
            'model': SVC(random_state=random_state, probability=True),
            'params': {
                'C': [0.1, 1, 10, 100],
                'kernel': ['linear', 'rbf', 'poly'],
                'gamma': ['scale', 'auto', 0.1, 1],
                'class_weight': [None, 'balanced']
            }
        }
    }
    
    for name, model_info in model_params.items():
        print(f"\nTraining {name}...")
        model = model_info['model']
        params = model_info['params']
        
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=random_state)
        
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
    
    if use_ensemble:
        print("\nTraining ensemble model...")
        # Use the top performing models for the ensemble
        sorted_models = sorted(models.items(), key=lambda x: x[1]['val_accuracy'], reverse=True)
        top_models = sorted_models[:3]  # Use top 3 models
        
        estimators = [(name, model_dict['model']) for name, model_dict in top_models]
        
        # Create and train ensemble model (soft voting)
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        ensemble.fit(X_train, y_train)
        
        val_accuracy = ensemble.score(X_val, y_val)
        print(f"Ensemble validation accuracy: {val_accuracy:.4f}")
        
        models['ensemble'] = {
            'model': ensemble,
            'params': {'estimators': [name for name, _ in estimators]},
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

def evaluate_models_on_test_set(models, kmeans, X_train, y_train, X_test, y_test, movement_names):
    """
    Evaluate all models on the test set.
    
    Args:
        models: Dictionary of supervised models
        kmeans: KMeans clustering model
        X_train: Training data
        y_train: Training labels
        X_test: Test data
        y_test: Test labels
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
    
    # Map clusters to labels using training data
    cluster_to_label = {}
    default_label = np.bincount(y_train).argmax()  # Fallback label (most common in training)
    for cluster in range(kmeans.n_clusters):
        train_indices = np.where(kmeans.labels_ == cluster)[0]
        if len(train_indices) > 0:
            cluster_to_label[cluster] = np.bincount(y_train[train_indices]).argmax()
        else:
            cluster_to_label[cluster] = default_label  # Assign default label to empty clusters
    
    # Evaluate K-means on test set
    kmeans_preds = np.array([cluster_to_label.get(label, default_label) for label in kmeans.predict(X_test)])
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

def create_visualizations(X_train, y_train, kmeans, X_test, y_test, movement_names, 
                         results, output_dir, random_state, models, feature_names, pca_obj=None):
    """
    Create visualizations of the results.
    
    Args:
        X_train, y_train: Training data
        kmeans: KMeans clustering model
        X_test, y_test: Test data
        movement_names: Names of literary movements
        results: Dictionary of test results
        output_dir: Directory to save plots
        random_state: Random seed
        models: Dictionary of trained models
        feature_names: Names of features
        pca_obj: PCA object if PCA was used for dimensionality reduction
    """
    viz_dir = os.path.join(output_dir, 'visualizations')
    os.makedirs(viz_dir, exist_ok=True)
    
    print("\nGenerating t-SNE visualization...")
    # If we have high-dimensional data, run PCA first to speed up t-SNE
    if X_train.shape[1] > 50 and pca_obj is None:
        pca_tsne = PCA(n_components=50, random_state=random_state)
        X_train_pca = pca_tsne.fit_transform(X_train)
        X_test_pca = pca_tsne.transform(X_test)
    else:
        X_train_pca = X_train
        X_test_pca = X_test
    
    tsne = TSNE(n_components=2, random_state=random_state, perplexity=min(30, X_train.shape[0]//5), 
               learning_rate='auto', init='pca')
    X_tsne = tsne.fit_transform(np.vstack([X_train_pca, X_test_pca]))
    X_train_tsne = X_tsne[:len(X_train_pca)]
    X_test_tsne = X_tsne[len(X_train_pca):]
    
    plt.figure(figsize=(12, 10))
    palette = sns.color_palette("husl", len(movement_names))
    for i, (movement, color) in enumerate(zip(movement_names, palette)):
        plt.scatter(X_train_tsne[y_train == i, 0], X_train_tsne[y_train == i, 1], 
                   label=movement, color=color, alpha=0.7)
    plt.title('t-SNE Visualization of Literary Movements')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tsne_movements.png'), dpi=300)
    
    plt.figure(figsize=(12, 10))
    cluster_palette = sns.color_palette("bright", kmeans.n_clusters)
    for i, color in enumerate(cluster_palette):
        plt.scatter(X_train_tsne[kmeans.labels_ == i, 0], X_train_tsne[kmeans.labels_ == i, 1], 
                   label=f'Cluster {i}', color=color, alpha=0.7)
    plt.title('t-SNE Visualization of K-means Clusters')
    plt.xlabel('t-SNE Component 1')
    plt.ylabel('t-SNE Component 2')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'tsne_kmeans_clusters.png'), dpi=300)
    
    best_model_name = max(results, key=lambda k: results[k]['accuracy'] if k != 'kmeans' else 0)
    best_model_preds = results[best_model_name]['predictions']
    
    plt.figure(figsize=(12, 10))
    cm = confusion_matrix(y_test, best_model_preds, normalize='true')
    sns.heatmap(cm, annot=True, fmt='.2f', cmap='Blues', xticklabels=movement_names, yticklabels=movement_names)
    plt.title(f'Confusion Matrix (Normalized) - {best_model_name.replace("_", " ").title()}')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, f'confusion_matrix_{best_model_name}.png'), dpi=300)
    
    # Display feature importances for tree-based models
    for model_name in ['random_forest', 'gradient_boosting']:
        if model_name in models:
            model = models[model_name]['model']
            
            if hasattr(model, 'feature_importances_'):
                feature_importances = model.feature_importances_
                
                # If we used feature selection, we might not have direct correspondence with original features
                if pca_obj is not None:
                    plt.figure(figsize=(12, 8))
                    n_components = len(feature_importances)
                    plt.barh(range(n_components), feature_importances, align='center')
                    plt.yticks(range(n_components), [f'Component {i+1}' for i in range(n_components)])
                    plt.xlabel('Feature Importance')
                    plt.title(f'Component Importance ({model_name.replace("_", " ").title()})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, f'feature_importance_{model_name}.png'), dpi=300)
                else:
                    # Show top 20 features or all if less than 20
                    n_to_show = min(20, len(feature_importances))
                    top_indices = np.argsort(feature_importances)[-n_to_show:]
                    plt.figure(figsize=(12, 8))
                    plt.barh(range(len(top_indices)), feature_importances[top_indices], align='center')
                    plt.yticks(range(len(top_indices)), 
                              [feature_names[i] if i < len(feature_names) else f'Feature {i}' for i in top_indices])
                    plt.xlabel('Feature Importance')
                    plt.title(f'Top {n_to_show} Most Important Features ({model_name.replace("_", " ").title()})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(viz_dir, f'feature_importance_{model_name}.png'), dpi=300)
    
    # Model accuracy comparison
    accuracies = {name: result['accuracy'] for name, result in results.items()}
    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(len(accuracies)), list(accuracies.values()), tick_label=list(accuracies.keys()))
    plt.title('Model Accuracy Comparison')
    plt.xlabel('Model')
    plt.ylabel('Accuracy')
    plt.xticks(rotation=45)
    
    # Add numeric labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2%}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'model_accuracy_comparison.png'), dpi=300)
    
    # Per-class performance for the best model
    best_report = results[best_model_name]['report']
    class_metrics = {label: metrics['f1-score'] for label, metrics in best_report.items() 
                    if label not in ['accuracy', 'macro avg', 'weighted avg']}
    
    plt.figure(figsize=(12, 6))
    movement_indices = range(len(class_metrics))
    bars = plt.bar(movement_indices, list(class_metrics.values()))
    plt.title(f'F1-Score Per Literary Movement ({best_model_name.replace("_", " ").title()})')
    plt.xlabel('Literary Movement')
    plt.ylabel('F1-Score')
    plt.xticks(movement_indices, movement_names, rotation=45)
    
    # Add numeric labels above bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{height:.2f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'per_class_performance.png'), dpi=300)

def save_results(models, kmeans, results, output_dir, scaler, feature_selector, feature_names, movement_names):
    """
    Save trained models and results.
    
    Args:
        models: Dictionary of supervised models
        kmeans: KMeans clustering model
        results: Dictionary of test results
        output_dir: Directory to save results
        scaler: Feature scaler
        feature_selector: Feature selection object
        feature_names: Names of features
        movement_names: Names of literary movements
    """
    os.makedirs(output_dir, exist_ok=True)
    models_dir = os.path.join(output_dir, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Save feature names and movement names
    np.save(os.path.join(output_dir, 'feature_names.npy'), feature_names)
    np.save(os.path.join(output_dir, 'movement_names.npy'), movement_names)
    
    # Save scaler and feature selector (if used)
    joblib.dump(scaler, os.path.join(models_dir, 'scaler.joblib'))
    if feature_selector is not None:
        joblib.dump(feature_selector, os.path.join(models_dir, 'feature_selector.joblib'))
    
    # Save each model
    for name, model_dict in models.items():
        model = model_dict['model']
        joblib.dump(model, os.path.join(models_dir, f'{name}_model.joblib'))
    
    # Save KMeans model and cluster mappings
    joblib.dump(kmeans, os.path.join(models_dir, 'kmeans_model.joblib'))
    cluster_mapping = results['kmeans']['cluster_mapping']
    np.save(os.path.join(models_dir, 'cluster_mapping.npy'), cluster_mapping)
    
    # Save results summary as a text file
    with open(os.path.join(output_dir, 'results_summary.txt'), 'w') as f:
        f.write("=== Model Accuracy Results ===\n")
        for name, accuracy in results.items():
            f.write(f"{name}: {accuracy['accuracy']:.4f}\n")
        f.write(f"\nBest model: {max(results, key=lambda x: results[x]['accuracy'] if x != 'kmeans' else 0)}\n")
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Feature count: {len(feature_names)}\n")
        f.write(f"Literary movements: {', '.join(movement_names)}\n")

def main():
    """Main function to run the script."""
    args = parse_arguments()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load and preprocess data
    X, y, feature_names, movement_names, df = load_and_preprocess_data(args.data, args.random_state)
    
    # Create data splits
    X_train, X_val, X_test, y_train, y_val, y_test = create_data_splits(
        X, y, args.test_size, args.val_size, args.random_state
    )
    
    # Scale features
    X_train_scaled, X_val_scaled, X_test_scaled, scaler = scale_features(X_train, X_val, X_test)
    
    # Apply feature selection
    X_train_reduced, X_val_reduced, X_test_reduced, feature_selector = apply_feature_selection(
        X_train_scaled, X_val_scaled, X_test_scaled, y_train,
        method=args.feature_selection,
        n_components=args.pca_components,
        n_features=args.n_features,
        random_state=args.random_state
    )
    
    # Train supervised models
    models = train_supervised_models(
        X_train_reduced, y_train, X_val_reduced, y_val,
        random_state=args.random_state,
        use_ensemble=args.use_ensemble
    )
    
    # Perform unsupervised clustering
    kmeans = perform_unsupervised_clustering(
        X_train_reduced, y_train, X_val_reduced, y_val,
        n_clusters=args.k_clusters,
        random_state=args.random_state
    )
    
    # Evaluate models on test set
    results = evaluate_models_on_test_set(
        models, kmeans, X_train_reduced, y_train, X_test_reduced, y_test, movement_names
    )
    
    # Create visualizations if requested
    if args.visualize:
        create_visualizations(
            X_train_reduced, y_train, kmeans, X_test_reduced, y_test, movement_names,
            results, args.output_dir, args.random_state, models, feature_names,
            pca_obj=feature_selector if args.feature_selection == 'pca' else None
        )
    
    # Save results and models
    save_results(
        models, kmeans, results, args.output_dir, scaler, feature_selector,
        feature_names, movement_names
    )
    
    print(f"\nPipeline completed successfully. Results saved to {args.output_dir}")

if __name__ == "__main__":
    main()
