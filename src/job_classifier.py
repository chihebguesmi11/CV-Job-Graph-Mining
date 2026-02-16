"""
Job Classifier Module

Implements job classification for domain and level prediction.
Owner: Chiheb
Dependencies: numpy, pandas, scikit-learn, networkx
"""

import json
import numpy as np
import pandas as pd
import pickle
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
import matplotlib.pyplot as plt
import seaborn as sns


class JobFeatureEngineer:
    """
    Creates feature vectors for job nodes.
    
    Features include:
    - Structural: degree, betweenness, PageRank, clustering
    - Community: community_id, community_size
    - Semantic: job embeddings
    - Neighborhood: average neighbor features
    """
    
    def __init__(self, graph: nx.Graph, 
                 communities: Dict[str, int],
                 job_nodes: List[str],
                 job_embeddings: Optional[Dict[str, np.ndarray]] = None):
        """
        Initialize feature engineer.
        
        Args:
            graph (nx.Graph): Bipartite graph
            communities (Dict[str, int]): Community assignments
            job_nodes (List[str]): List of job node IDs
            job_embeddings (Dict[str, np.ndarray]): Job embeddings (optional)
        """
        self.graph = graph
        self.communities = communities
        self.job_nodes = job_nodes
        self.job_embeddings = job_embeddings or {}
        
        # Precompute centrality measures
        print("Computing centrality measures...")
        self.betweenness = nx.betweenness_centrality(graph)
        self.pagerank = nx.pagerank(graph)
    
    def create_features(self, job_id: str, 
                       jobs_data: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Create feature vector for a job node.
        
        Args:
            job_id (str): Job ID
            jobs_data (Dict[str, Any]): Job data (for semantic features)
        
        Returns:
            np.ndarray: Feature vector
        """
        features = {}
        
        # Ensure job is in graph (add as isolated node if necessary)
        if job_id not in self.graph:
            self.graph.add_node(job_id)
        
        # Structural features
        features['degree'] = self.graph.degree(job_id)
        features['betweenness'] = self.betweenness.get(job_id, 0)
        features['pagerank'] = self.pagerank.get(job_id, 0)
        features['clustering'] = nx.clustering(self.graph, job_id)
        
        # Community features
        features['community_id'] = self.communities.get(job_id, -1)
        comm_nodes = [n for n in self.graph.nodes() 
                     if self.communities.get(n) == self.communities.get(job_id)]
        features['community_size'] = len(comm_nodes)
        
        # Neighborhood features
        neighbors = list(self.graph.neighbors(job_id))
        if neighbors:
            neighbor_degrees = [self.graph.degree(n) for n in neighbors]
            features['avg_neighbor_degree'] = np.mean(neighbor_degrees)
            features['max_neighbor_degree'] = max(neighbor_degrees)
            features['min_neighbor_degree'] = min(neighbor_degrees)
        else:
            features['avg_neighbor_degree'] = 0
            features['max_neighbor_degree'] = 0
            features['min_neighbor_degree'] = 0
        
        # Create feature array
        structural_feats = np.array([
            features['degree'],
            features['betweenness'],
            features['pagerank'],
            features['clustering'],
            features['community_size'],
            features['avg_neighbor_degree'],
            features['max_neighbor_degree'],
            features['min_neighbor_degree'],
        ])
        
        # Add embeddings if available
        if job_id in self.job_embeddings:
            embedding = self.job_embeddings[job_id]
        else:
            embedding = np.zeros(384)  # Default embedding dimension
        
        # Concatenate all features
        full_features = np.concatenate([structural_feats, embedding])
        
        return full_features
    
    def create_feature_matrix(self, jobs_data: Optional[List[Dict[str, Any]]] = None) -> Tuple[np.ndarray, List[str]]:
        """
        Create feature matrix for all jobs.
        
        Args:
            jobs_data (List[Dict[str, Any]]): Job data (optional)
        
        Returns:
            Tuple: (feature_matrix, feature_names)
        """
        feature_matrix = []
        
        for job_id in self.job_nodes:
            features = self.create_features(job_id, jobs_data)
            feature_matrix.append(features)
        
        feature_matrix = np.array(feature_matrix)
        
        # Feature names
        structural_names = [
            'degree', 'betweenness', 'pagerank', 'clustering',
            'community_size', 'avg_neighbor_degree', 'max_neighbor_degree', 'min_neighbor_degree'
        ]
        embedding_names = [f'embedding_{i}' for i in range(384)]
        feature_names = structural_names + embedding_names
        
        return feature_matrix, feature_names


class JobDomainClassifier:
    """
    Classifies jobs by domain (engineering, data_science, management).
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            random_state (int): Random seed
        """
        self.random_state = random_state
        self.model = None
        self.label_encoder = None
        self.feature_names = None
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              feature_names: Optional[List[str]] = None,
              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train domain classifier.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels (domain names)
            feature_names (List[str]): Names of features
            test_size (float): Test set proportion
        
        Returns:
            Dict[str, Any]: Training metrics and results
        """
        self.feature_names = feature_names
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state,
            stratify=y_encoded
        )
        
        print(f"\nTraining Job Domain Classifier")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        metrics = {
            'accuracy': np.mean(y_pred == y_test),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'classification_report': classification_report(y_test, y_pred, 
                                                          target_names=self.label_encoder.classes_),
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
        
        print(f"\nDomain Classification Results:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision (weighted): {metrics['precision']:.3f}")
        print(f"  Recall (weighted): {metrics['recall']:.3f}")
        print(f"  F1 (weighted): {metrics['f1']:.3f}")
        
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict job domain.
        
        Args:
            X (np.ndarray): Feature matrix
        
        Returns:
            Tuple: (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        y_pred_encoded = self.model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_proba = self.model.predict_proba(X)
        
        return y_pred, y_proba
    
    def save(self, filepath: str) -> None:
        """Save model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
            }, f)
        
        print(f"Saved domain classifier to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.feature_names = data['feature_names']
        
        print(f"Loaded domain classifier from {filepath}")


class JobLevelClassifier:
    """
    Classifies jobs by level (entry, mid, senior, lead).
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize classifier.
        
        Args:
            random_state (int): Random seed
        """
        self.random_state = random_state
        self.model = None
        self.label_encoder = None
        self.feature_names = None
    
    def train(self, X: np.ndarray, y: np.ndarray,
              feature_names: Optional[List[str]] = None,
              test_size: float = 0.2) -> Dict[str, Any]:
        """
        Train level classifier.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Target labels (level names)
            feature_names (List[str]): Names of features
            test_size (float): Test set proportion
        
        Returns:
            Dict[str, Any]: Training metrics and results
        """
        self.feature_names = feature_names
        
        # Encode labels
        self.label_encoder = LabelEncoder()
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state,
            stratify=y_encoded
        )
        
        print(f"\nTraining Job Level Classifier")
        print(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
        print(f"Classes: {self.label_encoder.classes_}")
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        
        # Get unique labels from test set to avoid mismatches
        unique_labels = np.unique(y_test)
        target_names = self.label_encoder.classes_[unique_labels]
        
        metrics = {
            'accuracy': np.mean(y_pred == y_test),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'classification_report': classification_report(y_test, y_pred,
                                                          labels=unique_labels,
                                                          target_names=target_names),
            'confusion_matrix': confusion_matrix(y_test, y_pred, labels=unique_labels)
        }
        
        print(f"\nLevel Classification Results:")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        print(f"  Precision (weighted): {metrics['precision']:.3f}")
        print(f"  Recall (weighted): {metrics['recall']:.3f}")
        print(f"  F1 (weighted): {metrics['f1']:.3f}")
        
        print("\nClassification Report:")
        print(metrics['classification_report'])
        
        return metrics
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict job level.
        
        Args:
            X (np.ndarray): Feature matrix
        
        Returns:
            Tuple: (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained")
        
        y_pred_encoded = self.model.predict(X)
        y_pred = self.label_encoder.inverse_transform(y_pred_encoded)
        y_proba = self.model.predict_proba(X)
        
        return y_pred, y_proba
    
    def save(self, filepath: str) -> None:
        """Save model."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'label_encoder': self.label_encoder,
                'feature_names': self.feature_names,
            }, f)
        
        print(f"Saved level classifier to {filepath}")
    
    def load(self, filepath: str) -> None:
        """Load model."""
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.label_encoder = data['label_encoder']
        self.feature_names = data['feature_names']
        
        print(f"Loaded level classifier from {filepath}")


class JobClassificationVisualizer:
    """
    Creates visualizations for job classification results.
    """
    
    @staticmethod
    def plot_confusion_matrix(cm: np.ndarray, labels: List[str],
                             title: str, filepath: str) -> None:
        """
        Plot confusion matrix.
        
        Args:
            cm (np.ndarray): Confusion matrix
            labels (List[str]): Class labels
            title (str): Plot title
            filepath (str): Output file path
        """
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=labels, yticklabels=labels)
        plt.title(title)
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Saved confusion matrix to {filepath}")
    
    @staticmethod
    def plot_feature_importance(importances: np.ndarray,
                               feature_names: List[str],
                               n_top: int = 15,
                               title: str = "Feature Importance",
                               filepath: str = None) -> None:
        """
        Plot feature importance.
        
        Args:
            importances (np.ndarray): Feature importances
            feature_names (List[str]): Feature names
            n_top (int): Number of top features to plot
            title (str): Plot title
            filepath (str): Output file path
        """
        indices = np.argsort(importances)[-n_top:]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(indices)), importances[indices])
        plt.yticks(range(len(indices)), 
                  [feature_names[i] if i < len(feature_names) else f'feat_{i}'
                   for i in indices])
        plt.xlabel('Feature Importance')
        plt.title(title)
        plt.tight_layout()
        
        if filepath:
            Path(filepath).parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(filepath, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Saved feature importance plot to {filepath}")


if __name__ == '__main__':
    print("Job Classifier Module Loaded")
