"""
Link Prediction Module

Implements hybrid link prediction using graph-based features and LLM embeddings.
Owner: Chiheb
Dependencies: networkx, numpy, pandas, scikit-learn, llm_utils
"""

import json
import numpy as np
import pandas as pd
import pickle
from typing import List, Dict, Tuple, Set, Optional
from pathlib import Path
import warnings

import networkx as nx
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    precision_score, recall_score, roc_auc_score, 
    classification_report, confusion_matrix, roc_curve, auc
)
import matplotlib.pyplot as plt
import seaborn as sns

from llm_utils import SemanticSimilarityMatrix


class GraphLinkFeatureComputer:
    """
    Computes graph-based link prediction features.
    
    Features include:
    - Common Neighbors (CN)
    - Adamic-Adar (AA)
    - Jaccard Coefficient (JC)
    - Preferential Attachment (PA)
    - Community features
    - Degree features
    """
    
    def __init__(self, graph: nx.Graph, communities: Dict[str, int],
                 cv_nodes: Set[str], job_nodes: Set[str]):
        """
        Initialize feature computer.
        
        Args:
            graph (nx.Graph): Bipartite graph
            communities (Dict[str, int]): Community assignments
            cv_nodes (Set[str]): CV node IDs
            job_nodes (Set[str]): Job node IDs
        """
        self.graph = graph
        self.communities = communities
        self.cv_nodes = cv_nodes
        self.job_nodes = job_nodes
    
    def common_neighbors(self, u: str, v: str) -> int:
        """
        Count common neighbors between two nodes.
        
        For bipartite graphs, this counts paths of length 2.
        
        Args:
            u (str): First node
            v (str): Second node
        
        Returns:
            int: Number of common neighbors
        """
        if u not in self.graph or v not in self.graph:
            return 0
        
        neighbors_u = set(self.graph.neighbors(u))
        neighbors_v = set(self.graph.neighbors(v))
        
        return len(neighbors_u & neighbors_v)
    
    def adamic_adar(self, u: str, v: str) -> float:
        """
        Compute Adamic-Adar similarity.
        
        score_aa(u,v) = Σ 1/log(deg(w)) for w ∈ common_neighbors(u,v)
        
        Args:
            u (str): First node
            v (str): Second node
        
        Returns:
            float: Adamic-Adar score
        """
        if u not in self.graph or v not in self.graph:
            return 0.0
        
        neighbors_u = set(self.graph.neighbors(u))
        neighbors_v = set(self.graph.neighbors(v))
        common = neighbors_u & neighbors_v
        
        aa_score = 0.0
        for node in common:
            deg = self.graph.degree(node)
            if deg > 1:  # Avoid log(1) = 0
                aa_score += 1.0 / np.log(deg)
        
        return aa_score
    
    def jaccard_coefficient(self, u: str, v: str) -> float:
        """
        Compute Jaccard coefficient.
        
        score_jc(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|
        
        Args:
            u (str): First node
            v (str): Second node
        
        Returns:
            float: Jaccard coefficient in [0, 1]
        """
        if u not in self.graph or v not in self.graph:
            return 0.0
        
        neighbors_u = set(self.graph.neighbors(u))
        neighbors_v = set(self.graph.neighbors(v))
        
        intersection = len(neighbors_u & neighbors_v)
        union = len(neighbors_u | neighbors_v)
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def preferential_attachment(self, u: str, v: str) -> float:
        """
        Compute preferential attachment score.
        
        score_pa(u,v) = deg(u) × deg(v)
        
        Args:
            u (str): First node
            v (str): Second node
        
        Returns:
            float: Preferential attachment score
        """
        deg_u = self.graph.degree(u) if u in self.graph else 0
        deg_v = self.graph.degree(v) if v in self.graph else 0
        
        return deg_u * deg_v
    
    def compute_features(self, u: str, v: str) -> Dict[str, float]:
        """
        Compute all link prediction features for a pair.
        
        Args:
            u (str): First node (CV)
            v (str): Second node (Job)
        
        Returns:
            Dict[str, float]: Dictionary of features
        """
        features = {
            'common_neighbors': self.common_neighbors(u, v),
            'adamic_adar': self.adamic_adar(u, v),
            'jaccard': self.jaccard_coefficient(u, v),
            'preferential_attachment': self.preferential_attachment(u, v),
            'same_community': 1 if self.communities.get(u) == self.communities.get(v) else 0,
            'cv_degree': self.graph.degree(u) if u in self.graph else 0,
            'job_degree': self.graph.degree(v) if v in self.graph else 0,
        }
        
        return features
    
    def compute_batch_features(self, pairs: List[Tuple[str, str]], 
                              verbose: bool = False) -> List[Dict[str, float]]:
        """
        Compute features for multiple pairs.
        
        Args:
            pairs (List[Tuple[str, str]]): List of (u, v) pairs
            verbose (bool): Whether to print progress
        
        Returns:
            List[Dict[str, float]]: List of feature dictionaries
        """
        all_features = []
        
        for i, (u, v) in enumerate(pairs):
            if verbose and (i + 1) % 100 == 0:
                print(f"Computed features for {i + 1}/{len(pairs)} pairs")
            
            features = self.compute_features(u, v)
            all_features.append(features)
        
        return all_features


class HybridLinkPredictionModel:
    """
    Hybrid link prediction model combining graph and semantic features.
    
    Uses Random Forest classifier trained on:
    - Graph-based features (CN, AA, JC, PA, community, degree)
    - Semantic similarity from embeddings
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize model.
        
        Args:
            random_state (int): Random seed
        """
        self.random_state = random_state
        self.model = None
        self.feature_names = None
        self.scaler = None
        self.threshold = 0.5
    
    def prepare_training_data(self, 
                             feature_computer: GraphLinkFeatureComputer,
                             cv_nodes: Set[str],
                             job_nodes: List[str],
                             graph: nx.Graph,
                             semantic_similarity_matrix: Optional[np.ndarray] = None,
                             cv_ids_sim: Optional[List[str]] = None,
                             job_ids_sim: Optional[List[str]] = None,
                             n_negative: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """
        Prepare training data with positive and negative samples.
        
        Args:
            feature_computer (GraphLinkFeatureComputer): Feature computer
            cv_nodes (Set[str]): Set of CV node IDs
            job_nodes (List[str]): List of job node IDs
            graph (nx.Graph): Graph object
            semantic_similarity_matrix (np.ndarray): Semantic similarity matrix (optional)
            cv_ids_sim (List[str]): CV IDs for semantic similarity indexing
            job_ids_sim (List[str]): Job IDs for semantic similarity indexing
            n_negative (int): Number of negative samples (defaults to #positive)
        
        Returns:
            Tuple: (X, y, feature_names)
                - X: Feature matrix
                - y: Labels (1=link, 0=no link)
                - feature_names: Names of features
        """
        import random
        
        # Positive samples: existing edges
        positive_samples = []
        positive_features = []
        
        for cv, job in graph.edges():
            if cv in cv_nodes and job in job_nodes:
                features_dict = feature_computer.compute_features(cv, job)
                
                # Add semantic similarity if available
                if semantic_similarity_matrix is not None:
                    cv_idx = cv_ids_sim.index(cv) if cv in cv_ids_sim else -1
                    job_idx = job_ids_sim.index(job) if job in job_ids_sim else -1
                    
                    if cv_idx >= 0 and job_idx >= 0:
                        sem_sim = semantic_similarity_matrix[cv_idx, job_idx]
                    else:
                        sem_sim = 0.5
                    
                    features_dict['semantic_similarity'] = sem_sim
                else:
                    features_dict['semantic_similarity'] = 0.5
                
                positive_samples.append((cv, job))
                positive_features.append(features_dict)
        
        print(f"Generated {len(positive_samples)} positive samples")
        
        # Negative samples: non-edges (equal number)
        if n_negative is None:
            n_negative = len(positive_samples)
        
        negative_samples = []
        negative_features = []
        
        # Limit attempts to avoid infinite loops
        max_attempts = n_negative * 100
        attempts = 0
        
        while len(negative_samples) < n_negative and attempts < max_attempts:
            cv = random.choice(list(cv_nodes))
            job = random.choice(job_nodes)
            attempts += 1
            
            if not graph.has_edge(cv, job) and (cv, job) not in negative_samples:
                features_dict = feature_computer.compute_features(cv, job)
                
                # Add semantic similarity if available
                if semantic_similarity_matrix is not None:
                    cv_idx = cv_ids_sim.index(cv) if cv in cv_ids_sim else -1
                    job_idx = job_ids_sim.index(job) if job in job_ids_sim else -1
                    
                    if cv_idx >= 0 and job_idx >= 0:
                        sem_sim = semantic_similarity_matrix[cv_idx, job_idx]
                    else:
                        sem_sim = 0.5
                    
                    features_dict['semantic_similarity'] = sem_sim
                else:
                    features_dict['semantic_similarity'] = 0.5
                
                negative_samples.append((cv, job))
                negative_features.append(features_dict)
        
        # If we couldn't find enough negative samples, reduce the requirement
        if len(negative_samples) < n_negative:
            print(f"⚠ Could only generate {len(negative_samples)} negative samples (requested {n_negative})")
            print(f"  Graph is too dense - ratio: {graph.number_of_edges()} / {len(cv_nodes) * len(job_nodes)} edges")
        
        print(f"Generated {len(negative_samples)} negative samples")
        
        # Combine samples
        all_features = positive_features + negative_features
        all_labels = [1] * len(positive_samples) + [0] * len(negative_samples)
        
        # Convert to numpy arrays
        feature_names = list(all_features[0].keys())
        self.feature_names = feature_names
        
        X = np.array([[f[name] for name in feature_names] for f in all_features])
        y = np.array(all_labels)
        
        return X, y, feature_names
    
    def train(self, X: np.ndarray, y: np.ndarray, 
              feature_names: Optional[List[str]] = None,
              test_size: float = 0.2) -> Dict[str, float]:
        """
        Train link prediction model.
        
        Args:
            X (np.ndarray): Feature matrix
            y (np.ndarray): Labels
            feature_names (Optional[List[str]]): Feature names for importance analysis
            test_size (float): Test set proportion
        
        Returns:
            Dict[str, float]: Performance metrics
        """
        # Store feature names if provided
        if feature_names is not None:
            self.feature_names = feature_names
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state, stratify=y
        )
        
        print(f"Training on {len(X_train)} samples, testing on {len(X_test)} samples")
        
        # Train model
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=self.random_state,
            n_jobs=-1
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        y_pred_proba = self.model.predict_proba(X_test)[:, 1]
        
        metrics = {
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'auc_roc': roc_auc_score(y_test, y_pred_proba),
            'accuracy': np.mean(y_pred == y_test),
            'X_test': X_test,
            'y_test': y_test,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        
        print(f"\nLink Prediction Performance:")
        print(f"  Precision: {metrics['precision']:.3f}")
        print(f"  Recall: {metrics['recall']:.3f}")
        print(f"  AUC-ROC: {metrics['auc_roc']:.3f}")
        print(f"  Accuracy: {metrics['accuracy']:.3f}")
        
        # Feature importance
        importances = self.model.feature_importances_
        for name, importance in sorted(zip(self.feature_names, importances), 
                                      key=lambda x: x[1], reverse=True)[:10]:
            print(f"    {name}: {importance:.3f}")
        
        return metrics
    
    def predict_links(self, X: np.ndarray, threshold: float = 0.75) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict links for new pairs.
        
        Args:
            X (np.ndarray): Feature matrix
            threshold (float): Probability threshold for positive prediction
        
        Returns:
            Tuple: (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        probilities = self.model.predict_proba(X)[:, 1]
        predictions = (probilities > threshold).astype(int)
        
        return predictions, probilities
    
    def save(self, filepath: str) -> None:
        """
        Save trained model.
        
        Args:
            filepath (str): Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'model': self.model,
                'feature_names': self.feature_names,
                'random_state': self.random_state
            }, f)
        
        print(f"Saved model to {filepath}")
    
    def load(self, filepath: str) -> None:
        """
        Load trained model.
        
        Args:
            filepath (str): Input file path
        """
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.model = data['model']
        self.feature_names = data['feature_names']
        self.random_state = data['random_state']
        
        print(f"Loaded model from {filepath}")


class LinkEnricher:
    """
    Enriches graph with predicted links.
    """
    
    @staticmethod
    def enrich_graph(graph: nx.Graph,
                    predicted_links: List[Tuple[str, str, float]],
                    threshold: float = 0.75) -> Tuple[nx.Graph, List[Tuple[str, str, float]]]:
        """
        Add predicted links to graph.
        
        Args:
            graph (nx.Graph): Original graph
            predicted_links (List[Tuple[str, str, float]]): (cv, job, probability) tuples
            threshold (float): Minimum probability to add link
        
        Returns:
            Tuple: (enriched_graph, added_links)
        """
        enriched_graph = graph.copy()
        added_links = []
        
        for cv, job, prob in predicted_links:
            if prob >= threshold and not enriched_graph.has_edge(cv, job):
                enriched_graph.add_edge(cv, job, weight=prob, predicted=True)
                added_links.append((cv, job, prob))
        
        print(f"Added {len(added_links)} predicted links to graph")
        
        return enriched_graph, added_links
    
    @staticmethod
    def save_enriched_graph(graph: nx.Graph, filepath: str) -> None:
        """
        Save enriched graph.
        
        Args:
            graph (nx.Graph): Graph to save
            filepath (str): Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f)
        print(f"Saved enriched graph to {filepath}")
    
    @staticmethod
    def save_predicted_links(links: List[Tuple[str, str, float]], filepath: str) -> None:
        """
        Save predicted links to CSV.
        
        Args:
            links (List[Tuple[str, str, float]]): List of predicted links
            filepath (str): Output file path
        """
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        df = pd.DataFrame(links, columns=['cv_id', 'job_id', 'probability'])
        df = df.sort_values('probability', ascending=False)
        df.to_csv(filepath, index=False)
        
        print(f"Saved {len(links)} predicted links to {filepath}")


if __name__ == '__main__':
    print("Link Prediction Module Loaded")
