"""
CV Node Classification Module

This module implements classification models for CVs:
1. Seniority classification (junior/mid/senior)
2. Specialization classification (specialist/polyvalent)
3. Semi-supervised iterative classification (REV2)

Owner: Ramy
Dependencies: cv_community.py, metrics_computer.py
"""

import json
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Any, Tuple


class CVClassifier:
    """
    Classifies CVs based on structural, semantic, and community features.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the CV classifier.
        
        Args:
            random_state (int): Random seed for reproducibility
        """
        self.random_state = random_state
        self.seniority_clf = None
        self.specialization_clf = None
        self.label_encoder = LabelEncoder()
    
    def build_feature_matrix(
        self,
        cvs: List[Dict[str, Any]],
        cv_metrics: pd.DataFrame,
        embeddings: np.ndarray = None
    ) -> Tuple[np.ndarray, List[str]]:
        """
        Build comprehensive feature matrix for CVs.
        
        Features include:
        - Structural: degree, betweenness, PageRank, clustering
        - Community: community_id, community_size
        - Internal: n_skills, experience_years, n_skill_communities, specialization_score
        - Semantic: embeddings (if provided by Chiheb)
        
        Args:
            cvs (List[Dict]): CV data
            cv_metrics (pd.DataFrame): Structural metrics
            embeddings (np.ndarray): Optional CV embeddings from Chiheb
        
        Returns:
            Tuple[np.ndarray, List[str]]: (feature matrix, feature names)
        """
        print(f"🔨 Building feature matrix for {len(cvs)} CVs...")
        
        features = []
        feature_names = []
        
        # Create CV ID to index mapping
        cv_dict = {cv['cv_id']: cv for cv in cvs}
        metrics_dict = cv_metrics.set_index('node').to_dict('index')
        
        for cv in cvs:
            cv_id = cv['cv_id']
            cv_features = []
            
            # Structural features
            if cv_id in metrics_dict:
                cv_features.extend([
                    metrics_dict[cv_id].get('degree', 0),
                    metrics_dict[cv_id].get('betweenness', 0),
                    metrics_dict[cv_id].get('pagerank', 0),
                    metrics_dict[cv_id].get('clustering', 0)
                ])
            else:
                cv_features.extend([0, 0, 0, 0])
            
            # Community features
            cv_features.extend([
                metrics_dict[cv_id].get('community_id', -1) if cv_id in metrics_dict else -1,
                metrics_dict[cv_id].get('community_size', 0) if cv_id in metrics_dict else 0
            ])
            
            # Internal features
            cv_features.extend([
                len(cv['skills']),
                cv['experience_years'],
                cv.get('n_skill_communities', 0),
                cv.get('specialization_score', 0),
                cv.get('skill_modularity', 0)
            ])
            
            features.append(cv_features)
        
        # Feature names
        feature_names = [
            'degree', 'betweenness', 'pagerank', 'clustering',
            'community_id', 'community_size',
            'n_skills', 'experience_years', 'n_skill_communities',
            'specialization_score', 'skill_modularity'
        ]
        
        # Add embeddings if provided
        if embeddings is not None:
            print(f"   Adding {embeddings.shape[1]} embedding features")
            features = np.hstack([np.array(features), embeddings])
            feature_names.extend([f'emb_{i}' for i in range(embeddings.shape[1])])
        
        feature_matrix = np.array(features)
        
        print(f"✅ Built feature matrix: {feature_matrix.shape}")
        print(f"   Features: {len(feature_names)}")
        
        return feature_matrix, feature_names
    
    def train_seniority_classifier(
        self,
        X: np.ndarray,
        cvs: List[Dict[str, Any]],
        test_size: float = 0.2
    ) -> Dict[str, Any]:
        """
        Train Random Forest classifier for seniority prediction.
        
        Args:
            X (np.ndarray): Feature matrix
            cvs (List[Dict]): CV data
            test_size (float): Test set proportion
        
        Returns:
            Dict: Training results and metrics
        """
        print(f"\n🎓 Training Seniority Classifier...")
        
        # Extract labels
        y = np.array([cv['seniority'] for cv in cvs])
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=test_size, random_state=self.random_state, stratify=y_encoded
        )
        
        # Train classifier
        self.seniority_clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=self.random_state
        )
        self.seniority_clf.fit(X_train, y_train)
        
        # Predict
        y_pred = self.seniority_clf.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Seniority Classification Accuracy: {accuracy:.3f}")
        
        # Classification report
        labels_str = self.label_encoder.inverse_transform(np.unique(y_encoded))
        y_test_str = self.label_encoder.inverse_transform(y_test)
        y_pred_str = self.label_encoder.inverse_transform(y_pred)
        
        report = classification_report(y_test_str, y_pred_str, output_dict=True)
        
        print("\n" + classification_report(y_test_str, y_pred_str))
        
        results = {
            'accuracy': accuracy,
            'classification_report': report,
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'test_predictions': y_pred_str.tolist(),
            'test_labels': y_test_str.tolist()
        }
        
        return results
    
    def train_specialization_classifier(
        self,
        cvs: List[Dict[str, Any]],
        threshold: float = 0.4
    ) -> Dict[str, Any]:
        """
        Train binary classifier for specialist vs. polyvalent.
        
        Uses specialization_score with threshold to create labels.
        
        Args:
            cvs (List[Dict]): CV data with specialization scores
            threshold (float): Threshold for specialist classification
        
        Returns:
            Dict: Training results
        """
        print(f"\n🎓 Training Specialization Classifier...")
        print(f"   Threshold: {threshold}")
        
        # Create binary labels
        y = np.array([
            1 if cv.get('specialization_score', 0) >= threshold else 0
            for cv in cvs
        ])
        
        # Use specialization-related features only
        X = np.array([
            [
                cv.get('n_skill_communities', 0),
                cv.get('skill_modularity', 0),
                len(cv['skills']),
                cv.get('experience_years', 0)
            ]
            for cv in cvs
        ])
        
        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=self.random_state, stratify=y
        )
        
        # Train classifier
        self.specialization_clf = RandomForestClassifier(
            n_estimators=50,
            max_depth=5,
            random_state=self.random_state
        )
        self.specialization_clf.fit(X_train, y_train)
        
        # Predict
        y_pred = self.specialization_clf.predict(X_test)
        
        # Metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"✅ Specialization Classification Accuracy: {accuracy:.3f}")
        
        # Class distribution
        n_specialists = np.sum(y)
        n_polyvalents = len(y) - n_specialists
        print(f"   Specialists: {n_specialists} ({n_specialists/len(y)*100:.1f}%)")
        print(f"   Polyvalents: {n_polyvalents} ({n_polyvalents/len(y)*100:.1f}%)")
        
        results = {
            'accuracy': accuracy,
            'threshold': threshold,
            'n_specialists': int(n_specialists),
            'n_polyvalents': int(n_polyvalents)
        }
        
        return results
    
    def semi_supervised_classification(
        self,
        X: np.ndarray,
        cvs: List[Dict[str, Any]],
        G,
        labeled_ratio: float = 0.8,
        max_iterations: int = 10
    ) -> Dict[str, Any]:
        """
        Implement iterative classification algorithm (REV2 from course).
        
        Args:
            X (np.ndarray): Feature matrix
            cvs (List[Dict]): CV data
            G: NetworkX graph
            labeled_ratio (float): Proportion of labeled data
            max_iterations (int): Maximum iterations
        
        Returns:
            Dict: Semi-supervised results
        """
        print(f"\n🔄 Semi-Supervised Iterative Classification (REV2)...")
        print(f"   Labeled ratio: {labeled_ratio}")
        print(f"   Max iterations: {max_iterations}")
        
        # Get labels
        y = np.array([cv['seniority'] for cv in cvs])
        y_encoded = self.label_encoder.fit_transform(y)
        
        # Split into labeled and unlabeled
        n_labeled = int(len(cvs) * labeled_ratio)
        indices = np.random.permutation(len(cvs))
        labeled_indices = indices[:n_labeled]
        unlabeled_indices = indices[n_labeled:]
        
        # Initialize predictions for unlabeled
        y_semi = y_encoded.copy()
        
        # Train initial classifier on labeled data
        clf = RandomForestClassifier(n_estimators=100, random_state=self.random_state)
        clf.fit(X[labeled_indices], y_semi[labeled_indices])
        
        # Iterative classification
        prev_accuracy = 0
        for iteration in range(max_iterations):
            # Predict unlabeled
            y_semi[unlabeled_indices] = clf.predict(X[unlabeled_indices])
            
            # Add relational features (neighbor label counts)
            X_augmented = self._add_relational_features(X, y_semi, cvs, G)
            
            # Retrain
            clf.fit(X_augmented[labeled_indices], y_semi[labeled_indices])
            
            # Evaluate on unlabeled
            accuracy = accuracy_score(y_encoded[unlabeled_indices], y_semi[unlabeled_indices])
            
            print(f"   Iteration {iteration+1}: Accuracy = {accuracy:.3f}")
            
            # Check convergence
            if abs(accuracy - prev_accuracy) < 0.001:
                print(f"   ✅ Converged at iteration {iteration+1}")
                break
            
            prev_accuracy = accuracy
        
        final_accuracy = accuracy_score(y_encoded[unlabeled_indices], y_semi[unlabeled_indices])
        
        print(f"✅ Semi-Supervised Final Accuracy: {final_accuracy:.3f}")
        
        results = {
            'final_accuracy': final_accuracy,
            'n_iterations': iteration + 1,
            'n_labeled': n_labeled,
            'n_unlabeled': len(unlabeled_indices)
        }
        
        return results
    
    def _add_relational_features(self, X, y_current, cvs, G):
        """Add neighbor label distribution features."""
        X_augmented = []
        
        cv_to_idx = {cv['cv_id']: i for i, cv in enumerate(cvs)}
        
        for i, cv in enumerate(cvs):
            cv_id = cv['cv_id']
            
            # Get neighbors
            neighbors = list(G.neighbors(cv_id))
            
            # Count neighbor labels (only CV neighbors)
            cv_neighbors = [n for n in neighbors if n in cv_to_idx]
            
            if cv_neighbors:
                neighbor_labels = [y_current[cv_to_idx[n]] for n in cv_neighbors]
                label_counts = np.bincount(neighbor_labels, minlength=3)
                label_probs = label_counts / len(cv_neighbors)
            else:
                label_probs = np.zeros(3)
            
            # Concatenate original features with relational features
            augmented_features = np.concatenate([X[i], label_probs])
            X_augmented.append(augmented_features)
        
        return np.array(X_augmented)
    
    def save_models(self, output_dir: str = 'models'):
        """Save trained models."""
        if self.seniority_clf:
            with open(f'{output_dir}/cv_seniority_classifier.pkl', 'wb') as f:
                pickle.dump(self.seniority_clf, f)
            print(f"✅ Saved seniority classifier")
        
        if self.specialization_clf:
            with open(f'{output_dir}/cv_specialization_classifier.pkl', 'wb') as f:
                pickle.dump(self.specialization_clf, f)
            print(f"✅ Saved specialization classifier")
        
        # Save label encoder
        with open(f'{output_dir}/cv_label_encoder.pkl', 'wb') as f:
            pickle.dump(self.label_encoder, f)
    
    def plot_confusion_matrix(
        self,
        results: Dict[str, Any],
        output_path: str = 'results/figures/seniority_confusion_matrix.png'
    ):
        """Plot confusion matrix for seniority classification."""
        cm = np.array(results['confusion_matrix'])
        labels = ['junior', 'mid', 'senior']
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=labels, yticklabels=labels)
        plt.title(f'Seniority Classification\nAccuracy: {results["accuracy"]:.3f}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved confusion matrix to {output_path}")
        plt.close()


if __name__ == "__main__":
    print("CV Classifier module - use in main pipeline")