"""
Pipeline Visualizations - Sophisticated plots for each phase

Creates publication-quality visualizations for all pipeline phases.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings

warnings.filterwarnings('ignore')


class PipelineVisualizations:
    """Create sophisticated visualizations for pipeline phases."""
    
    def __init__(self, output_dir: str = 'results/figures'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        sns.set_style("whitegrid")
        sns.set_palette("husl")
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['font.size'] = 10
        plt.rcParams['axes.labelsize'] = 11
        plt.rcParams['axes.titlesize'] = 12
        plt.rcParams['xtick.labelsize'] = 9
        plt.rcParams['ytick.labelsize'] = 9
    
    def plot_graph_evolution(self, 
                            edges_before: int,
                            edges_after_community: int,
                            edges_after_prediction: int,
                            density_before: float,
                            density_after_prediction: float):
        """Plot graph growth across phases."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Edge count
        phases = ['Initial\nMatching', 'After\nPrediction']
        edge_counts = [edges_after_community, edges_after_prediction]
        colors = ['#3498db', '#2ecc71']
        
        axes[0].bar(phases, edge_counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[0].set_ylabel('Number of Edges', fontweight='bold')
        axes[0].set_title('Graph Edge Growth', fontweight='bold')
        axes[0].grid(axis='y', alpha=0.3)
        for i, v in enumerate(edge_counts):
            axes[0].text(i, v + 50, str(v), ha='center', fontweight='bold')
        
        # Density evolution
        densities = [density_before, density_after_prediction]
        axes[1].bar(phases, densities, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('Graph Density', fontweight='bold')
        axes[1].set_title('Graph Density Evolution', fontweight='bold')
        axes[1].set_ylim(0, max(densities) * 1.2)
        axes[1].grid(axis='y', alpha=0.3)
        for i, v in enumerate(densities):
            axes[1].text(i, v + 0.01, f'{v:.3f}', ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase1_graph_evolution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: phase1_graph_evolution.png")
        plt.close()
    
    def plot_community_structure(self, 
                                graph: nx.Graph,
                                communities: dict,
                                modularity: float,
                                cv_nodes: set,
                                job_nodes: set):
        """Plot community structure visualization."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Community size distribution
        comm_sizes = {}
        for node, comm_id in communities.items():
            if comm_id not in comm_sizes:
                comm_sizes[comm_id] = {'total': 0, 'cv': 0, 'job': 0}
            comm_sizes[comm_id]['total'] += 1
            if node in cv_nodes:
                comm_sizes[comm_id]['cv'] += 1
            else:
                comm_sizes[comm_id]['job'] += 1
        
        comm_ids = sorted(comm_sizes.keys())
        cv_counts = [comm_sizes[c]['cv'] for c in comm_ids]
        job_counts = [comm_sizes[c]['job'] for c in comm_ids]
        
        x = np.arange(len(comm_ids))
        width = 0.35
        
        axes[0].bar(x - width/2, cv_counts, width, label='CVs', alpha=0.8, edgecolor='black')
        axes[0].bar(x + width/2, job_counts, width, label='Jobs', alpha=0.8, edgecolor='black')
        axes[0].set_xlabel('Community ID', fontweight='bold')
        axes[0].set_ylabel('Node Count', fontweight='bold')
        axes[0].set_title('Community Composition', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels([f'C{c}' for c in comm_ids])
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Community metrics
        metrics = ['Communities', 'Modularity', 'Density']
        n_communities = len(set(communities.values()))
        density = nx.density(graph)
        values = [n_communities, modularity * 10, density]  # Scale modularity for visibility
        colors_metrics = ['#e74c3c', '#f39c12', '#3498db']
        
        bars = axes[1].bar(metrics, values, color=colors_metrics, alpha=0.8, edgecolor='black', linewidth=1.5)
        axes[1].set_ylabel('Value', fontweight='bold')
        axes[1].set_title('Community Detection Metrics', fontweight='bold')
        axes[1].grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, val, metric_name) in enumerate(zip(bars, values, metrics)):
            height = bar.get_height()
            if 'Modularity' in metric_name:
                label = f'{modularity:.3f}'
            elif 'Density' in metric_name:
                label = f'{density:.3f}'
            else:
                label = str(int(val))
            axes[1].text(bar.get_x() + bar.get_width()/2., height,
                        label, ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase2_community_structure.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: phase2_community_structure.png")
        plt.close()
    
    def plot_embedding_space(self, 
                            cv_embeddings: dict,
                            job_embeddings: dict,
                            cv_domains: dict = None,
                            job_domains: dict = None):
        """Plot embedding space using PCA/t-SNE."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Prepare data
        all_embeddings = []
        labels = []
        colors = []
        cv_ids = list(cv_embeddings.keys())
        job_ids = list(job_embeddings.keys())
        
        for cv_id in cv_ids:
            all_embeddings.append(cv_embeddings[cv_id])
            labels.append('CV')
            colors.append('#3498db')
        
        for job_id in job_ids:
            all_embeddings.append(job_embeddings[job_id])
            labels.append('Job')
            colors.append('#e74c3c')
        
        X = np.array(all_embeddings)
        
        # PCA
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X)
        
        axes[0].scatter(X_pca[:len(cv_ids), 0], X_pca[:len(cv_ids), 1], 
                       c='#3498db', label='CVs', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0].scatter(X_pca[len(cv_ids):, 0], X_pca[len(cv_ids):, 1], 
                       c='#e74c3c', label='Jobs', alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
        axes[0].set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})', fontweight='bold')
        axes[0].set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})', fontweight='bold')
        axes[0].set_title('Embedding Space (PCA)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(alpha=0.3)
        
        # Variance explained
        cumsum_var = np.cumsum(pca.explained_variance_ratio_)
        axes[1].plot(range(1, min(21, len(cumsum_var)+1)), cumsum_var[:20], 
                    marker='o', linewidth=2, markersize=6, color='#2c3e50', markeredgecolor='black')
        axes[1].fill_between(range(1, min(21, len(cumsum_var)+1)), cumsum_var[:20], 
                            alpha=0.3, color='#3498db')
        axes[1].set_xlabel('Number of Components', fontweight='bold')
        axes[1].set_ylabel('Cumulative Variance Explained', fontweight='bold')
        axes[1].set_title('PCA Variance Explained', fontweight='bold')
        axes[1].grid(alpha=0.3)
        axes[1].set_ylim(0, 1.05)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase3_embedding_space.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: phase3_embedding_space.png")
        plt.close()
    
    def plot_link_prediction_performance(self, 
                                        y_test: np.ndarray,
                                        y_pred_proba: np.ndarray,
                                        precision: float,
                                        recall: float,
                                        auc_roc: float,
                                        accuracy: float,
                                        feature_importance: dict):
        """Plot link prediction performance metrics."""
        fig = plt.figure(figsize=(15, 10))
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.35)
        
        # 1. ROC Curve
        from sklearn.metrics import roc_curve, auc
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(fpr, tpr, color='#e74c3c', lw=2.5, label=f'ROC (AUC={roc_auc:.3f})')
        ax1.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random')
        ax1.set_xlabel('False Positive Rate', fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontweight='bold')
        ax1.set_title('ROC Curve', fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)
        
        # 2. Precision-Recall Curve
        from sklearn.metrics import precision_recall_curve, f1_score
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_pred_proba)
        
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(recall_curve, precision_curve, color='#3498db', lw=2.5)
        ax2.fill_between(recall_curve, precision_curve, alpha=0.3, color='#3498db')
        ax2.set_xlabel('Recall', fontweight='bold')
        ax2.set_ylabel('Precision', fontweight='bold')
        ax2.set_title('Precision-Recall Curve', fontweight='bold')
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        
        # 3. Performance Metrics
        ax3 = fig.add_subplot(gs[0, 2])
        metrics_names = ['Precision', 'Recall', 'Accuracy', 'AUC-ROC']
        metrics_values = [precision, recall, accuracy, auc_roc]
        colors_metrics = ['#3498db', '#2ecc71', '#f39c12', '#e74c3c']
        
        bars = ax3.barh(metrics_names, metrics_values, color=colors_metrics, alpha=0.8, edgecolor='black', linewidth=1.5)
        ax3.set_xlabel('Score', fontweight='bold')
        ax3.set_title('Performance Metrics', fontweight='bold')
        ax3.set_xlim(0, 1)
        ax3.grid(axis='x', alpha=0.3)
        
        for bar, val in zip(bars, metrics_values):
            ax3.text(val + 0.02, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                    va='center', fontweight='bold', fontsize=9)
        
        # 4. Feature Importance
        ax4 = fig.add_subplot(gs[1:, :])
        if feature_importance:
            features = list(feature_importance.keys())
            importances = list(feature_importance.values())
            
            # Sort by importance
            sorted_pairs = sorted(zip(features, importances), key=lambda x: x[1], reverse=True)
            features_sorted, importances_sorted = zip(*sorted_pairs)
            
            colors_imp = plt.cm.RdYlGn(np.linspace(0.3, 0.9, len(features_sorted)))
            bars = ax4.barh(features_sorted, importances_sorted, color=colors_imp, edgecolor='black', linewidth=1.5)
            ax4.set_xlabel('Importance Score', fontweight='bold')
            ax4.set_title('Feature Importance (Random Forest)', fontweight='bold')
            ax4.grid(axis='x', alpha=0.3)
            
            for bar, val in zip(bars, importances_sorted):
                ax4.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}', 
                        va='center', fontweight='bold', fontsize=8)
        
        plt.suptitle('Link Prediction Performance Analysis', fontsize=14, fontweight='bold', y=0.995)
        plt.savefig(self.output_dir / 'phase4_link_prediction.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: phase4_link_prediction.png")
        plt.close()
    
    def plot_predicted_links_distribution(self, 
                                         predicted_links: list,
                                         n_cv: int,
                                         n_jobs: int):
        """Analyze distribution of predicted links."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # CV degree from predictions
        cv_degrees = {}
        job_degrees = {}
        
        for cv_id, job_id, prob in predicted_links:
            cv_degrees[cv_id] = cv_degrees.get(cv_id, 0) + 1
            job_degrees[job_id] = job_degrees.get(job_id, 0) + 1
        
        # CV degree distribution
        cv_degree_values = list(cv_degrees.values())
        axes[0].hist(cv_degree_values, bins=20, color='#3498db', alpha=0.8, edgecolor='black')
        axes[0].axvline(np.mean(cv_degree_values), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(cv_degree_values):.2f}')
        axes[0].set_xlabel('Number of Predicted Links per CV', fontweight='bold')
        axes[0].set_ylabel('Frequency', fontweight='bold')
        axes[0].set_title('CV Degree Distribution (Predicted)', fontweight='bold')
        axes[0].legend()
        axes[0].grid(axis='y', alpha=0.3)
        
        # Job degree distribution
        job_degree_values = list(job_degrees.values())
        axes[1].hist(job_degree_values, bins=20, color='#e74c3c', alpha=0.8, edgecolor='black')
        axes[1].axvline(np.mean(job_degree_values), color='darkred', linestyle='--', linewidth=2, label=f'Mean: {np.mean(job_degree_values):.2f}')
        axes[1].set_xlabel('Number of Predicted Links per Job', fontweight='bold')
        axes[1].set_ylabel('Frequency', fontweight='bold')
        axes[1].set_title('Job Degree Distribution (Predicted)', fontweight='bold')
        axes[1].legend()
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase4_predicted_links_distribution.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: phase4_predicted_links_distribution.png")
        plt.close()
    
    def plot_job_classification_summary(self,
                                       domain_report: dict,
                                       level_report: dict):
        """Plot job classification summary."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        
        # Domain classification
        domain_classes = [k for k in domain_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        domain_f1 = [domain_report[cls].get('f1-score', 0) for cls in domain_classes]
        domain_recall = [domain_report[cls].get('recall', 0) for cls in domain_classes]
        
        x = np.arange(len(domain_classes))
        width = 0.35
        
        axes[0].bar(x - width/2, domain_f1, width, label='F1-Score', alpha=0.8, edgecolor='black')
        axes[0].bar(x + width/2, domain_recall, width, label='Recall', alpha=0.8, edgecolor='black')
        axes[0].set_ylabel('Score', fontweight='bold')
        axes[0].set_xlabel('Domain Class', fontweight='bold')
        axes[0].set_title('Job Domain Classification Performance', fontweight='bold')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(domain_classes, rotation=15, ha='right')
        axes[0].legend()
        axes[0].set_ylim(0, 1.1)
        axes[0].grid(axis='y', alpha=0.3)
        
        # Level classification
        level_classes = [k for k in level_report.keys() if k not in ['accuracy', 'macro avg', 'weighted avg']]
        level_f1 = [level_report[cls].get('f1-score', 0) for cls in level_classes]
        level_recall = [level_report[cls].get('recall', 0) for cls in level_classes]
        
        x = np.arange(len(level_classes))
        
        axes[1].bar(x - width/2, level_f1, width, label='F1-Score', alpha=0.8, edgecolor='black')
        axes[1].bar(x + width/2, level_recall, width, label='Recall', alpha=0.8, edgecolor='black')
        axes[1].set_ylabel('Score', fontweight='bold')
        axes[1].set_xlabel('Level Class', fontweight='bold')
        axes[1].set_title('Job Level Classification Performance', fontweight='bold')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(level_classes, rotation=15, ha='right')
        axes[1].legend()
        axes[1].set_ylim(0, 1.1)
        axes[1].grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'phase5_job_classification.png', dpi=300, bbox_inches='tight')
        print(f"✓ Saved: phase5_job_classification.png")
        plt.close()
