"""
Visualizer Module (SHARED)

Common visualization utilities for the CV-Job graph project.

Owner: Shared (used by both Ramy and Chiheb)
Dependencies: None
"""

import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import numpy as np
import pandas as pd
from typing import Dict, List, Any


class GraphVisualizer:
    """
    Visualization utilities for graphs and analysis results.
    """
    
    def __init__(self):
        """Initialize the visualizer."""
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.dpi'] = 300
    
    def plot_bipartite_graph(
        self,
        G: nx.Graph,
        cv_nodes: List[str],
        job_nodes: List[str],
        communities: Dict[str, int] = None,
        output_path: str = 'results/figures/bipartite_graph.png',
        max_nodes: int = 150
    ):
        """
        Visualize bipartite graph with optional community colors.
        
        Args:
            G (nx.Graph): Bipartite graph
            cv_nodes (List[str]): CV node IDs
            job_nodes (List[str]): Job node IDs
            communities (Dict): Optional community assignments
            output_path (str): Output file path
            max_nodes (int): Maximum nodes to display (for clarity)
        """
        print(f"📊 Creating bipartite graph visualization...")
        
        # Sample if too many nodes
        if len(cv_nodes) + len(job_nodes) > max_nodes:
            sample_size_cv = min(len(cv_nodes), max_nodes // 2)
            sample_size_job = min(len(job_nodes), max_nodes // 2)
            
            cv_sample = np.random.choice(cv_nodes, sample_size_cv, replace=False)
            job_sample = np.random.choice(job_nodes, sample_size_job, replace=False)
            
            # Create subgraph
            nodes_to_keep = list(cv_sample) + list(job_sample)
            G_viz = G.subgraph(nodes_to_keep).copy()
            cv_viz = [n for n in cv_sample if n in G_viz]
            job_viz = [n for n in job_sample if n in G_viz]
        else:
            G_viz = G
            cv_viz = cv_nodes
            job_viz = job_nodes
        
        # Create layout
        pos = nx.bipartite_layout(G_viz, cv_viz, align='horizontal')
        
        # Create figure
        plt.figure(figsize=(15, 10))
        
        # Node colors
        if communities:
            # Color by community
            community_ids = [communities.get(n, -1) for n in cv_viz]
            cv_colors = plt.cm.tab20(np.array(community_ids) % 20)
            
            community_ids_job = [communities.get(n, -1) for n in job_viz]
            job_colors = plt.cm.tab20(np.array(community_ids_job) % 20)
        else:
            cv_colors = 'lightblue'
            job_colors = 'lightcoral'
        
        # Draw nodes
        nx.draw_networkx_nodes(
            G_viz, pos, nodelist=cv_viz,
            node_color=cv_colors, node_size=200,
            label='CVs', alpha=0.8
        )
        
        nx.draw_networkx_nodes(
            G_viz, pos, nodelist=job_viz,
            node_color=job_colors, node_size=200,
            label='Jobs', alpha=0.8
        )
        
        # Draw edges
        nx.draw_networkx_edges(G_viz, pos, alpha=0.2)
        
        plt.legend(fontsize=12)
        plt.title('CV-Job Bipartite Graph', fontsize=14)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved bipartite graph to {output_path}")
        plt.close()
    
    def plot_degree_distributions(
        self,
        G: nx.Graph,
        cv_nodes: List[str],
        job_nodes: List[str],
        output_path: str = 'results/figures/degree_distributions.png'
    ):
        """
        Plot degree distributions for CVs and Jobs separately.
        
        Args:
            G (nx.Graph): Graph
            cv_nodes (List[str]): CV node IDs
            job_nodes (List[str]): Job node IDs
            output_path (str): Output file path
        """
        print(f"📊 Creating degree distribution plots...")
        
        cv_degrees = [G.degree(cv) for cv in cv_nodes]
        job_degrees = [G.degree(job) for job in job_nodes]
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # CV degrees
        axes[0].hist(cv_degrees, bins=20, color='lightblue', edgecolor='black', alpha=0.7)
        axes[0].set_title('CV Degree Distribution', fontsize=12)
        axes[0].set_xlabel('Degree')
        axes[0].set_ylabel('Count')
        axes[0].axvline(np.mean(cv_degrees), color='red', linestyle='--', 
                        label=f'Mean: {np.mean(cv_degrees):.1f}')
        axes[0].legend()
        
        # Job degrees
        axes[1].hist(job_degrees, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        axes[1].set_title('Job Degree Distribution', fontsize=12)
        axes[1].set_xlabel('Degree')
        axes[1].set_ylabel('Count')
        axes[1].axvline(np.mean(job_degrees), color='red', linestyle='--',
                        label=f'Mean: {np.mean(job_degrees):.1f}')
        axes[1].legend()
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved degree distributions to {output_path}")
        plt.close()
    
    def plot_specialization_distribution(
        self,
        cvs: List[Dict[str, Any]],
        output_path: str = 'results/figures/cv_specialization_distribution.png'
    ):
        """
        Plot distribution of CV specialization scores.
        
        Args:
            cvs (List[Dict]): CV data with specialization scores
            output_path (str): Output file path
        """
        print(f"📊 Creating specialization distribution plot...")
        
        spec_scores = [cv.get('specialization_score', 0) for cv in cvs]
        
        plt.figure(figsize=(10, 6))
        plt.hist(spec_scores, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        plt.xlabel('Specialization Score', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        plt.title('CV Specialization Score Distribution', fontsize=14)
        
        # Add vertical lines for categories
        plt.axvline(0.3, color='orange', linestyle='--', label='Polyvalent threshold')
        plt.axvline(0.6, color='green', linestyle='--', label='Specialist threshold')
        plt.axvline(np.mean(spec_scores), color='red', linestyle='-', 
                   label=f'Mean: {np.mean(spec_scores):.3f}')
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved specialization distribution to {output_path}")
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_names: List[str],
        importances: np.ndarray,
        title: str = 'Feature Importance',
        output_path: str = 'results/figures/feature_importance.png',
        top_n: int = 15
    ):
        """
        Plot feature importance from Random Forest.
        
        Args:
            feature_names (List[str]): Feature names
            importances (np.ndarray): Feature importances
            title (str): Plot title
            output_path (str): Output file path
            top_n (int): Number of top features to show
        """
        print(f"📊 Creating feature importance plot...")
        
        # Get top N features
        indices = np.argsort(importances)[::-1][:top_n]
        top_features = [feature_names[i] for i in indices]
        top_importances = importances[indices]
        
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(top_features)), top_importances, color='steelblue', alpha=0.8)
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel('Importance', fontsize=12)
        plt.title(title, fontsize=14)
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved feature importance to {output_path}")
        plt.close()


if __name__ == "__main__":
    print("Visualizer module - use in main pipeline")