"""
Metrics Computer Module (SHARED)

This module computes structural metrics for the bipartite graph:
- Centrality measures (degree, betweenness, PageRank)
- Community metrics
- Network statistics

Owner: Shared (used by both Ramy and Chiheb)
Dependencies: graph_builder.py
"""

import pandas as pd
import networkx as nx
import numpy as np
from typing import Dict, Any, List


class MetricsComputer:
    """
    Computes various graph metrics and centrality measures.
    """
    
    def __init__(self):
        """Initialize the metrics computer."""
        pass
    
    def compute_all_metrics(
        self,
        G: nx.Graph,
        cv_nodes: List[str],
        job_nodes: List[str]
    ) -> pd.DataFrame:
        """
        Compute all structural metrics for nodes.
        
        Args:
            G (nx.Graph): Bipartite graph
            cv_nodes (List[str]): CV node IDs
            job_nodes (List[str]): Job node IDs
        
        Returns:
            pd.DataFrame: Node metrics
        """
        print(f"📊 Computing structural metrics...")
        
        metrics_data = []
        
        # Compute centrality measures
        print("   Computing degree centrality...")
        degree_centrality = dict(G.degree())
        
        print("   Computing betweenness centrality...")
        betweenness = nx.betweenness_centrality(G)
        
        print("   Computing PageRank...")
        pagerank = nx.pagerank(G)
        
        print("   Computing clustering coefficient...")
        clustering = nx.clustering(G)
        
        # Collect metrics for each node
        for node in G.nodes():
            node_metrics = {
                'node': node,
                'node_type': 'cv' if node in cv_nodes else 'job',
                'degree': degree_centrality.get(node, 0),
                'betweenness': betweenness.get(node, 0),
                'pagerank': pagerank.get(node, 0),
                'clustering': clustering.get(node, 0)
            }
            
            # Add node attributes
            if node in cv_nodes:
                node_metrics['seniority'] = G.nodes[node].get('seniority', '')
                node_metrics['domain'] = G.nodes[node].get('domain', '')
            elif node in job_nodes:
                node_metrics['level'] = G.nodes[node].get('level', '')
                node_metrics['domain'] = G.nodes[node].get('domain', '')
            
            metrics_data.append(node_metrics)
        
        df = pd.DataFrame(metrics_data)
        
        print(f"✅ Computed metrics for {len(df)} nodes")
        self._print_metrics_summary(df)
        
        return df
    
    def add_community_metrics(
        self,
        metrics_df: pd.DataFrame,
        communities: Dict[str, int],
        G: nx.Graph
    ) -> pd.DataFrame:
        """
        Add community-related metrics to the DataFrame.
        
        Args:
            metrics_df (pd.DataFrame): Existing metrics
            communities (Dict): Community assignments
            G (nx.Graph): Graph
        
        Returns:
            pd.DataFrame: Updated metrics with community info
        """
        print(f"📊 Adding community metrics...")
        
        # Add community ID
        metrics_df['community_id'] = metrics_df['node'].map(communities)
        
        # Compute community sizes
        community_sizes = {}
        for comm_id in set(communities.values()):
            size = sum(1 for v in communities.values() if v == comm_id)
            community_sizes[comm_id] = size
        
        metrics_df['community_size'] = metrics_df['community_id'].map(community_sizes)
        
        print(f"✅ Added community metrics")
        
        return metrics_df
    
    def save_metrics(
        self,
        metrics_df: pd.DataFrame,
        filepath: str = 'results/metrics/node_metrics.csv'
    ):
        """
        Save metrics to CSV.
        
        Args:
            metrics_df (pd.DataFrame): Metrics data
            filepath (str): Output file path
        """
        metrics_df.to_csv(filepath, index=False)
        print(f"✅ Saved metrics to {filepath}")
    
    def get_cv_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Extract CV metrics only."""
        return metrics_df[metrics_df['node_type'] == 'cv'].copy()
    
    def get_job_metrics(self, metrics_df: pd.DataFrame) -> pd.DataFrame:
        """Extract Job metrics only."""
        return metrics_df[metrics_df['node_type'] == 'job'].copy()
    
    def _print_metrics_summary(self, df: pd.DataFrame):
        """Print summary statistics of metrics."""
        print("\n   Metrics Summary:")
        print(f"   Degree - min: {df['degree'].min()}, max: {df['degree'].max()}, avg: {df['degree'].mean():.2f}")
        print(f"   Betweenness - min: {df['betweenness'].min():.4f}, max: {df['betweenness'].max():.4f}, avg: {df['betweenness'].mean():.4f}")
        print(f"   PageRank - min: {df['pagerank'].min():.6f}, max: {df['pagerank'].max():.6f}, avg: {df['pagerank'].mean():.6f}")


if __name__ == "__main__":
    print("Metrics Computer module - use in main pipeline")