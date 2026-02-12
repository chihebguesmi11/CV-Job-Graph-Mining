"""
Graph Builder Module (SHARED)

This module constructs the main bipartite CV-Job graph.
Both Ramy and Chiheb use this module - coordinate carefully!

Owner: Shared (Ramy initializes, Chiheb adds Jobs)
Dependencies: cv_generator.py (Ramy), job_generator.py (Chiheb)
"""

import json
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple


class BipartiteGraphBuilder:
    """
    Builds and manages the CV-Job bipartite graph.
    """
    
    def __init__(self):
        """Initialize the graph builder."""
        self.G = nx.Graph()
        self.cv_nodes = []
        self.job_nodes = []
    
    def build_initial_graph(
        self, 
        cvs: List[Dict[str, Any]], 
        n_initial_edges: int = 175
    ) -> nx.Graph:
        """
        Build initial bipartite graph with CV nodes and placeholder edges.
        This is called by RAMY first.
        
        Args:
            cvs (List[Dict]): List of CV dictionaries
            n_initial_edges (int): Number of initial edges to create (150-200)
        
        Returns:
            nx.Graph: Initial bipartite graph
        """
        print(f"🔨 Building initial bipartite graph...")
        print(f"   Adding {len(cvs)} CV nodes...")
        
        # Add CV nodes with attributes (bipartite=0)
        for cv in cvs:
            self.G.add_node(
                cv['cv_id'],
                bipartite=0,
                node_type='cv',
                skills=cv['skills'],
                experience_years=cv['experience_years'],
                education=cv['education'],
                seniority=cv['seniority'],
                domain=cv['domain'],
                raw_text=cv['raw_text']
            )
        
        self.cv_nodes = [cv['cv_id'] for cv in cvs]
        
        print(f"✅ Added {len(self.cv_nodes)} CV nodes")
        print(f"   Graph will have {n_initial_edges} initial edges (to be added by Chiheb)")
        
        return self.G
    
    def add_job_nodes(
        self, 
        jobs: List[Dict[str, Any]]
    ) -> nx.Graph:
        """
        Add Job nodes to the graph.
        This is called by CHIHEB after getting the graph from Ramy.
        
        Args:
            jobs (List[Dict]): List of Job dictionaries
        
        Returns:
            nx.Graph: Updated bipartite graph
        """
        print(f"🔨 Adding {len(jobs)} Job nodes to graph...")
        
        # Add Job nodes with attributes (bipartite=1)
        for job in jobs:
            self.G.add_node(
                job['job_id'],
                bipartite=1,
                node_type='job',
                required_skills=job['required_skills'],
                experience_required=job['experience_required'],
                level=job['level'],
                domain=job['domain'],
                description=job['description']
            )
        
        self.job_nodes = [job['job_id'] for job in jobs]
        
        print(f"✅ Added {len(self.job_nodes)} Job nodes")
        print(f"   Total nodes: {self.G.number_of_nodes()}")
        
        return self.G
    
    def add_initial_edges(
        self, 
        cvs: List[Dict[str, Any]], 
        jobs: List[Dict[str, Any]],
        n_edges: int = 175
    ) -> nx.Graph:
        """
        Add initial CV-Job edges based on skill matching.
        This is called by CHIHEB after adding Job nodes.
        
        Args:
            cvs (List[Dict]): List of CVs
            jobs (List[Dict]): List of Jobs
            n_edges (int): Target number of edges (150-200)
        
        Returns:
            nx.Graph: Graph with initial edges
        """
        print(f"🔨 Adding {n_edges} initial edges...")
        
        # Create edges based on skill similarity
        edges_to_add = []
        
        for cv in cvs:
            cv_skills = set(cv['skills'])
            
            # Find compatible jobs
            compatible_jobs = []
            for job in jobs:
                job_skills = set(job['required_skills'])
                overlap = len(cv_skills & job_skills)
                
                if overlap > 0:
                    # Compute edge weight based on skill overlap
                    weight = overlap / max(len(cv_skills), len(job_skills))
                    # Add experience matching bonus
                    exp_diff = abs(cv['experience_years'] - job['experience_required'])
                    exp_factor = max(0, 1 - exp_diff / 10)
                    final_weight = (weight + exp_factor) / 2
                    
                    compatible_jobs.append((job['job_id'], final_weight))
            
            # Sort by weight and add top matches
            compatible_jobs.sort(key=lambda x: x[1], reverse=True)
            
            # Add 1-3 edges per CV
            n_cv_edges = np.random.randint(1, 4)
            for job_id, weight in compatible_jobs[:n_cv_edges]:
                edges_to_add.append((cv['cv_id'], job_id, weight))
        
        # Sample to get exactly n_edges
        if len(edges_to_add) > n_edges:
            # Sample higher-weight edges preferentially
            weights = np.array([e[2] for e in edges_to_add])
            probs = weights / weights.sum()
            indices = np.random.choice(
                len(edges_to_add), 
                size=n_edges, 
                replace=False, 
                p=probs
            )
            edges_to_add = [edges_to_add[i] for i in indices]
        
        # Add edges to graph
        for cv_id, job_id, weight in edges_to_add:
            self.G.add_edge(cv_id, job_id, weight=weight)
        
        print(f"✅ Added {self.G.number_of_edges()} edges")
        self._print_graph_stats()
        
        return self.G
    
    def load_graph(self, filepath: str = 'data/cv_job_graph.pkl') -> nx.Graph:
        """
        Load graph from pickle file.
        
        Args:
            filepath (str): Path to graph file
        
        Returns:
            nx.Graph: Loaded graph
        """
        self.G = nx.read_gpickle(filepath)
        
        # Extract CV and Job nodes
        self.cv_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('bipartite') == 0]
        self.job_nodes = [n for n in self.G.nodes() if self.G.nodes[n].get('bipartite') == 1]
        
        print(f"✅ Loaded graph from {filepath}")
        self._print_graph_stats()
        
        return self.G
    
    def save_graph(self, filepath: str = 'data/cv_job_graph.pkl'):
        """
        Save graph to pickle file.
        
        Args:
            filepath (str): Output file path
        """
        nx.write_gpickle(self.G, filepath)
        print(f"✅ Saved graph to {filepath}")
    
    def get_graph_statistics(self) -> Dict[str, Any]:
        """
        Compute and return graph statistics.
        
        Returns:
            Dict: Graph statistics
        """
        stats = {
            'n_cv_nodes': len(self.cv_nodes),
            'n_job_nodes': len(self.job_nodes),
            'n_total_nodes': self.G.number_of_nodes(),
            'n_edges': self.G.number_of_edges(),
            'density': nx.density(self.G),
            'cv_degree_avg': np.mean([self.G.degree(cv) for cv in self.cv_nodes]) if self.cv_nodes else 0,
            'job_degree_avg': np.mean([self.G.degree(job) for job in self.job_nodes]) if self.job_nodes else 0
        }
        
        if self.G.number_of_nodes() > 0:
            stats['avg_clustering'] = nx.average_clustering(self.G)
        
        return stats
    
    def save_statistics(self, filepath: str = 'data/graph_statistics.json'):
        """
        Save graph statistics to JSON file.
        
        Args:
            filepath (str): Output file path
        """
        stats = self.get_graph_statistics()
        
        with open(filepath, 'w') as f:
            json.dump(stats, f, indent=2)
        
        print(f"✅ Saved graph statistics to {filepath}")
    
    def _print_graph_stats(self):
        """Print graph statistics."""
        stats = self.get_graph_statistics()
        
        print("\n📊 Graph Statistics:")
        print(f"   CV nodes: {stats['n_cv_nodes']}")
        print(f"   Job nodes: {stats['n_job_nodes']}")
        print(f"   Total nodes: {stats['n_total_nodes']}")
        print(f"   Edges: {stats['n_edges']}")
        print(f"   Density: {stats['density']:.4f}")
        print(f"   Avg CV degree: {stats['cv_degree_avg']:.2f}")
        print(f"   Avg Job degree: {stats['job_degree_avg']:.2f}")
        if 'avg_clustering' in stats:
            print(f"   Avg clustering: {stats['avg_clustering']:.4f}")


if __name__ == "__main__":
    # Example: Ramy's usage
    print("=== RAMY'S WORKFLOW ===")
    
    # Load CVs
    with open('data/cvs.json', 'r') as f:
        cvs = json.load(f)
    
    # Build initial graph
    builder = BipartiteGraphBuilder()
    G = builder.build_initial_graph(cvs, n_initial_edges=175)
    
    # Save graph and statistics
    builder.save_graph('data/cv_job_graph.pkl')
    builder.save_statistics('data/graph_statistics.json')
    
    print("\n✅ Initial graph construction complete!")
    print("📤 Handoff: Graph ready for Chiheb to add Job nodes")