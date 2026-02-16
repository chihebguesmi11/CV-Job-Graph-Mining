"""
CV Internal Skill Graph Builder Module

This module creates internal skill co-occurrence graphs for each CV.
Skills are nodes, and edges represent co-occurrence in the same experience/project.

Owner: Ramy
Dependencies: cv_generator.py (CV data)
"""

import json
import networkx as nx
from typing import List, Dict, Any
import matplotlib.pyplot as plt


class CVSkillGraphBuilder:
    """
    Builds internal skill co-occurrence graphs for CVs.
    """
    
    def __init__(self):
        """Initialize the skill graph builder."""
        pass
    
    def build_skill_graphs(self, cvs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Build skill co-occurrence graphs for all CVs.
        
        Args:
            cvs (List[Dict]): List of CV dictionaries
        
        Returns:
            List[Dict]: Updated CVs with internal_skill_graph attribute
        """
        print(f"🔨 Building skill graphs for {len(cvs)} CVs...")
        
        for cv in cvs:
            skill_graph = self._build_single_skill_graph(cv)
            cv['internal_skill_graph'] = skill_graph
            cv['skill_graph_metrics'] = self._compute_graph_metrics(skill_graph)
        
        print(f"✅ Built {len(cvs)} skill graphs")
        self._print_statistics(cvs)
        
        return cvs
    
    def _build_single_skill_graph(self, cv: Dict[str, Any]) -> nx.Graph:
        """
        Build skill co-occurrence graph for a single CV.
        
        Args:
            cv (Dict): CV dictionary with skills and experiences
        
        Returns:
            nx.Graph: Skill co-occurrence graph
        """
        G = nx.Graph()
        
        # Add all skills as nodes
        skills = cv['skills']
        G.add_nodes_from(skills)
        
        # Add edges for skills that co-occur in same experience
        experiences = cv.get('experiences', [])
        
        for exp in experiences:
            exp_skills = exp.get('skills_used', [])
            
            # Create edges between all pairs of skills in this experience
            for i, skill1 in enumerate(exp_skills):
                for skill2 in exp_skills[i+1:]:
                    if G.has_edge(skill1, skill2):
                        # Increment weight if edge exists
                        G[skill1][skill2]['weight'] += 1
                    else:
                        # Create new edge with weight 1
                        G.add_edge(skill1, skill2, weight=1)
        
        return G
    
    def _compute_graph_metrics(self, G: nx.Graph) -> Dict[str, float]:
        """
        Compute basic metrics for a skill graph.
        
        Args:
            G (nx.Graph): Skill graph
        
        Returns:
            Dict: Graph metrics
        """
        metrics = {
            'n_nodes': G.number_of_nodes(),
            'n_edges': G.number_of_edges(),
            'density': nx.density(G) if G.number_of_nodes() > 1 else 0,
            'avg_degree': sum(dict(G.degree()).values()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
        }
        
        # Connected components
        if G.number_of_nodes() > 0:
            metrics['n_components'] = nx.number_connected_components(G)
        else:
            metrics['n_components'] = 0
        
        return metrics
    
    def save_cvs_with_graphs(
        self, 
        cvs: List[Dict[str, Any]], 
        filepath: str = 'data/cvs_with_graphs.json'
    ):
        """
        Save CVs with skill graphs to JSON.
        Note: NetworkX graphs are converted to edge lists for JSON serialization.
        
        Args:
            cvs (List[Dict]): CVs with skill graphs
            filepath (str): Output file path
        """
        cvs_serializable = []
        
        for cv in cvs:
            cv_copy = cv.copy()
            
            # Convert NetworkX graph to edge list
            if 'internal_skill_graph' in cv_copy:
                G = cv_copy['internal_skill_graph']
                cv_copy['internal_skill_graph'] = {
                    'nodes': list(G.nodes()),
                    'edges': [
                        {
                            'source': u, 
                            'target': v, 
                            'weight': G[u][v].get('weight', 1)
                        } 
                        for u, v in G.edges()
                    ]
                }
            
            cvs_serializable.append(cv_copy)
        
        with open(filepath, 'w') as f:
            json.dump(cvs_serializable, f, indent=2)
        
        print(f"✅ Saved {len(cvs)} CVs with skill graphs to {filepath}")
    
    def load_cvs_with_graphs(self, filepath: str = 'data/cvs_with_graphs.json') -> List[Dict[str, Any]]:
        """
        Load CVs and reconstruct NetworkX graphs from edge lists.
        
        Args:
            filepath (str): Input file path
        
        Returns:
            List[Dict]: CVs with reconstructed skill graphs
        """
        with open(filepath, 'r') as f:
            cvs = json.load(f)
        
        for cv in cvs:
            if 'internal_skill_graph' in cv and isinstance(cv['internal_skill_graph'], dict):
                # Reconstruct NetworkX graph from edge list
                G = nx.Graph()
                G.add_nodes_from(cv['internal_skill_graph']['nodes'])
                
                for edge in cv['internal_skill_graph']['edges']:
                    G.add_edge(
                        edge['source'], 
                        edge['target'], 
                        weight=edge.get('weight', 1)
                    )
                
                cv['internal_skill_graph'] = G
        
        print(f"✅ Loaded {len(cvs)} CVs with skill graphs from {filepath}")
        return cvs
    
    def visualize_skill_graphs(
        self, 
        cvs: List[Dict[str, Any]], 
        n_examples: int = 3,
        output_path: str = 'results/figures/skill_graph_examples.png'
    ):
        """
        Visualize example skill graphs from top CVs.
        
        Args:
            cvs (List[Dict]): CVs with skill graphs
            n_examples (int): Number of examples to show
            output_path (str): Output file path
        """
        # Sort CVs by number of skills (show diverse examples)
        cvs_sorted = sorted(cvs, key=lambda x: len(x['skills']), reverse=True)
        example_cvs = cvs_sorted[:n_examples]
        
        fig, axes = plt.subplots(1, n_examples, figsize=(15, 5))
        if n_examples == 1:
            axes = [axes]
        
        for i, cv in enumerate(example_cvs):
            G = cv['internal_skill_graph']
            
            # Create layout
            pos = nx.spring_layout(G, seed=42)
            
            # Draw graph
            ax = axes[i]
            nx.draw_networkx_nodes(G, pos, ax=ax, node_color='lightblue', 
                                   node_size=500, alpha=0.8)
            nx.draw_networkx_labels(G, pos, ax=ax, font_size=8)
            
            # Draw edges with weights
            edges = G.edges()
            weights = [G[u][v].get('weight', 1) for u, v in edges]
            nx.draw_networkx_edges(G, pos, ax=ax, width=weights, alpha=0.5)
            
            ax.set_title(f"{cv['cv_id']}\n{cv['seniority']} - {cv['domain']}\n"
                        f"{G.number_of_nodes()} skills, {G.number_of_edges()} connections",
                        fontsize=10)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved skill graph visualizations to {output_path}")
        plt.close()
    
    def _print_statistics(self, cvs: List[Dict[str, Any]]):
        """Print statistics about skill graphs."""
        print("\n📊 Skill Graph Statistics:")
        
        n_nodes = [cv['skill_graph_metrics']['n_nodes'] for cv in cvs]
        n_edges = [cv['skill_graph_metrics']['n_edges'] for cv in cvs]
        densities = [cv['skill_graph_metrics']['density'] for cv in cvs]
        
        print(f"   Avg nodes per graph: {sum(n_nodes)/len(n_nodes):.1f}")
        print(f"   Avg edges per graph: {sum(n_edges)/len(n_edges):.1f}")
        print(f"   Avg density: {sum(densities)/len(densities):.3f}")
        print(f"   Density range: {min(densities):.3f} - {max(densities):.3f}")


if __name__ == "__main__":
    # Load CVs
    with open('data/cvs.json', 'r') as f:
        cvs = json.load(f)
    
    # Build skill graphs
    builder = CVSkillGraphBuilder()
    cvs_with_graphs = builder.build_skill_graphs(cvs)
    
    # Save updated CVs
    builder.save_cvs_with_graphs(cvs_with_graphs, 'data/cvs_with_graphs.json')
    
    # Visualize examples
    builder.visualize_skill_graphs(cvs_with_graphs, n_examples=3)
    
    print("\n✅ Skill graph building complete!")