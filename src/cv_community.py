"""
CV Community Detection Module

This module performs community detection on:
1. Internal CV skill graphs (Louvain algorithm)
2. Global bipartite graph (CV communities)

Owner: Ramy
Dependencies: cv_graph_builder.py, graph_builder.py
"""

import json
import pandas as pd
import networkx as nx
import numpy as np
from typing import List, Dict, Any, Tuple
from community import best_partition
import community as community_louvain


class CVCommunityDetector:
    """
    Detects communities in CV skill graphs and the global bipartite graph.
    """
    
    def __init__(self):
        """Initialize the community detector."""
        pass
    
    def detect_skill_communities(
        self, 
        cvs: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Detect communities within each CV's skill graph using Louvain algorithm.
        
        Args:
            cvs (List[Dict]): CVs with internal skill graphs
        
        Returns:
            List[Dict]: CVs with skill community information
        """
        print(f"🔍 Detecting skill communities for {len(cvs)} CVs...")
        
        for cv in cvs:
            G = cv.get('internal_skill_graph')
            
            if G is None or G.number_of_nodes() == 0:
                cv['skill_communities'] = {}
                cv['n_skill_communities'] = 0
                cv['skill_modularity'] = 0
                cv['specialization_score'] = 0
                continue
            
            # Apply Louvain algorithm
            if G.number_of_edges() > 0:
                communities = best_partition(G)
                modularity = community_louvain.modularity(communities, G)
            else:
                # No edges = each skill is its own community
                communities = {skill: i for i, skill in enumerate(G.nodes())}
                modularity = 0
            
            # Count communities
            n_communities = len(set(communities.values()))
            
            # Compute specialization score
            # High modularity + few communities = specialized
            # Low modularity + many communities = polyvalent
            specialization_score = self._compute_specialization_score(
                modularity, n_communities, G.number_of_nodes()
            )
            
            cv['skill_communities'] = communities
            cv['n_skill_communities'] = n_communities
            cv['skill_modularity'] = modularity
            cv['specialization_score'] = specialization_score
        
        print(f"✅ Detected skill communities")
        self._print_skill_community_stats(cvs)
        
        return cvs
    
    def _compute_specialization_score(
        self, 
        modularity: float, 
        n_communities: int, 
        n_skills: int
    ) -> float:
        """
        Compute specialization score from skill graph structure.
        
        Score is high when:
        - High modularity (skills form tight clusters)
        - Few communities relative to skills (focused expertise)
        
        Args:
            modularity (float): Graph modularity
            n_communities (int): Number of communities
            n_skills (int): Total number of skills
        
        Returns:
            float: Specialization score [0, 1]
        """
        if n_skills == 0:
            return 0
        
        # Modularity component (0-1)
        mod_component = max(0, modularity)
        
        # Community concentration component
        # Few communities relative to skills = specialized
        community_ratio = n_communities / n_skills if n_skills > 0 else 1
        concentration_component = 1 - community_ratio
        
        # Combined score (weighted average)
        specialization = 0.6 * mod_component + 0.4 * concentration_component
        
        return max(0, min(1, specialization))
    
    def detect_global_communities(
        self, 
        G: nx.Graph,
        cv_nodes: List[str]
    ) -> Tuple[Dict[str, int], float]:
        """
        Detect communities in the global bipartite graph using Louvain.
        
        Args:
            G (nx.Graph): Bipartite CV-Job graph
            cv_nodes (List[str]): List of CV node IDs
        
        Returns:
            Tuple[Dict, float]: (community assignments, modularity Q)
        """
        print(f"🔍 Detecting global communities...")
        
        # Apply Louvain algorithm
        communities = best_partition(G)
        modularity = community_louvain.modularity(communities, G)
        
        n_communities = len(set(communities.values()))
        
        print(f"✅ Detected {n_communities} communities")
        print(f"   Modularity Q: {modularity:.3f}")
        
        if modularity > 0.3:
            print(f"   ✅ Significant community structure (Q > 0.3)")
        else:
            print(f"   ⚠️  Weak community structure (Q < 0.3)")
        
        # Analyze CV distribution
        self._analyze_cv_communities(G, communities, cv_nodes)
        
        return communities, modularity
    
    def _analyze_cv_communities(
        self, 
        G: nx.Graph,
        communities: Dict[str, int],
        cv_nodes: List[str]
    ):
        """
        Analyze how CVs are distributed across communities.
        
        Args:
            G (nx.Graph): Graph
            communities (Dict): Community assignments
            cv_nodes (List[str]): CV node IDs
        """
        print("\n📊 CV Community Distribution:")
        
        # Count CVs per community
        cv_communities = {}
        for cv in cv_nodes:
            comm_id = communities[cv]
            cv_communities[comm_id] = cv_communities.get(comm_id, 0) + 1
        
        # Print distribution
        for comm_id in sorted(cv_communities.keys()):
            count = cv_communities[comm_id]
            print(f"   Community {comm_id}: {count} CVs")
    
    def save_community_assignments(
        self,
        communities: Dict[str, int],
        G: nx.Graph,
        filepath: str = 'results/metrics/community_assignments.csv'
    ):
        """
        Save community assignments to CSV.
        
        Args:
            communities (Dict): Community assignments
            G (nx.Graph): Graph with node attributes
            filepath (str): Output file path
        """
        data = []
        
        for node_id, comm_id in communities.items():
            node_data = {
                'node_id': node_id,
                'community_id': comm_id,
                'node_type': G.nodes[node_id].get('node_type', 'unknown'),
                'degree': G.degree(node_id)
            }
            
            # Add type-specific attributes
            if G.nodes[node_id].get('node_type') == 'cv':
                node_data['seniority'] = G.nodes[node_id].get('seniority', '')
                node_data['domain'] = G.nodes[node_id].get('domain', '')
            
            data.append(node_data)
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        print(f"✅ Saved community assignments to {filepath}")
    
    def save_skill_communities(
        self,
        cvs: List[Dict[str, Any]],
        filepath: str = 'results/metrics/cv_skill_communities.csv'
    ):
        """
        Save CV skill community metrics to CSV.
        
        Args:
            cvs (List[Dict]): CVs with skill community data
            filepath (str): Output file path
        """
        data = []
        
        for cv in cvs:
            data.append({
                'cv_id': cv['cv_id'],
                'n_skills': len(cv['skills']),
                'n_skill_communities': cv.get('n_skill_communities', 0),
                'skill_modularity': cv.get('skill_modularity', 0),
                'specialization_score': cv.get('specialization_score', 0),
                'seniority': cv['seniority'],
                'domain': cv['domain']
            })
        
        df = pd.DataFrame(data)
        df.to_csv(filepath, index=False)
        
        print(f"✅ Saved skill community metrics to {filepath}")
    
    def _print_skill_community_stats(self, cvs: List[Dict[str, Any]]):
        """Print statistics about skill communities."""
        print("\n📊 Skill Community Statistics:")
        
        n_communities = [cv.get('n_skill_communities', 0) for cv in cvs]
        modularities = [cv.get('skill_modularity', 0) for cv in cvs]
        spec_scores = [cv.get('specialization_score', 0) for cv in cvs]
        
        print(f"   Avg communities per CV: {np.mean(n_communities):.1f}")
        print(f"   Avg modularity: {np.mean(modularities):.3f}")
        print(f"   Avg specialization: {np.mean(spec_scores):.3f}")
        print(f"   Specialization range: {min(spec_scores):.3f} - {max(spec_scores):.3f}")
        
        # Classify CVs
        specialists = [cv for cv in cvs if cv.get('specialization_score', 0) > 0.6]
        polyvalents = [cv for cv in cvs if cv.get('specialization_score', 0) < 0.3]
        
        print(f"\n   Specialists (score > 0.6): {len(specialists)} CVs")
        print(f"   Polyvalents (score < 0.3): {len(polyvalents)} CVs")


if __name__ == "__main__":
    # Load CVs with skill graphs
    from cv_graph_builder import CVSkillGraphBuilder
    
    builder = CVSkillGraphBuilder()
    cvs = builder.load_cvs_with_graphs('data/cvs_with_graphs.json')
    
    # Detect skill communities
    detector = CVCommunityDetector()
    cvs = detector.detect_skill_communities(cvs)
    
    # Save results
    detector.save_skill_communities(cvs)
    
    # Update CVs file
    builder.save_cvs_with_graphs(cvs, 'data/cvs_with_graphs.json')
    
    print("\n✅ CV skill community detection complete!")