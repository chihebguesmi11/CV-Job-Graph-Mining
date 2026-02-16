"""
CV Analyzer Module

This module performs advanced CV analysis:
1. Atypical profile detection (polyvalent, bridge, misclassified)
2. Specialization coherence analysis

Owner: Ramy
Dependencies: cv_community.py, cv_classifier.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import List, Dict, Any


class CVAnalyzer:
    """
    Analyzes CVs to detect atypical profiles and coherence patterns.
    """
    
    def __init__(self):
        """Initialize the CV analyzer."""
        pass
    
    def detect_atypical_profiles(
        self,
        cvs: List[Dict[str, Any]],
        betweenness_centrality: Dict[str, float],
        classification_results: Dict[str, Any] = None
    ) -> Dict[str, List[Dict]]:
        """
        Detect atypical CV profiles.
        
        Categories:
        1. Polyvalent profiles (low specialization score)
        2. Bridge profiles (high betweenness centrality)
        3. Misclassified profiles (if classification results provided)
        
        Args:
            cvs (List[Dict]): CV data
            betweenness_centrality (Dict): Betweenness scores
            classification_results (Dict): Optional classification results
        
        Returns:
            Dict: Atypical profiles by category
        """
        print(f"\n🔍 Detecting Atypical Profiles...")
        
        atypical = {
            'polyvalent': [],
            'bridge': [],
            'misclassified': []
        }
        
        # 1. Polyvalent profiles (specialization < 0.3)
        polyvalent_cvs = [
            cv for cv in cvs
            if cv.get('specialization_score', 0) < 0.3
        ]
        
        for cv in polyvalent_cvs:
            atypical['polyvalent'].append({
                'cv_id': cv['cv_id'],
                'specialization_score': cv.get('specialization_score', 0),
                'n_skills': len(cv['skills']),
                'n_skill_communities': cv.get('n_skill_communities', 0),
                'domain': cv['domain'],
                'seniority': cv['seniority']
            })
        
        print(f"   Polyvalent CVs: {len(polyvalent_cvs)} ({len(polyvalent_cvs)/len(cvs)*100:.1f}%)")
        
        # 2. Bridge profiles (high betweenness - top 10%)
        betweenness_values = list(betweenness_centrality.values())
        if betweenness_values:
            betweenness_threshold = np.percentile(betweenness_values, 90)
            
            bridge_cvs = [
                cv for cv in cvs
                if betweenness_centrality.get(cv['cv_id'], 0) > betweenness_threshold
            ]
            
            for cv in bridge_cvs:
                atypical['bridge'].append({
                    'cv_id': cv['cv_id'],
                    'betweenness': betweenness_centrality.get(cv['cv_id'], 0),
                    'domain': cv['domain'],
                    'seniority': cv['seniority'],
                    'n_skills': len(cv['skills'])
                })
            
            print(f"   Bridge CVs: {len(bridge_cvs)} ({len(bridge_cvs)/len(cvs)*100:.1f}%)")
        
        # 3. Misclassified profiles
        if classification_results and 'test_predictions' in classification_results:
            y_pred = classification_results['test_predictions']
            y_test = classification_results['test_labels']
            
            # Find indices of misclassified samples
            misclassified_indices = [
                i for i in range(len(y_test))
                if y_pred[i] != y_test[i]
            ]
            
            # Get CV data for misclassified (assuming test set is last 20%)
            test_start_idx = int(len(cvs) * 0.8)
            
            for idx in misclassified_indices:
                if test_start_idx + idx < len(cvs):
                    cv = cvs[test_start_idx + idx]
                    atypical['misclassified'].append({
                        'cv_id': cv['cv_id'],
                        'true_seniority': y_test[idx],
                        'predicted_seniority': y_pred[idx],
                        'specialization_score': cv.get('specialization_score', 0),
                        'domain': cv['domain']
                    })
            
            print(f"   Misclassified CVs: {len(atypical['misclassified'])}")
            
            if atypical['misclassified']:
                avg_spec = np.mean([
                    cv['specialization_score'] for cv in atypical['misclassified']
                ])
                print(f"   Avg specialization of misclassified: {avg_spec:.3f}")
        
        return atypical
    
    def analyze_specialization_coherence(
        self,
        cvs: List[Dict[str, Any]],
        communities: Dict[str, int]
    ) -> Dict[str, Any]:
        """
        Analyze correlation between CV specialization and community specialization.
        
        Args:
            cvs (List[Dict]): CV data with specialization scores
            communities (Dict): Community assignments
        
        Returns:
            Dict: Coherence analysis results
        """
        print(f"\n📊 Specialization Coherence Analysis...")
        
        # Get CV specialization scores
        cv_spec_scores = []
        cv_ids = []
        
        for cv in cvs:
            cv_spec_scores.append(cv.get('specialization_score', 0))
            cv_ids.append(cv['cv_id'])
        
        cv_spec_scores = np.array(cv_spec_scores)
        
        # Compute community specialization (average of member specializations)
        community_spec = {}
        for comm_id in set(communities.values()):
            cvs_in_comm = [
                cv for cv in cvs
                if communities.get(cv['cv_id'], -1) == comm_id
            ]
            
            if cvs_in_comm:
                community_spec[comm_id] = np.mean([
                    cv.get('specialization_score', 0) for cv in cvs_in_comm
                ])
            else:
                community_spec[comm_id] = 0
        
        # Map each CV to its community specialization
        cv_comm_spec = []
        for cv in cvs:
            comm_id = communities.get(cv['cv_id'], -1)
            cv_comm_spec.append(community_spec.get(comm_id, 0))
        
        cv_comm_spec = np.array(cv_comm_spec)
        
        # Compute correlation
        if len(cv_spec_scores) > 1:
            correlation = np.corrcoef(cv_spec_scores, cv_comm_spec)[0, 1]
        else:
            correlation = 0
        
        print(f"✅ Correlation (CV spec vs Community spec): {correlation:.3f}")
        
        # Interpretation
        if abs(correlation) > 0.7:
            interpretation = "Strong correlation - CVs align well with community specialization"
        elif abs(correlation) > 0.4:
            interpretation = "Moderate correlation - Some alignment with community structure"
        else:
            interpretation = "Weak correlation - Community structure not driven by specialization"
        
        print(f"   {interpretation}")
        
        results = {
            'correlation': correlation,
            'interpretation': interpretation,
            'cv_specialization_avg': float(np.mean(cv_spec_scores)),
            'community_specialization_avg': float(np.mean(list(community_spec.values()))),
            'cv_specialization_std': float(np.std(cv_spec_scores)),
            'n_communities': len(community_spec)
        }
        
        return results
    
    def save_atypical_profiles(
        self,
        atypical_profiles: Dict[str, List[Dict]],
        filepath: str = 'results/analysis/atypical_cvs.json'
    ):
        """
        Save atypical profiles to JSON.
        
        Args:
            atypical_profiles (Dict): Atypical profile data
            filepath (str): Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(atypical_profiles, f, indent=2)
        
        print(f"✅ Saved atypical profiles to {filepath}")
    
    def save_coherence_analysis(
        self,
        coherence_results: Dict[str, Any],
        filepath: str = 'results/metrics/coherence_analysis.json'
    ):
        """
        Save coherence analysis results to JSON.
        
        Args:
            coherence_results (Dict): Coherence analysis data
            filepath (str): Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(coherence_results, f, indent=2)
        
        print(f"✅ Saved coherence analysis to {filepath}")
    
    def plot_specialization_coherence(
        self,
        cvs: List[Dict[str, Any]],
        communities: Dict[str, int],
        correlation: float,
        output_path: str = 'results/figures/specialization_coherence.png'
    ):
        """
        Create scatter plot of CV specialization vs community specialization.
        
        Args:
            cvs (List[Dict]): CV data
            communities (Dict): Community assignments
            correlation (float): Correlation coefficient
            output_path (str): Output file path
        """
        # Compute data points
        cv_spec_scores = []
        cv_comm_specs = []
        
        # Community specializations
        community_spec = {}
        for comm_id in set(communities.values()):
            cvs_in_comm = [
                cv for cv in cvs
                if communities.get(cv['cv_id'], -1) == comm_id
            ]
            
            if cvs_in_comm:
                community_spec[comm_id] = np.mean([
                    cv.get('specialization_score', 0) for cv in cvs_in_comm
                ])
        
        # Collect points
        for cv in cvs:
            cv_spec_scores.append(cv.get('specialization_score', 0))
            comm_id = communities.get(cv['cv_id'], -1)
            cv_comm_specs.append(community_spec.get(comm_id, 0))
        
        # Create plot
        plt.figure(figsize=(10, 6))
        plt.scatter(cv_spec_scores, cv_comm_specs, alpha=0.6, s=50)
        plt.xlabel('CV Specialization Score', fontsize=12)
        plt.ylabel('Community Specialization Score', fontsize=12)
        plt.title(f'Specialization Coherence\n(r = {correlation:.3f})', fontsize=14)
        
        # Add trend line
        z = np.polyfit(cv_spec_scores, cv_comm_specs, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(cv_spec_scores), max(cv_spec_scores), 100)
        plt.plot(x_line, p(x_line), "r--", alpha=0.8, linewidth=2)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved specialization coherence plot to {output_path}")
        plt.close()
    
    def generate_cv_analysis_summary(
        self,
        cvs: List[Dict[str, Any]],
        atypical_profiles: Dict[str, List[Dict]],
        coherence_results: Dict[str, Any],
        classification_results: Dict[str, Any]
    ) -> str:
        """
        Generate summary text for CV analysis.
        
        Args:
            cvs (List[Dict]): CV data
            atypical_profiles (Dict): Atypical profile data
            coherence_results (Dict): Coherence analysis results
            classification_results (Dict): Classification results
        
        Returns:
            str: Analysis summary
        """
        summary = f"""
# CV Analysis Summary

## Dataset Overview
- Total CVs: {len(cvs)}
- Domains: {', '.join(set(cv['domain'] for cv in cvs))}
- Seniority Levels: {', '.join(set(cv['seniority'] for cv in cvs))}

## Specialization Distribution
- Average Specialization Score: {coherence_results.get('cv_specialization_avg', 0):.3f}
- Specialists (>0.6): {len([cv for cv in cvs if cv.get('specialization_score', 0) > 0.6])}
- Polyvalents (<0.3): {len(atypical_profiles.get('polyvalent', []))}

## Classification Performance
- Seniority Classification Accuracy: {classification_results.get('accuracy', 0):.3f}

## Atypical Profiles
- Polyvalent CVs: {len(atypical_profiles.get('polyvalent', []))} ({len(atypical_profiles.get('polyvalent', []))/len(cvs)*100:.1f}%)
- Bridge CVs: {len(atypical_profiles.get('bridge', []))} ({len(atypical_profiles.get('bridge', []))/len(cvs)*100:.1f}%)
- Misclassified CVs: {len(atypical_profiles.get('misclassified', []))}

## Specialization Coherence
- Correlation: {coherence_results.get('correlation', 0):.3f}
- Interpretation: {coherence_results.get('interpretation', 'N/A')}

## Key Insights
1. Community structure {'is strongly' if abs(coherence_results.get('correlation', 0)) > 0.7 else 'is moderately' if abs(coherence_results.get('correlation', 0)) > 0.4 else 'is weakly'} related to specialization patterns
2. {len(atypical_profiles.get('polyvalent', []))/len(cvs)*100:.1f}% of CVs are polyvalent, indicating diverse skill sets
3. Classification accuracy of {classification_results.get('accuracy', 0):.3f} suggests {'good' if classification_results.get('accuracy', 0) > 0.7 else 'moderate'} predictability from features
"""
        
        return summary


if __name__ == "__main__":
    print("CV Analyzer module - use in main pipeline")