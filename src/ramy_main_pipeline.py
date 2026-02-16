"""
RAMY's Main Pipeline

This script orchestrates all CV-related tasks for the CV-Job graph mining project.

Execution order:
1. Phase 1: Data Generation & Infrastructure (Days 1-2)
2. Phase 2: CV Community Analysis (Days 3-4)  
3. Phase 3: CV Metrics & Visualization (Days 4-5)
4. Phase 4: CV Classification (Days 6-7)
5. Phase 5: Analysis & Insights (Day 8)
6. Phase 6: Documentation (Day 9)

Owner: Ramy
"""

import os
import sys
import json
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from cv_generator import CVGenerator
from cv_graph_builder import CVSkillGraphBuilder
from graph_builder import BipartiteGraphBuilder
from cv_community import CVCommunityDetector
from metrics_computer import MetricsComputer
from visualizer import GraphVisualizer
from cv_classifier import CVClassifier
from cv_analyzer import CVAnalyzer


def setup_directories():
    """
    Create all necessary directories for the project.
    """
    directories = [
        'data',
        'results',
        'results/figures',
        'results/metrics',
        'results/analysis',
        'models',
        'tests',
        'notebooks'
    ]
    
    print("🔧 Setting up project directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    
    print("✅ Created all project directories\n")


def phase1_data_generation():
    """
    Phase 1: Data Generation & Infrastructure
    - Generate 100 CVs
    - Create internal skill graphs
    - Build initial bipartite graph structure
    """
    print("\n" + "="*80)
    print("PHASE 1: DATA GENERATION & INFRASTRUCTURE")
    print("="*80)
    
    # Task 1.1: Generate CVs
    print("\n--- Task 1.1: Generate CVs ---")
    generator = CVGenerator(seed=42)
    cvs = generator.generate_cvs(n_cvs=100)
    generator.save_cvs(cvs, 'data/cvs.json')
    
    # Task 1.3: Build internal skill graphs
    print("\n--- Task 1.3: Build Internal Skill Graphs ---")
    builder = CVSkillGraphBuilder()
    cvs_with_graphs = builder.build_skill_graphs(cvs)
    builder.save_cvs_with_graphs(cvs_with_graphs, 'data/cvs_with_graphs.json')
    builder.visualize_skill_graphs(cvs_with_graphs, n_examples=3)
    
    # Task 1.2: Build initial bipartite graph
    print("\n--- Task 1.2: Build Initial Graph Structure ---")
    graph_builder = BipartiteGraphBuilder()
    G = graph_builder.build_initial_graph(cvs, n_initial_edges=175)
    graph_builder.save_graph('data/cv_job_graph.pkl')
    graph_builder.save_statistics('data/graph_statistics.json')
    
    print("\n✅ PHASE 1 COMPLETE!")
    print("📤 Handoff: Graph ready for Chiheb to add Job nodes")
    
    return cvs_with_graphs, G


def phase2_community_analysis(cvs):
    """
    Phase 2: CV Community Analysis
    - Detect skill communities within CVs
    - Global community detection (after Chiheb adds Jobs)
    
    Note: Global community detection will be run after Chiheb's handoff
    """
    print("\n" + "="*80)
    print("PHASE 2: CV COMMUNITY ANALYSIS")
    print("="*80)
    
    # Task 2.1: CV Internal Community Detection
    print("\n--- Task 2.1: CV Internal Community Detection ---")
    detector = CVCommunityDetector()
    cvs = detector.detect_skill_communities(cvs)
    detector.save_skill_communities(cvs)
    
    # Update CVs file
    builder = CVSkillGraphBuilder()
    builder.save_cvs_with_graphs(cvs, 'data/cvs_with_graphs.json')
    
    print("\n✅ PHASE 2 (PART 1) COMPLETE!")
    print("⏳ Task 2.2 (Global Communities) waiting for Chiheb to add Job nodes...")
    
    return cvs


def phase2b_global_communities(G, cv_nodes, job_nodes):
    """
    Phase 2B: Global Community Detection (after Chiheb adds Jobs)
    This runs after coordination with Chiheb
    """
    print("\n" + "="*80)
    print("PHASE 2B: GLOBAL COMMUNITY DETECTION")
    print("="*80)
    
    print("\n--- Task 2.2: Global Community Detection ---")
    detector = CVCommunityDetector()
    communities, modularity = detector.detect_global_communities(G, cv_nodes)
    detector.save_community_assignments(communities, G)
    
    print(f"\n✅ PHASE 2B COMPLETE!")
    print(f"   Modularity Q: {modularity:.3f}")
    
    return communities, modularity


def phase3_metrics_visualization(G, cv_nodes, job_nodes, communities, cvs):
    """
    Phase 3: CV Metrics & Visualization
    - Compute structural metrics
    - Create visualizations
    """
    print("\n" + "="*80)
    print("PHASE 3: CV METRICS & VISUALIZATION")
    print("="*80)
    
    # Task 3.1: Compute CV structural metrics
    print("\n--- Task 3.1: Compute CV Structural Metrics ---")
    metrics_comp = MetricsComputer()
    metrics_df = metrics_comp.compute_all_metrics(G, cv_nodes, job_nodes)
    metrics_df = metrics_comp.add_community_metrics(metrics_df, communities, G)
    metrics_comp.save_metrics(metrics_df)
    
    # Extract CV metrics
    cv_metrics = metrics_comp.get_cv_metrics(metrics_df)
    cv_metrics.to_csv('results/metrics/cv_node_metrics.csv', index=False)
    
    # Task 3.2: Create visualizations
    print("\n--- Task 3.2: Create CV Visualizations ---")
    visualizer = GraphVisualizer()
    
    # Degree distribution
    visualizer.plot_degree_distributions(G, cv_nodes, job_nodes)
    
    # Specialization distribution
    visualizer.plot_specialization_distribution(cvs)
    
    # Bipartite graph with communities
    visualizer.plot_bipartite_graph(G, cv_nodes, job_nodes, communities)
    
    print("\n✅ PHASE 3 COMPLETE!")
    
    return metrics_df, cv_metrics


def phase4_classification(cvs, cv_metrics, G):
    """
    Phase 4: CV Classification
    - Feature engineering
    - Seniority classification
    - Specialization classification
    - Semi-supervised iterative classification
    """
    print("\n" + "="*80)
    print("PHASE 4: CV CLASSIFICATION")
    print("="*80)
    
    classifier = CVClassifier(random_state=42)
    
    # Task 4.1: Feature Engineering
    print("\n--- Task 4.1: CV Feature Engineering ---")
    # Note: Using without embeddings first (embeddings come from Chiheb)
    X, feature_names = classifier.build_feature_matrix(cvs, cv_metrics, embeddings=None)
    
    # Save feature matrix
    np.save('data/cv_feature_matrix.npy', X)
    with open('data/cv_feature_names.json', 'w') as f:
        json.dump(feature_names, f)
    
    # Task 4.2: Seniority Classification
    print("\n--- Task 4.2: Seniority Classification ---")
    seniority_results = classifier.train_seniority_classifier(X, cvs, test_size=0.2)
    
    # Save results
    with open('results/metrics/cv_seniority_metrics.json', 'w') as f:
        json.dump(seniority_results, f, indent=2)
    
    # Plot confusion matrix
    classifier.plot_confusion_matrix(seniority_results)
    
    # Plot feature importance
    if classifier.seniority_clf:
        visualizer = GraphVisualizer()
        visualizer.plot_feature_importance(
            feature_names,
            classifier.seniority_clf.feature_importances_,
            title='CV Seniority - Feature Importance',
            output_path='results/figures/cv_feature_importance.png'
        )
    
    # Task 4.3: Specialization Classification
    print("\n--- Task 4.3: Specialization Classification ---")
    spec_results = classifier.train_specialization_classifier(cvs, threshold=0.4)
    
    with open('results/metrics/cv_specialization_metrics.json', 'w') as f:
        json.dump(spec_results, f, indent=2)
    
    # Task 4.4: Semi-Supervised Classification
    print("\n--- Task 4.4: Semi-Supervised Iterative Classification (REV2) ---")
    semi_results = classifier.semi_supervised_classification(
        X, cvs, G, labeled_ratio=0.8, max_iterations=10
    )
    
    with open('results/metrics/semi_supervised_results.json', 'w') as f:
        json.dump(semi_results, f, indent=2)
    
    # Save models
    classifier.save_models('models')
    
    print("\n✅ PHASE 4 COMPLETE!")
    
    return classifier, seniority_results, spec_results, semi_results


def phase5_analysis(cvs, communities, cv_metrics, seniority_results):
    """
    Phase 5: Analysis & Insights
    - Detect atypical profiles
    - Specialization coherence analysis
    """
    print("\n" + "="*80)
    print("PHASE 5: ANALYSIS & INSIGHTS")
    print("="*80)
    
    analyzer = CVAnalyzer()
    
    # Task 5.1: Detect atypical profiles
    print("\n--- Task 5.1: Atypical CV Detection ---")
    betweenness_dict = cv_metrics.set_index('node')['betweenness'].to_dict()
    
    atypical_profiles = analyzer.detect_atypical_profiles(
        cvs, betweenness_dict, seniority_results
    )
    
    analyzer.save_atypical_profiles(atypical_profiles)
    
    # Task 5.2: Specialization coherence analysis
    print("\n--- Task 5.2: Specialization Coherence Analysis ---")
    coherence_results = analyzer.analyze_specialization_coherence(cvs, communities)
    
    analyzer.save_coherence_analysis(coherence_results)
    
    # Plot coherence
    analyzer.plot_specialization_coherence(
        cvs, communities, coherence_results['correlation']
    )
    
    print("\n✅ PHASE 5 COMPLETE!")
    
    return atypical_profiles, coherence_results


def phase6_documentation(cvs, atypical_profiles, coherence_results, seniority_results):
    """
    Phase 6: Documentation
    - Generate CV analysis report
    """
    print("\n" + "="*80)
    print("PHASE 6: DOCUMENTATION")
    print("="*80)
    
    print("\n--- Task 6.1: Generate CV Analysis Report ---")
    
    analyzer = CVAnalyzer()
    report = analyzer.generate_cv_analysis_summary(
        cvs, atypical_profiles, coherence_results, seniority_results
    )
    
    with open('results/cv_analysis_report.md', 'w') as f:
        f.write(report)
    
    print("✅ Saved CV analysis report to results/cv_analysis_report.md")
    
    print("\n✅ PHASE 6 COMPLETE!")
    print("\n" + "="*80)
    print("🎉 ALL RAMY'S TASKS COMPLETE!")
    print("="*80)


def main():
    """
    Main execution pipeline for Ramy's work.
    """
    print("\n" + "="*80)
    print("CV-JOB GRAPH MINING PROJECT - RAMY'S PIPELINE")
    print("="*80)
    
    # Setup directories FIRST
    setup_directories()
    
    # Execute phases sequentially
    
    # Phase 1: Data Generation (BLOCKING - highest priority)
    cvs, G = phase1_data_generation()
    
    # Phase 2: CV Community Analysis (internal only for now)
    cvs = phase2_community_analysis(cvs)
    
    print("\n" + "="*80)
    print("⏸️  COORDINATION POINT")
    print("="*80)
    print("Waiting for Chiheb to:")
    print("1. Add Job nodes to the graph")
    print("2. Generate CV embeddings")
    print("\nOnce Chiheb completes his tasks, continue with:")
    print("- Phase 2B: Global community detection")
    print("- Phase 3-6: Metrics, classification, analysis, documentation")
    print("="*80)
    
    # The following phases would run after Chiheb's handoff
    # For now, we'll demonstrate the structure with mock data
    
    print("\n📝 Note: Remaining phases (2B-6) will execute after coordination with Chiheb")
    print("    To run them, ensure:")
    print("    1. Graph has Job nodes (cv_job_graph.pkl updated by Chiheb)")
    print("    2. CV embeddings available (cv_embeddings.npy from Chiheb)")
    
    # Create README for handoff
    readme = """# RAMY'S DELIVERABLES - HANDOFF POINT

## ✅ Completed Tasks (Phase 1-2)

### Data Files Created:
- `data/cvs.json` - 100 CVs with all attributes
- `data/cvs_with_graphs.json` - CVs with internal skill graphs
- `data/cv_job_graph.pkl` - Initial bipartite graph structure (CV nodes only)
- `data/graph_statistics.json` - Initial graph statistics

### Results Generated:
- `results/metrics/cv_skill_communities.csv` - CV skill community metrics
- `results/figures/skill_graph_examples.png` - Example skill graphs

## 📤 Ready for Chiheb:

1. **Graph File**: `data/cv_job_graph.pkl`
   - Contains 100 CV nodes with all attributes
   - Ready for Job node addition
   - Need ~175 edges total after Job integration

2. **CV Data**: `data/cvs.json` or `data/cvs_with_graphs.json`
   - Use for generating CV embeddings
   - All CV attributes available

## ⏳ Waiting for from Chiheb:

1. Updated `data/cv_job_graph.pkl` with:
   - 80 Job nodes added
   - ~175 initial CV-Job edges

2. `data/cv_embeddings.npy`
   - Embeddings for all 100 CVs
   - Needed for Phase 4 classification

## 🔄 Next Steps After Handoff:

Once Chiheb provides the above:
1. Run Phase 2B: Global community detection
2. Continue with Phases 3-6

---
Generated by Ramy's Pipeline
"""
    
    with open('RAMY_HANDOFF_README.md', 'w', encoding='utf-8') as f:
        f.write(readme)
    
    print("\n✅ Created RAMY_HANDOFF_README.md for coordination")


if __name__ == "__main__":
    main()