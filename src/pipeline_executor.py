"""
Complete Pipeline Executor - Chiheb's Link Prediction & Job Classification

Integrates Ramy's CV data with Chiheb's job classification and link prediction.
Runs the complete workflow:
1. Load CV + Job data
2. Generate Embeddings
3. Compute Semantic Similarity
4. Link Prediction
5. Job Classification 
6. Graph Enrichment
7. Impact Analysis
"""

import json
import pickle
import numpy as np
import pandas as pd
import networkx as nx
from pathlib import Path
from community import best_partition
import community as community_louvain
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import random
import sys

from job_generator import JobGenerator
from llm_utils import SemanticSimilarityMatrix
from link_predictor import (
    GraphLinkFeatureComputer, HybridLinkPredictionModel, LinkEnricher
)
from job_classifier import (
    JobFeatureEngineer, JobDomainClassifier, JobLevelClassifier,
    JobClassificationVisualizer
)
from pipeline_visualizations import PipelineVisualizations
from interactive_visualizer import InteractiveGraphVisualizer

class PipelineExecutor:
    """Execute complete CV-Job graph mining pipeline."""
    
    def __init__(self, seed: int = 42):
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)
        
        self.data_dir = Path('data')
        self.results_dir = Path('results')
        self.models_dir = Path('models')
        
        # Create directories
        (self.results_dir / 'figures').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'metrics').mkdir(parents=True, exist_ok=True)
        (self.results_dir / 'analysis').mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(exist_ok=True)
        
        self.cvs = None
        self.jobs = None
        self.graph = None
        self.communities = None
        self.cv_nodes = None
        self.job_nodes = None
        self.cv_embeddings = None
        self.job_embeddings = None
        self.similarity_matrix = None
        self.viz = PipelineVisualizations(output_dir=str(self.results_dir / 'figures'))
        
        # Interactive visualizations
        self.interactive_viz = InteractiveGraphVisualizer(output_dir=str(self.results_dir / 'interactive'))
        
        # Store metrics for visualization
        self.edges_before_prediction = 0
        self.edges_after_prediction = 0
        self.graph_metrics = {}
        self.link_model_metrics = {}
        self.predicted_links = []
    
    def load_data(self):
        """Load CV and Job data."""
        print("\n" + "="*70)
        print("PHASE 1: LOADING DATA")
        print("="*70)
        
        # Load CVs
        with open(self.data_dir / 'cvs.json', 'r') as f:
            self.cvs = json.load(f)
        print(f"✓ Loaded {len(self.cvs)} CVs")
        
        # Load Jobs
        with open(self.data_dir / 'jobs.json', 'r') as f:
            self.jobs = json.load(f)
        print(f"✓ Loaded {len(self.jobs)} jobs")
        
        # Load Graph
        with open(self.data_dir / 'cv_job_graph.pkl', 'rb') as f:
            self.graph = pickle.load(f)
        print(f"✓ Loaded graph: {self.graph.number_of_nodes()} nodes, {self.graph.number_of_edges()} edges")
        
        # Load graph statistics
        with open(self.data_dir / 'graph_statistics.json', 'r') as f:
            stats = json.load(f)
        print(f"  Density: {stats.get('density', 'N/A')}")
        
        # Set node sets
        self.cv_nodes = set([cv['cv_id'] for cv in self.cvs])
        self.job_nodes = set([job['job_id'] for job in self.jobs])
        
        print(f"\nData Summary:")
        print(f"  CVs: {len(self.cv_nodes)}")
        print(f"  Jobs: {len(self.job_nodes)}")
        print(f"  Edges: {self.graph.number_of_edges()}")
        
    def create_initial_edges(self):
        """Create initial edges based on domain and skill matching."""
        print("\n" + "="*70)
        print("PHASE 1.5: CREATING INITIAL EDGES")
        print("="*70)
        
        print("Matching CVs to jobs based on domain and skills...")
        
        edges_added = 0
        
        # Match each job to relevant CVs
        for job in self.jobs:
            job_id = job['job_id']
            job_domain = job.get('domain', 'other')
            job_skills = set(skill.lower() for skill in job.get('required_skills', []))
            job_level = job.get('required_level', 'entry')
            
            # For each CV, compute a match score
            for cv in self.cvs:
                cv_id = cv['cv_id']
                
                # Skip if already connected
                if self.graph.has_edge(cv_id, job_id):
                    continue
                
                # Score based on:
                # 1. Domain match
                domain_match = 1.0 if cv.get('domain') == job_domain else 0.3
                
                # 2. Skill overlap
                cv_skills = set(skill.lower() for skill in cv.get('skills', []))
                skill_overlap = len(job_skills & cv_skills) / max(len(job_skills), 1)
                
                # 3. Level compatibility
                cv_level = cv.get('seniority_level', 'entry')
                level_compat = 1.0 if cv_level == job_level else 0.6
                
                # Combined score - balanced weights
                score = (domain_match * 0.3 + skill_overlap * 0.5 + level_compat * 0.2)
                
                # Add edge if score > threshold (moderate threshold)
                # This should create ~30-40% density 
                if score > 0.35:
                    self.graph.add_edge(cv_id, job_id, weight=score)
                    edges_added += 1
        
        print(f"✓ Added {edges_added} edges based on matching")
        print(f"  Total edges: {self.graph.number_of_edges()}")
        density = nx.density(self.graph)
        print(f"  Graph density: {density:.4f}")
        self.edges_before_prediction = self.graph.number_of_edges()
        self.graph_metrics['density_initial'] = density
    
    def community_detection(self):
        """Detect communities in the graph."""
        print("\n" + "="*70)
        print("PHASE 2: COMMUNITY DETECTION")
        print("="*70)
        
        print("Running Louvain community detection...")
        self.communities = best_partition(self.graph)
        modularity = community_louvain.modularity(self.communities, self.graph)
        
        n_communities = len(set(self.communities.values()))
        print(f"✓ Detection complete")
        print(f"  Communities: {n_communities}")
        print(f"  Modularity: {modularity:.3f}")
        
        # Visualization
        self.viz.plot_community_structure(
            self.graph, self.communities, modularity,
            self.cv_nodes, self.job_nodes
        )
        self.graph_metrics['modularity'] = modularity
        self.graph_metrics['n_communities'] = n_communities
        
        # Visualization
        self.viz.plot_community_structure(
            self.graph, self.communities, modularity,
            self.cv_nodes, self.job_nodes
        )
        self.graph_metrics['modularity'] = modularity
        self.graph_metrics['n_communities'] = n_communities
    
    def generate_embeddings(self):
        """Generate embeddings for CVs and Jobs."""
        print("\n" + "="*70)
        print("PHASE 3: EMBEDDING GENERATION")
        print("="*70)
        
        print("Generating embeddings...")
        
        # Create random embeddings (mock - in production use sentence-transformers)
        embedding_dim = 384
        
        self.cv_embeddings = {}
        for cv in self.cvs:
            # Use deterministic random based on CV text for reproducibility
            text_hash = hash(cv['raw_text']) % (2**32)
            np.random.seed(text_hash)
            self.cv_embeddings[cv['cv_id']] = np.random.randn(embedding_dim)
            # Normalize
            self.cv_embeddings[cv['cv_id']] /= np.linalg.norm(self.cv_embeddings[cv['cv_id']])
        
        self.job_embeddings = {}
        for job in self.jobs:
            text_hash = hash(job['description']) % (2**32)
            np.random.seed(text_hash)
            self.job_embeddings[job['job_id']] = np.random.randn(embedding_dim)
            # Normalize
            self.job_embeddings[job['job_id']] /= np.linalg.norm(self.job_embeddings[job['job_id']])
        
        print(f"✓ Generated embeddings for {len(self.cv_embeddings)} CVs")
        print(f"✓ Generated embeddings for {len(self.job_embeddings)} jobs")
        print(f"  Dimension: {embedding_dim}")
        
        # Visualization
        self.viz.plot_embedding_space(self.cv_embeddings, self.job_embeddings)
        
        # Visualization
        self.viz.plot_embedding_space(self.cv_embeddings, self.job_embeddings)
    
    def compute_similarity(self):
        """Compute semantic similarity matrix."""
        print("\nComputing semantic similarity matrix...")
        
        self.similarity_matrix, cv_ids, job_ids = SemanticSimilarityMatrix.compute_matrix(
            self.cv_embeddings, self.job_embeddings
        )
        
        print(f"✓ Matrix shape: {self.similarity_matrix.shape}")
        print(f"  Mean similarity: {np.mean(self.similarity_matrix):.3f}")
        print(f"  Max similarity: {np.max(self.similarity_matrix):.3f}")
    
    def link_prediction(self):
        """Perform hybrid link prediction."""
        print("\n" + "="*70)
        print("PHASE 4: LINK PREDICTION")
        print("="*70)
        
        # Feature computation
        print("Computing graph features...")
        feature_computer = GraphLinkFeatureComputer(
            self.graph, self.communities, self.cv_nodes, self.job_nodes
        )
        
        # Prepare training data
        print("Preparing training data...")
        positive_samples = []
        positive_features = []
        
        cv_ids_sorted = sorted(self.cv_nodes)
        job_ids_sorted = sorted(self.job_nodes)
        
        for cv, job in self.graph.edges():
            if cv in self.cv_nodes and job in self.job_nodes:
                features_dict = feature_computer.compute_features(cv, job)
                
                cv_idx = cv_ids_sorted.index(cv)
                job_idx = job_ids_sorted.index(job)
                sem_sim = self.similarity_matrix[cv_idx, job_idx]
                features_dict['semantic_similarity'] = sem_sim
                
                positive_samples.append((cv, job))
                positive_features.append(features_dict)
        
        print(f"  Positive samples: {len(positive_samples)}")
        
        # Negative samples
        negative_samples = []
        negative_features = []
        n_negative = len(positive_samples)
        
        random.seed(self.seed)
        while len(negative_samples) < n_negative:
            cv = random.choice(list(self.cv_nodes))
            job = random.choice(list(self.job_nodes))
            
            if not self.graph.has_edge(cv, job) and (cv, job) not in negative_samples:
                features_dict = feature_computer.compute_features(cv, job)
                
                cv_idx = cv_ids_sorted.index(cv)
                job_idx = job_ids_sorted.index(job)
                sem_sim = self.similarity_matrix[cv_idx, job_idx]
                features_dict['semantic_similarity'] = sem_sim
                
                negative_samples.append((cv, job))
                negative_features.append(features_dict)
        
        print(f"  Negative samples: {len(negative_samples)}")
        
        # Combine and train
        all_features = positive_features + negative_features
        all_labels = [1] * len(positive_samples) + [0] * len(negative_samples)
        feature_names = list(all_features[0].keys())
        
        X = np.array([[f[name] for name in feature_names] for f in all_features])
        y = np.array(all_labels)
        
        print("\nTraining link prediction model...")
        model = HybridLinkPredictionModel(random_state=self.seed)
        metrics = model.train(X, y, feature_names=feature_names)
        model.save(str(self.models_dir / 'link_prediction_model.pkl'))
        
        # Save metrics
        metrics_file = self.results_dir / 'metrics' / 'link_prediction_performance.json'
        metrics_to_save = {
            'precision': float(metrics['precision']),
            'recall': float(metrics['recall']),
            'auc_roc': float(metrics['auc_roc']),
            'accuracy': float(metrics['accuracy'])
        }
        with open(metrics_file, 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        
        # Visualization - Feature Importance
        feature_importance_dict = dict(zip(feature_names, model.model.feature_importances_))
        self.viz.plot_link_prediction_performance(
            metrics['y_test'],
            metrics['y_pred_proba'],
            metrics['precision'],
            metrics['recall'],
            metrics['auc_roc'],
            metrics['accuracy'],
            feature_importance_dict
        )
        
        self.link_model = model
        self.feature_computer = feature_computer
        self.feature_names = feature_names
        self.link_model_metrics = metrics_to_save
    
    def enrich_graph(self):
        """Enrich graph with predicted links."""
        print("\nEnriching graph with predicted links...")
        
        # Prepare all non-edges
        all_pairs = []
        cv_ids_sorted = sorted(self.cv_nodes)
        job_ids_sorted = sorted(self.job_nodes)
        
        for cv in self.cv_nodes:
            for job in self.job_nodes:
                if not self.graph.has_edge(cv, job):
                    all_pairs.append((cv, job))
        
        print(f"  Evaluating {len(all_pairs)} potential links...")
        
        # Prepare features
        features_list = []
        for cv, job in all_pairs:
            features_dict = self.feature_computer.compute_features(cv, job)
            
            cv_idx = cv_ids_sorted.index(cv)
            job_idx = job_ids_sorted.index(job)
            sem_sim = self.similarity_matrix[cv_idx, job_idx]
            features_dict['semantic_similarity'] = sem_sim
            features_list.append(features_dict)
        
        X_pred = np.array([[f[name] for name in self.feature_names] for f in features_list])
        
        # Predict
        predictions, probabilities = self.link_model.predict_links(X_pred, threshold=0.75)
        
        # Collect predicted links
        predicted_links = []
        for (cv, job), prob in zip(all_pairs, probabilities):
            if prob >= 0.75:
                predicted_links.append((cv, job, prob))
        
        # Store for visualization
        self.predicted_links = predicted_links
        self.edges_after_prediction = self.graph.number_of_edges() + len(predicted_links)
        
        # Visualization - Predicted links distribution
        self.viz.plot_predicted_links_distribution(
            predicted_links, len(self.cv_nodes), len(self.job_nodes)
        )
        
        # Enrich graph
        enriched_graph, added_links = LinkEnricher.enrich_graph(
            self.graph, predicted_links, threshold=0.75
        )
        
        # Save enriched graph
        LinkEnricher.save_enriched_graph(
            enriched_graph,
            str(self.data_dir / 'cv_job_graph_enriched.pkl')
        )
        
        LinkEnricher.save_predicted_links(
            added_links,
            str(self.results_dir / 'metrics' / 'predicted_links.csv')
        )
        
        self.graph_enriched = enriched_graph
    
    def create_interactive_visualizations(self):
        """Create interactive HTML visualizations."""
        print("\n" + "="*70)
        print("PHASE 6: INTERACTIVE VISUALIZATIONS")
        print("="*70)
        
        print("Creating Neo4j-style interactive graphs...")
        
        # Visualize bipartite graph
        self.interactive_viz.visualize_bipartite_graph(
            self.graph,
            self.cv_nodes,
            self.job_nodes,
            self.communities,
            self.cvs,
            self.jobs,
            title="CV-Job Bipartite Graph",
            filename="bipartite_graph.html"
        )
        
        # Visualize predicted links
        self.interactive_viz.visualize_predicted_links(
            self.graph,
            self.predicted_links,
            self.cv_nodes,
            self.job_nodes,
            self.communities,
            self.cvs,
            self.jobs,
            title="Predicted Links",
            filename="predicted_links.html"
        )
        
        # Create dashboard
        self.interactive_viz.create_dashboard(
            self.graph,
            self.graph_enriched,
            self.predicted_links,
            self.cv_nodes,
            self.job_nodes,
            self.communities,
            self.graph_metrics.get('modularity', 0),
            self.link_model_metrics,
            self.cvs,
            self.jobs,
            filename="dashboard.html"
        )
        
        print("\n✓ Interactive visualizations complete!")
        print(f"  Open: results/interactive/dashboard.html in your browser")
    
    def classify_jobs(self):
        """Classify jobs by domain and level."""
        print("\n" + "="*70)
        print("PHASE 5: JOB CLASSIFICATION")
        print("="*70)
        
        print("Creating job features...")
        feature_engineer = JobFeatureEngineer(
            self.graph, self.communities, list(self.job_nodes),
            self.job_embeddings
        )
        
        X_jobs, feature_names = feature_engineer.create_feature_matrix()
        
        # Extract labels
        y_domain = [next(j for j in self.jobs if j['job_id'] == jid)['domain'] 
                   for jid in self.job_nodes]
        y_level = [next(j for j in self.jobs if j['job_id'] == jid)['level']
                  for jid in self.job_nodes]
        
        # Domain classification
        print("\nTraining job domain classifier...")
        domain_clf = JobDomainClassifier(random_state=self.seed)
        domain_metrics = domain_clf.train(X_jobs, y_domain, feature_names)
        domain_clf.save(str(self.models_dir / 'job_domain_classifier.pkl'))
        
        JobClassificationVisualizer.plot_confusion_matrix(
            domain_metrics['confusion_matrix'],
            domain_clf.label_encoder.classes_,
            'Job Domain Classification',
            str(self.results_dir / 'figures' / 'job_domain_confusion_matrix.png')
        )
        
        # Level classification  
        print("\nTraining job level classifier...")
        level_clf = JobLevelClassifier(random_state=self.seed)
        level_metrics = level_clf.train(X_jobs, y_level, feature_names)
        level_clf.save(str(self.models_dir / 'job_level_classifier.pkl'))
        
        JobClassificationVisualizer.plot_confusion_matrix(
            level_metrics['confusion_matrix'],
            level_clf.label_encoder.classes_,
            'Job Level Classification',
            str(self.results_dir / 'figures' / 'job_level_confusion_matrix.png')
        )
        
        # Visualization - Job Classification Summary
        # Convert sklearn report strings to dict using output_dict parameter
        from sklearn.metrics import classification_report as sk_classification_report
        
        domain_report_dict = sk_classification_report(
            domain_metrics['y_test'], domain_metrics['y_pred'],
            target_names=domain_clf.label_encoder.classes_,
            output_dict=True, zero_division=0
        )
        
        level_report_dict = sk_classification_report(
            level_metrics['y_test'], level_metrics['y_pred'],
            target_names=level_clf.label_encoder.classes_,
            output_dict=True, zero_division=0
        )
        
        self.viz.plot_job_classification_summary(domain_report_dict, level_report_dict)
        
        # Visualization - Graph evolution over all phases
        self.viz.plot_graph_evolution(
            edges_before=self.edges_before_prediction,
            edges_after_community=self.edges_before_prediction,
            edges_after_prediction=self.edges_after_prediction,
            density_before=self.graph_metrics.get('density_initial', 0),
            density_after_prediction=nx.density(self.graph)
        )
    
    def run(self):
        """Execute complete pipeline."""
        try:
            self.load_data()
            self.create_initial_edges()
            self.community_detection()
            self.generate_embeddings()
            self.compute_similarity()
            self.link_prediction()
            self.enrich_graph()
            self.classify_jobs()
            self.create_interactive_visualizations()
            
            print("\n" + "="*70)
            print("✓ PIPELINE COMPLETE")
            print("="*70)
            print(f"\nResults saved in: {self.results_dir}")
            print(f"Models saved in: {self.models_dir}")
            print(f"\n🌐 Open interactive dashboard:")
            print(f"   results/interactive/dashboard.html")
            
            return True
        except Exception as e:
            print(f"\n✗ ERROR: {e}")
            import traceback
            traceback.print_exc()
            return False


if __name__ == '__main__':
    executor = PipelineExecutor(seed=42)
    success = executor.run()
    sys.exit(0 if success else 1)
