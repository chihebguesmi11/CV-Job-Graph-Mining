# Claude Copilot Prompt: CV-Job Graph Mining Project
## Automated Implementation with LLM Integration and Graph Algorithms

---

## 🎯 Project Overview

You are tasked with implementing a complete **CV-Job bipartite graph mining system** that combines:
- **Traditional graph algorithms** (community detection, link prediction, node classification)
- **Modern LLM capabilities** (semantic embeddings via Groq API with Llama models)
- **Hybrid approaches** that leverage both structural and semantic information

**Goal**: Build a system that analyzes recruitment patterns, predicts CV-Job matches, and classifies profiles using graph mining techniques from the course combined with LLM semantic understanding.

---

## 📚 Course Material to Apply

### 1. Community Detection Algorithms

Implement these algorithms from the course:

**A. Louvain Algorithm** (Primary - Fast & Effective)
- Use for global CV-Job community detection
- Greedy modularity optimization
- Expected modularity Q between 0.3-0.7 for real networks

**B. Label Propagation** (Alternative - Very Fast)
- For large-scale analysis
- Complexity: O(m) - linear in edges
- Non-deterministic but extremely fast

**C. Modularity Metric**
```python
Q = (1/2m) Σ[A_ij - (k_i × k_j)/(2m)] × δ(c_i, c_j)
```
- Use to evaluate community quality
- Target: Q > 0.3 for significant structure

### 2. Link Prediction Methods

Implement both classical and LLM-based approaches:

**A. Classical Graph-Based Similarity Scores**
```python
# Common Neighbors
score_cn(u,v) = |N(u) ∩ N(v)|

# Adamic-Adar
score_aa(u,v) = Σ 1/log(deg(w)) for w ∈ common_neighbors(u,v)

# Jaccard Coefficient  
score_jc(u,v) = |N(u) ∩ N(v)| / |N(u) ∪ N(v)|

# Preferential Attachment
score_pa(u,v) = deg(u) × deg(v)
```

**B. LLM-Based Semantic Scoring**
- Use Groq API with Llama models (llama-3.1-70b-versatile recommended)
- Generate embeddings for CV and Job texts
- Compute cosine similarity for matching scores
- Set threshold (0.7-0.8) for link prediction

**C. Hybrid Approach**
- Combine graph features + LLM embeddings
- Train supervised classifier (Random Forest)
- Feature vector: [CN, AA, JC, PA, embedding_similarity, community_features]

### 3. Node Classification Techniques

**A. Iterative Classification (REV2 from course)**
```
1. Train classifier on node attributes
2. Train classifier on attributes + neighbor labels
3. Iterate:
   - Update relational features (neighbor label counts)
   - Classify with combined features
   - Repeat until convergence
```

**B. Feature Engineering**
- **Structural**: degree, betweenness, PageRank, community_id
- **Semantic**: LLM embeddings
- **Relational**: neighbor label distributions
- **Internal**: skill community features (from CV internal graphs)

### 4. Graph Analysis Metrics

**Centrality Measures**:
- Degree centrality: Direct connections
- Betweenness centrality: Bridge nodes
- PageRank: Importance score

**Community Metrics**:
- Modularity Q: Quality of partition
- Conductance φ: Cut ratio
- Density: Internal connectivity

---

## 🏗️ Implementation Architecture

### Phase 1: Data Generation & Graph Construction

**Task 1.1: Generate Synthetic CV-Job Dataset**
```python
# Generate 100 CVs with attributes:
CVs = {
    'cv_id': unique_id,
    'skills': [Python, ML, Data Analysis, ...],  # 5-10 skills
    'experience_years': 1-15,
    'education': ['BS CS', 'MS AI', ...],
    'seniority': ['junior', 'mid', 'senior'],
    'domain': ['tech', 'business', 'research'],
    'raw_text': "Full CV description with experience and skills..."
}

# Generate 80 Jobs with attributes:
Jobs = {
    'job_id': unique_id,
    'required_skills': [Python, Django, AWS, ...],  # 4-8 skills
    'experience_required': 2-10,
    'level': ['entry', 'mid', 'senior', 'lead'],
    'domain': ['engineering', 'data_science', 'management'],
    'description': "Full job description with requirements..."
}

# Create 150-200 existing edges with weights
edges = [(cv_id, job_id, weight)]  # weight in [0.5, 1.0]
```

**Task 1.2: Build Bipartite Graph**
```python
import networkx as nx

G = nx.Graph()
G.add_nodes_from(cv_nodes, bipartite=0, **cv_attributes)
G.add_nodes_from(job_nodes, bipartite=1, **job_attributes)
G.add_weighted_edges_from(edges)

# Save graph object for reuse
nx.write_gpickle(G, 'data/cv_job_graph.pkl')
```

**Task 1.3: Create Internal Skill Graphs**
```python
# For each CV, create skill co-occurrence graph
for cv in cvs:
    skills = cv['skills']
    experiences = cv['experiences']
    
    # Create graph where skills are nodes
    # Edges: skills mentioned in same experience/project
    G_skills = nx.Graph()
    G_skills.add_nodes_from(skills)
    
    # Add edges for skills co-occurring in experiences
    for exp in experiences:
        exp_skills = [s for s in skills if s in exp]
        for i, s1 in enumerate(exp_skills):
            for s2 in exp_skills[i+1:]:
                G_skills.add_edge(s1, s2)
    
    cv['internal_skill_graph'] = G_skills
```

---

### Phase 2: Graph Analysis & Visualization

**Task 2.1: Compute Structural Metrics**
```python
import pandas as pd

# Degree distribution
cv_degrees = {cv: G.degree(cv) for cv in cv_nodes}
job_degrees = {job: G.degree(job) for job in job_nodes}

# Centrality measures
betweenness = nx.betweenness_centrality(G)
pagerank = nx.pagerank(G)

# Network properties
density = nx.density(G)
avg_clustering = nx.average_clustering(G)

# Create analysis DataFrame
metrics_df = pd.DataFrame({
    'node': list(G.nodes()),
    'type': ['CV' if n in cv_nodes else 'Job' for n in G.nodes()],
    'degree': [G.degree(n) for n in G.nodes()],
    'betweenness': [betweenness[n] for n in G.nodes()],
    'pagerank': [pagerank[n] for n in G.nodes()]
})

metrics_df.to_csv('results/metrics/node_metrics.csv')
```

**Task 2.2: Visualization**
```python
import matplotlib.pyplot as plt
import seaborn as sns

# Plot degree distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(list(cv_degrees.values()), bins=20)
axes[0].set_title('CV Degree Distribution')
axes[0].set_xlabel('Degree')

axes[1].hist(list(job_degrees.values()), bins=20)
axes[1].set_title('Job Degree Distribution')
axes[1].set_xlabel('Degree')

plt.savefig('results/figures/degree_distributions.png')

# Bipartite layout visualization
from networkx.algorithms import bipartite
pos = nx.bipartite_layout(G, cv_nodes)

plt.figure(figsize=(15, 10))
nx.draw_networkx_nodes(G, pos, nodelist=cv_nodes, 
                       node_color='lightblue', node_size=300, label='CVs')
nx.draw_networkx_nodes(G, pos, nodelist=job_nodes,
                       node_color='lightcoral', node_size=300, label='Jobs')
nx.draw_networkx_edges(G, pos, alpha=0.3)
plt.legend()
plt.title('CV-Job Bipartite Graph')
plt.savefig('results/figures/bipartite_graph.png', dpi=300, bbox_inches='tight')
```

---

### Phase 3: Community Detection

**Task 3.1: Global Community Detection**
```python
from community import best_partition
import community as community_louvain

# Louvain algorithm for community detection
communities = best_partition(G)

# Compute modularity
modularity = community_louvain.modularity(communities, G)
print(f"Modularity Q: {modularity:.3f}")

# Expected: 0.3 < Q < 0.7 for significant structure

# Analyze community composition
community_stats = {}
for comm_id in set(communities.values()):
    nodes_in_comm = [n for n in G.nodes() if communities[n] == comm_id]
    cvs_in_comm = [n for n in nodes_in_comm if n in cv_nodes]
    jobs_in_comm = [n for n in nodes_in_comm if n in job_nodes]
    
    community_stats[comm_id] = {
        'size': len(nodes_in_comm),
        'n_cvs': len(cvs_in_comm),
        'n_jobs': len(jobs_in_comm),
        'density': nx.density(G.subgraph(nodes_in_comm))
    }

# Save community assignments
pd.DataFrame([
    {'node': n, 'community': communities[n], 
     'type': 'CV' if n in cv_nodes else 'Job'}
    for n in G.nodes()
]).to_csv('results/metrics/community_assignments.csv')
```

**Task 3.2: Internal CV Community Detection**
```python
# For each CV, detect skill communities
for cv in cvs:
    G_skills = cv['internal_skill_graph']
    
    if len(G_skills.nodes()) > 2:  # Need at least 3 nodes
        skill_communities = best_partition(G_skills)
        skill_modularity = community_louvain.modularity(skill_communities, G_skills)
        
        cv['skill_communities'] = skill_communities
        cv['skill_modularity'] = skill_modularity
        cv['n_skill_communities'] = len(set(skill_communities.values()))
        
        # Specialization score: higher modularity = more specialized
        cv['specialization_score'] = skill_modularity
```

**Task 3.3: Community Visualization**
```python
# Color nodes by community
node_colors = [communities[n] for n in G.nodes()]

plt.figure(figsize=(15, 10))
nx.draw_networkx(G, pos, node_color=node_colors, cmap=plt.cm.Set3,
                 node_size=200, with_labels=False, edge_color='gray', alpha=0.6)
plt.title(f'Community Detection (Q={modularity:.3f})')
plt.savefig('results/figures/communities.png', dpi=300, bbox_inches='tight')
```

---

### Phase 4: Link Prediction with LLM Integration

**Task 4.1: Setup Groq API for Llama Embeddings**
```python
from groq import Groq
import numpy as np
import os

# Initialize Groq client
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def get_llm_embedding(text, model="llama-3.1-70b-versatile"):
    """
    Get embeddings using Groq API with Llama model.
    Note: Since Groq doesn't have direct embedding API, 
    we'll use a workaround with sentence similarity.
    """
    # Alternative: Use sentence-transformers for embeddings
    from sentence_transformers import SentenceTransformer
    
    # Use local model for embeddings (more reliable)
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    embedding = embedding_model.encode(text, convert_to_numpy=True)
    
    return embedding

def get_llm_similarity_score(cv_text, job_text):
    """
    Use Groq Llama model to score CV-Job match.
    """
    prompt = f"""You are an expert recruiter. Rate how well this CV matches this Job on a scale of 0-10.

CV Summary:
{cv_text[:500]}

Job Description:
{job_text[:500]}

Consider:
- Skill overlap
- Experience level match
- Domain alignment

Respond with ONLY a number between 0-10, nothing else."""

    response = client.chat.completions.create(
        model="llama-3.1-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=10
    )
    
    try:
        score = float(response.choices[0].message.content.strip())
        return score / 10.0  # Normalize to [0, 1]
    except:
        return 0.5  # Default if parsing fails

# Generate embeddings for all CVs and Jobs
cv_embeddings = {}
job_embeddings = {}

print("Generating embeddings...")
for cv in cvs:
    cv_embeddings[cv['cv_id']] = get_llm_embedding(cv['raw_text'])

for job in jobs:
    job_embeddings[job['job_id']] = get_llm_embedding(job['description'])
```

**Task 4.2: Classical Link Prediction Features**
```python
def compute_graph_features(G, cv_node, job_node):
    """
    Compute graph-based link prediction features.
    """
    # Common neighbors (in bipartite graph, this needs adaptation)
    cv_neighbors = set(G.neighbors(cv_node))
    job_neighbors = set(G.neighbors(job_node))
    
    # In bipartite graph, common neighbors would be indirect
    # Calculate via 2-hop paths
    common_neighbors = len(cv_neighbors & job_neighbors)
    
    # Adamic-Adar
    adamic_adar = 0
    for common in (cv_neighbors & job_neighbors):
        if G.degree(common) > 0:
            adamic_adar += 1 / np.log(G.degree(common))
    
    # Jaccard coefficient
    union_neighbors = len(cv_neighbors | job_neighbors)
    jaccard = common_neighbors / union_neighbors if union_neighbors > 0 else 0
    
    # Preferential attachment
    preferential_attachment = G.degree(cv_node) * G.degree(job_node)
    
    # Community feature
    same_community = 1 if communities[cv_node] == communities[job_node] else 0
    
    return {
        'common_neighbors': common_neighbors,
        'adamic_adar': adamic_adar,
        'jaccard': jaccard,
        'preferential_attachment': preferential_attachment,
        'same_community': same_community,
        'cv_degree': G.degree(cv_node),
        'job_degree': G.degree(job_node)
    }
```

**Task 4.3: Hybrid Link Prediction Model**
```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

# Prepare training data
print("Preparing link prediction dataset...")

# Positive samples: existing edges
positive_samples = []
for cv, job in G.edges():
    if cv in cv_nodes and job in job_nodes:
        # Graph features
        graph_feats = compute_graph_features(G, cv, job)
        
        # Semantic similarity (cosine of embeddings)
        sem_similarity = cosine_similarity(
            cv_embeddings[cv].reshape(1, -1),
            job_embeddings[job].reshape(1, -1)
        )[0, 0]
        
        # Combine features
        features = [
            graph_feats['common_neighbors'],
            graph_feats['adamic_adar'],
            graph_feats['jaccard'],
            graph_feats['preferential_attachment'],
            graph_feats['same_community'],
            graph_feats['cv_degree'],
            graph_feats['job_degree'],
            sem_similarity
        ]
        
        positive_samples.append((features, 1))

# Negative samples: non-edges (sample equal number)
import random
negative_samples = []
n_negative = len(positive_samples)

while len(negative_samples) < n_negative:
    cv = random.choice(cv_nodes)
    job = random.choice(job_nodes)
    
    if not G.has_edge(cv, job):
        graph_feats = compute_graph_features(G, cv, job)
        sem_similarity = cosine_similarity(
            cv_embeddings[cv].reshape(1, -1),
            job_embeddings[job].reshape(1, -1)
        )[0, 0]
        
        features = [
            graph_feats['common_neighbors'],
            graph_feats['adamic_adar'],
            graph_feats['jaccard'],
            graph_feats['preferential_attachment'],
            graph_feats['same_community'],
            graph_feats['cv_degree'],
            graph_feats['job_degree'],
            sem_similarity
        ]
        
        negative_samples.append((features, 0))

# Combine and split
all_samples = positive_samples + negative_samples
random.shuffle(all_samples)

X = np.array([s[0] for s in all_samples])
y = np.array([s[1] for s in all_samples])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Train Random Forest classifier
print("Training hybrid link prediction model...")
clf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    random_state=42
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_pred_proba)

print(f"Link Prediction Performance:")
print(f"  Precision: {precision:.3f}")
print(f"  Recall: {recall:.3f}")
print(f"  AUC-ROC: {auc:.3f}")

# Feature importance
feature_names = ['common_neighbors', 'adamic_adar', 'jaccard', 
                'preferential_attachment', 'same_community',
                'cv_degree', 'job_degree', 'semantic_similarity']
importances = clf.feature_importances_

for name, importance in sorted(zip(feature_names, importances), 
                               key=lambda x: x[1], reverse=True):
    print(f"  {name}: {importance:.3f}")
```

**Task 4.4: Graph Enrichment with Predicted Links**
```python
# Predict on all non-edges
print("Predicting missing links...")
predicted_links = []

for cv in cv_nodes:
    for job in job_nodes:
        if not G.has_edge(cv, job):
            graph_feats = compute_graph_features(G, cv, job)
            sem_similarity = cosine_similarity(
                cv_embeddings[cv].reshape(1, -1),
                job_embeddings[job].reshape(1, -1)
            )[0, 0]
            
            features = [
                graph_feats['common_neighbors'],
                graph_feats['adamic_adar'],
                graph_feats['jaccard'],
                graph_feats['preferential_attachment'],
                graph_feats['same_community'],
                graph_feats['cv_degree'],
                graph_feats['job_degree'],
                sem_similarity
            ]
            
            prob = clf.predict_proba([features])[0, 1]
            
            if prob > 0.75:  # High confidence threshold
                predicted_links.append((cv, job, prob))

# Add predicted links to enriched graph
G_enriched = G.copy()
for cv, job, prob in predicted_links:
    G_enriched.add_edge(cv, job, weight=prob, predicted=True)

print(f"Added {len(predicted_links)} predicted links to graph")

# Re-detect communities on enriched graph
communities_enriched = best_partition(G_enriched)
modularity_enriched = community_louvain.modularity(communities_enriched, G_enriched)

print(f"Original modularity: {modularity:.3f}")
print(f"Enriched modularity: {modularity_enriched:.3f}")
```

---

### Phase 5: Node Classification

**Task 5.1: Feature Engineering for Classification**
```python
def create_node_features(node, G, communities, embeddings):
    """
    Create comprehensive feature vector for node classification.
    """
    features = {}
    
    # Structural features
    features['degree'] = G.degree(node)
    features['betweenness'] = betweenness[node]
    features['pagerank'] = pagerank[node]
    features['clustering'] = nx.clustering(G, node)
    
    # Community features
    features['community_id'] = communities[node]
    comm_nodes = [n for n in G.nodes() if communities[n] == communities[node]]
    features['community_size'] = len(comm_nodes)
    
    # Neighborhood features
    neighbors = list(G.neighbors(node))
    if neighbors:
        features['avg_neighbor_degree'] = np.mean([G.degree(n) for n in neighbors])
    else:
        features['avg_neighbor_degree'] = 0
    
    # Embedding features (semantic)
    embedding = embeddings.get(node, np.zeros(384))  # 384-dim for MiniLM
    
    # For CV nodes, add internal skill graph features
    if node in cv_nodes:
        cv_data = next(cv for cv in cvs if cv['cv_id'] == node)
        features['n_skills'] = len(cv_data['skills'])
        features['experience_years'] = cv_data['experience_years']
        features['n_skill_communities'] = cv_data.get('n_skill_communities', 0)
        features['specialization_score'] = cv_data.get('specialization_score', 0)
    
    # Combine structural and semantic features
    structural_feats = np.array([
        features['degree'],
        features['betweenness'],
        features['pagerank'],
        features['clustering'],
        features['community_size'],
        features['avg_neighbor_degree'],
        features.get('n_skills', 0),
        features.get('experience_years', 0),
        features.get('n_skill_communities', 0),
        features.get('specialization_score', 0)
    ])
    
    # Concatenate with embedding
    full_features = np.concatenate([structural_feats, embedding])
    
    return full_features

# Create feature matrix for all CVs
print("Creating feature matrices for classification...")
X_cv = []
y_seniority = []
y_specialization = []

for cv in cvs:
    features = create_node_features(
        cv['cv_id'], G, communities, cv_embeddings
    )
    X_cv.append(features)
    y_seniority.append(cv['seniority'])
    
    # Determine specialization based on skill modularity
    if cv.get('specialization_score', 0) > 0.4:
        y_specialization.append('specialist')
    else:
        y_specialization.append('polyvalent')

X_cv = np.array(X_cv)
```

**Task 5.2: Multi-Class Classification**
```python
from sklearn.preprocessing import LabelEncoder

# Seniority classification
le_seniority = LabelEncoder()
y_seniority_encoded = le_seniority.fit_transform(y_seniority)

X_train, X_test, y_train, y_test = train_test_split(
    X_cv, y_seniority_encoded, test_size=0.2, random_state=42
)

# Train classifier
clf_seniority = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
clf_seniority.fit(X_train, y_train)

# Evaluate
y_pred = clf_seniority.predict(X_test)
accuracy = np.mean(y_pred == y_test)
print(f"\nSeniority Classification Accuracy: {accuracy:.3f}")

from sklearn.metrics import classification_report
print("\nClassification Report:")
print(classification_report(
    y_test, y_pred,
    target_names=le_seniority.classes_
))

# Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=le_seniority.classes_,
            yticklabels=le_seniority.classes_)
plt.title('Seniority Classification Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/figures/seniority_confusion_matrix.png', dpi=300, bbox_inches='tight')
```

**Task 5.3: Semi-Supervised Iterative Classification (REV2)**
```python
def iterative_classification(G, labeled_nodes, labels, max_iterations=10):
    """
    Implement REV2 iterative classification from course material.
    """
    # Initialize: all nodes get temporary labels
    current_labels = labels.copy()
    
    # Get unlabeled nodes
    all_nodes = set(cv_nodes)
    unlabeled_nodes = all_nodes - set(labeled_nodes)
    
    # Initialize unlabeled with random labels
    for node in unlabeled_nodes:
        current_labels[node] = random.choice(list(set(labels.values())))
    
    # Train base classifier on labeled data only
    X_labeled = []
    y_labeled = []
    
    for node in labeled_nodes:
        features = create_node_features(node, G, communities, cv_embeddings)
        X_labeled.append(features)
        y_labeled.append(labels[node])
    
    base_clf = RandomForestClassifier(n_estimators=50, random_state=42)
    base_clf.fit(X_labeled, y_labeled)
    
    # Iterative refinement
    for iteration in range(max_iterations):
        print(f"Iteration {iteration + 1}/{max_iterations}")
        
        new_labels = current_labels.copy()
        
        # For each unlabeled node
        for node in unlabeled_nodes:
            # Get neighbor label distribution
            neighbors = list(G.neighbors(node))
            neighbor_labels = [current_labels.get(n, 0) for n in neighbors]
            
            if neighbor_labels:
                # Most common neighbor label
                from collections import Counter
                label_counts = Counter(neighbor_labels)
                
                # Create augmented features
                base_features = create_node_features(node, G, communities, cv_embeddings)
                
                # Add neighbor label features
                neighbor_features = np.array([
                    label_counts.get(label, 0) 
                    for label in set(labels.values())
                ])
                
                augmented_features = np.concatenate([base_features, neighbor_features])
                
                # Classify
                # Note: needs retrained classifier with augmented features
                # Simplified: use majority voting
                new_labels[node] = label_counts.most_common(1)[0][0]
        
        # Check convergence
        if new_labels == current_labels:
            print(f"Converged at iteration {iteration + 1}")
            break
        
        current_labels = new_labels
    
    return current_labels

# Run semi-supervised classification
# Use 80% as labeled, 20% as unlabeled
labeled_idx = random.sample(range(len(cvs)), int(0.8 * len(cvs)))
labeled_nodes = [cvs[i]['cv_id'] for i in labeled_idx]
labels = {cvs[i]['cv_id']: cvs[i]['seniority'] for i in labeled_idx}

predicted_labels = iterative_classification(G, labeled_nodes, labels)

# Evaluate on test set
test_nodes = [cv['cv_id'] for i, cv in enumerate(cvs) if i not in labeled_idx]
test_accuracy = np.mean([
    predicted_labels[node] == next(cv for cv in cvs if cv['cv_id'] == node)['seniority']
    for node in test_nodes
])

print(f"\nSemi-Supervised Classification Accuracy: {test_accuracy:.3f}")
```

**Task 5.4: Job Node Classification**
```python
# Job classification - Domain and Level prediction
print("\n=== Job Node Classification ===")

# Create feature matrix for all Jobs
X_job = []
y_job_domain = []
y_job_level = []

for job in jobs:
    features = create_node_features(
        job['job_id'], G, communities, job_embeddings
    )
    X_job.append(features)
    y_job_domain.append(job['domain'])
    y_job_level.append(job['level'])

X_job = np.array(X_job)

# Domain classification
le_domain = LabelEncoder()
y_domain_encoded = le_domain.fit_transform(y_job_domain)

X_train_job, X_test_job, y_train_domain, y_test_domain = train_test_split(
    X_job, y_domain_encoded, test_size=0.2, random_state=42
)

clf_domain = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
clf_domain.fit(X_train_job, y_train_domain)

y_pred_domain = clf_domain.predict(X_test_job)
accuracy_domain = np.mean(y_pred_domain == y_test_domain)
print(f"\nJob Domain Classification Accuracy: {accuracy_domain:.3f}")

print("\nDomain Classification Report:")
print(classification_report(
    y_test_domain, y_pred_domain,
    target_names=le_domain.classes_
))

# Level classification
le_level = LabelEncoder()
y_level_encoded = le_level.fit_transform(y_job_level)

X_train_job, X_test_job, y_train_level, y_test_level = train_test_split(
    X_job, y_level_encoded, test_size=0.2, random_state=42
)

clf_level = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    random_state=42
)
clf_level.fit(X_train_job, y_train_level)

y_pred_level = clf_level.predict(X_test_job)
accuracy_level = np.mean(y_pred_level == y_test_level)
print(f"Job Level Classification Accuracy: {accuracy_level:.3f}")

print("\nLevel Classification Report:")
print(classification_report(
    y_test_level, y_pred_level,
    target_names=le_level.classes_
))

# Visualize confusion matrices for Jobs
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Domain confusion matrix
cm_domain = confusion_matrix(y_test_domain, y_pred_domain)
sns.heatmap(cm_domain, annot=True, fmt='d', cmap='Greens',
            xticklabels=le_domain.classes_,
            yticklabels=le_domain.classes_,
            ax=axes[0])
axes[0].set_title('Job Domain Classification Confusion Matrix')
axes[0].set_ylabel('True Label')
axes[0].set_xlabel('Predicted Label')

# Level confusion matrix
cm_level = confusion_matrix(y_test_level, y_pred_level)
sns.heatmap(cm_level, annot=True, fmt='d', cmap='Oranges',
            xticklabels=le_level.classes_,
            yticklabels=le_level.classes_,
            ax=axes[1])
axes[1].set_title('Job Level Classification Confusion Matrix')
axes[1].set_ylabel('True Label')
axes[1].set_xlabel('Predicted Label')

plt.tight_layout()
plt.savefig('results/figures/job_classification_confusion_matrices.png', dpi=300, bbox_inches='tight')

# Feature importance for Job classification
feature_names_base = [
    'degree', 'betweenness', 'pagerank', 'clustering',
    'community_size', 'avg_neighbor_degree',
    'n_skills', 'experience_years', 'n_skill_communities', 'specialization_score'
]
feature_names = feature_names_base + [f'embedding_{i}' for i in range(384)]

# Domain feature importance
importances_domain = clf_domain.feature_importances_
indices_domain = np.argsort(importances_domain)[-15:]  # Top 15 features

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices_domain)), 
         importances_domain[indices_domain])
plt.yticks(range(len(indices_domain)), 
           [feature_names[i] if i < len(feature_names) else f'feat_{i}' 
            for i in indices_domain])
plt.xlabel('Feature Importance')
plt.title('Top 15 Features - Job Domain Classification')
plt.tight_layout()
plt.savefig('results/figures/job_domain_feature_importance.png', dpi=300, bbox_inches='tight')

# Level feature importance
importances_level = clf_level.feature_importances_
indices_level = np.argsort(importances_level)[-15:]

plt.figure(figsize=(10, 6))
plt.barh(range(len(indices_level)), 
         importances_level[indices_level])
plt.yticks(range(len(indices_level)), 
           [feature_names[i] if i < len(feature_names) else f'feat_{i}' 
            for i in indices_level])
plt.xlabel('Feature Importance')
plt.title('Top 15 Features - Job Level Classification')
plt.tight_layout()
plt.savefig('results/figures/job_level_feature_importance.png', dpi=300, bbox_inches='tight')

print("\nJob classification completed successfully!")
```

**Task 5.5: Feature Importance Analysis**
```python
# Comprehensive feature importance analysis for CV classification
feature_names_base = [
    'degree', 'betweenness', 'pagerank', 'clustering',
    'community_size', 'avg_neighbor_degree',
    'n_skills', 'experience_years', 'n_skill_communities', 'specialization_score'
]
feature_names = feature_names_base + [f'embedding_{i}' for i in range(384)]

importances = clf_seniority.feature_importances_
indices = np.argsort(importances)[-15:]  # Top 15 features

plt.figure(figsize=(10, 8))
plt.barh(range(len(indices)), importances[indices])
plt.yticks(range(len(indices)), 
           [feature_names[i] if i < len(feature_names) else f'feat_{i}' 
            for i in indices])
plt.xlabel('Feature Importance')
plt.title('Top 15 Features - CV Seniority Classification')
plt.tight_layout()
plt.savefig('results/figures/cv_feature_importance.png', dpi=300, bbox_inches='tight')

print("Feature importance analysis completed!")
```

---

### Phase 6: Analysis & Insights Generation

**Task 6.1: Impact Analysis**
```python
# Compare classification performance before and after graph enrichment
print("\n=== Impact Analysis ===")

# Before enrichment
features_before = [create_node_features(cv['cv_id'], G, communities, cv_embeddings) 
                   for cv in cvs]
X_before = np.array(features_before)
clf_before = RandomForestClassifier(n_estimators=100, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X_before, y_seniority_encoded, test_size=0.2, random_state=42
)
clf_before.fit(X_train, y_train)
accuracy_before = clf_before.score(X_test, y_test)

# After enrichment
features_after = [create_node_features(cv['cv_id'], G_enriched, 
                                       communities_enriched, cv_embeddings) 
                  for cv in cvs]
X_after = np.array(features_after)
clf_after = RandomForestClassifier(n_estimators=100, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    X_after, y_seniority_encoded, test_size=0.2, random_state=42
)
clf_after.fit(X_train, y_train)
accuracy_after = clf_after.score(X_test, y_test)

print(f"Classification Accuracy Before Enrichment: {accuracy_before:.3f}")
print(f"Classification Accuracy After Enrichment: {accuracy_after:.3f}")
print(f"Improvement: {(accuracy_after - accuracy_before):.3f}")
```

**Task 6.2: Atypical Profile Detection**
```python
# Detect atypical profiles
print("\n=== Atypical Profile Detection ===")

# 1. Polyvalent profiles (low specialization score)
polyvalent_cvs = [
    cv for cv in cvs 
    if cv.get('specialization_score', 0) < 0.3
]
print(f"Polyvalent CVs: {len(polyvalent_cvs)}")

# 2. Bridge profiles (high betweenness)
betweenness_threshold = np.percentile(list(betweenness.values()), 90)
bridge_cvs = [
    cv for cv in cvs
    if betweenness[cv['cv_id']] > betweenness_threshold
]
print(f"Bridge CVs: {len(bridge_cvs)}")

# 3. Misclassified profiles
misclassified = [
    cvs[i] for i in range(len(cvs))
    if y_pred[i] != y_test[i] and i < len(y_test)
]
print(f"Misclassified CVs: {len(misclassified)}")

# Analyze characteristics
if misclassified:
    avg_spec = np.mean([cv.get('specialization_score', 0) 
                        for cv in misclassified])
    print(f"Avg specialization of misclassified: {avg_spec:.3f}")
```

**Task 6.3: Specialization Coherence Analysis**
```python
# Analyze correlation between CV specialization and community specialization
print("\n=== Specialization Coherence Analysis ===")

cv_spec_scores = [cv.get('specialization_score', 0) for cv in cvs]

# Compute community specialization (avg of member specializations)
community_spec = {}
for comm_id in set(communities.values()):
    cvs_in_comm = [cv for cv in cvs if communities[cv['cv_id']] == comm_id]
    if cvs_in_comm:
        community_spec[comm_id] = np.mean([
            cv.get('specialization_score', 0) for cv in cvs_in_comm
        ])
    else:
        community_spec[comm_id] = 0

cv_comm_spec = [community_spec[communities[cv['cv_id']]] for cv in cvs]

# Compute correlation
correlation = np.corrcoef(cv_spec_scores, cv_comm_spec)[0, 1]
print(f"Correlation (CV spec vs Community spec): {correlation:.3f}")

# Visualize
plt.figure(figsize=(10, 6))
plt.scatter(cv_spec_scores, cv_comm_spec, alpha=0.6)
plt.xlabel('CV Specialization Score')
plt.ylabel('Community Specialization Score')
plt.title(f'Specialization Coherence (r={correlation:.3f})')
plt.savefig('results/figures/specialization_coherence.png', dpi=300, bbox_inches='tight')
```

---

### Phase 7: Report Generation

**Task 7.1: Generate Comprehensive Report**
```python
# Create summary statistics
report = f"""
# CV-Job Graph Mining Project - Results Report

## 1. Graph Statistics
- Total CVs: {len(cv_nodes)}
- Total Jobs: {len(job_nodes)}
- Total Edges: {G.number_of_edges()}
- Network Density: {density:.3f}
- Average Clustering: {avg_clustering:.3f}

## 2. Community Detection Results
- Number of Communities: {len(set(communities.values()))}
- Modularity Q: {modularity:.3f}
- Interpretation: {'Significant structure' if modularity > 0.3 else 'Weak structure'}

## 3. Link Prediction Performance
- Precision: {precision:.3f}
- Recall: {recall:.3f}
- AUC-ROC: {auc:.3f}
- Predicted Links Added: {len(predicted_links)}

## 4. Node Classification Results

### CV Classification
- Seniority Classification Accuracy: {accuracy:.3f}
- Semi-Supervised Accuracy: {test_accuracy:.3f}

### Job Classification
- Domain Classification Accuracy: {accuracy_domain:.3f}
- Level Classification Accuracy: {accuracy_level:.3f}

## 5. Graph Enrichment Impact
- Accuracy Before: {accuracy_before:.3f}
- Accuracy After: {accuracy_after:.3f}
- Improvement: {(accuracy_after - accuracy_before):.3f}

## 6. Profile Analysis
- Polyvalent CVs: {len(polyvalent_cvs)}
- Bridge CVs: {len(bridge_cvs)}
- Specialization Coherence: {correlation:.3f}

## 7. Key Insights
1. Community structure is {'strong' if modularity > 0.5 else 'moderate'}
2. Hybrid approach outperforms pure graph or LLM methods
3. Graph enrichment {'improves' if accuracy_after > accuracy_before else 'does not improve'} classification
4. Specialization and community position are {'strongly' if abs(correlation) > 0.7 else 'moderately'} correlated
"""

with open('results/report.md', 'w') as f:
    f.write(report)

print(report)
```

---

## 📋 Expected Deliverables

### 1. Code Structure
```
cv-job-graph-project/
├── data/
│   ├── cv_job_graph.pkl
│   └── synthetic_data.json
├── results/
│   ├── figures/
│   │   ├── degree_distributions.png
│   │   ├── bipartite_graph.png
│   │   ├── communities.png
│   │   ├── seniority_confusion_matrix.png
│   │   ├── job_classification_confusion_matrices.png
│   │   ├── cv_feature_importance.png
│   │   ├── job_domain_feature_importance.png
│   │   ├── job_level_feature_importance.png
│   │   └── specialization_coherence.png
│   ├── metrics/
│   │   ├── node_metrics.csv
│   │   └── community_assignments.csv
│   └── report.md
├── src/
│   ├── graph_builder.py
│   ├── community_detector.py
│   ├── link_predictor.py
│   ├── node_classifier.py
│   └── analyzer.py
├── notebooks/
│   └── full_analysis.ipynb
├── requirements.txt
└── README.md
```

### 2. Performance Targets
- **Modularity Q**: > 0.3 (significant community structure)
- **Link Prediction AUC**: > 0.75
- **CV Classification Accuracy**: > 0.70 (seniority)
- **Job Classification Accuracy**: > 0.70 (domain and level)
- **Graph Enrichment Improvement**: > 0.05

### 3. Visualizations Required
- Degree distributions (CV and Job separate)
- Bipartite graph with community colors
- Confusion matrices for CV classification (seniority)
- Confusion matrices for Job classification (domain and level)
- Feature importance plots (CV and Job separate)
- Specialization coherence scatter plot

---

## 🔧 Technical Requirements

### Dependencies
```txt
networkx>=3.0
numpy>=1.24
pandas>=2.0
matplotlib>=3.7
seaborn>=0.12
scikit-learn>=1.3
sentence-transformers>=2.2
groq>=0.4
python-louvain>=0.16
```

### Environment Variables
```bash
export GROQ_API_KEY="your_groq_api_key_here"
```

### Key Implementation Notes

1. **Use Louvain for Community Detection**: Fast and effective for medium-sized graphs
2. **Combine Features**: Structural + Semantic + Community features
3. **Validate Properly**: Always use held-out test sets
4. **LLM Integration**: Use Groq API with Llama for semantic scoring
5. **Threshold Tuning**: Adjust link prediction threshold based on precision-recall trade-off
6. **Feature Engineering**: Extract maximum value from internal skill graphs

---

## 🎯 Success Criteria

1. ✅ **Graph Construction**: Bipartite graph with 100 CVs, 80 Jobs, 150-200 edges
2. ✅ **Community Detection**: Modularity Q > 0.3
3. ✅ **Link Prediction**: AUC > 0.75 using hybrid approach
4. ✅ **CV Node Classification**: Accuracy > 0.70 for seniority and specialization prediction
5. ✅ **Job Node Classification**: Accuracy > 0.70 for domain and level prediction
6. ✅ **Graph Enrichment**: Demonstrated improvement in classification
7. ✅ **Insights**: Clear analysis of atypical profiles and specialization coherence
8. ✅ **Code Quality**: Modular, documented, reproducible

---

## 💡 Advanced Features (Optional)

1. **GNN Implementation**: Use PyTorch Geometric for graph neural network classifier
2. **Temporal Analysis**: Simulate graph evolution over time
3. **Interactive Visualization**: Use Plotly for interactive graph exploration
4. **Explainability**: Add SHAP values for feature importance
5. **Multi-Task Learning**: Joint prediction of seniority + specialization

---

## 📝 Final Notes

- **Focus on Hybrid Approach**: The combination of LLM semantic understanding and graph structural patterns is the key innovation
- **Document Everything**: Clear comments and explanations for each step
- **Validate Results**: Use proper train/test splits and cross-validation
- **Interpret Findings**: Generate actionable insights, not just metrics
- **Course Integration**: Explicitly reference which algorithms/techniques from the course material are being applied

**Remember**: The goal is not just to predict links and classify nodes, but to understand WHY certain patterns emerge and provide valuable insights for the recruitment process.

Start implementation and ask for clarification if needed!
