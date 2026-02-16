# CV-Job Graph Mining Project

A bipartite graph mining system combining traditional graph algorithms with modern LLM capabilities for CV-Job matching and analysis.

## Team Members
- **Ramy**: CV Data Pipeline, Community Detection, CV Classification
- **Chiheb**: Job Data, LLM Integration, Link Prediction

## Project Structure

```
cv-job-graph-project/
├── data/                           # Data files
│   ├── cvs.json                   # CV data (Ramy)
│   ├── cvs_with_graphs.json       # CVs with skill graphs (Ramy)
│   ├── jobs.json                  # Job data (Chiheb)
│   ├── cv_job_graph.pkl           # Main bipartite graph
│   ├── cv_embeddings.npy          # CV embeddings (Chiheb)
│   └── job_embeddings.npy         # Job embeddings (Chiheb)
│
├── src/                           # Source code
│   ├── cv_generator.py            # Ramy: CV data generation
│   ├── cv_graph_builder.py        # Ramy: Internal skill graphs
│   ├── cv_community.py            # Ramy: CV community detection
│   ├── cv_classifier.py           # Ramy: CV classification
│   ├── cv_analyzer.py             # Ramy: Atypical profile analysis
│   │
│   ├── job_generator.py           # Chiheb: Job data generation
│   ├── llm_utils.py               # Chiheb: LLM/embedding utilities
│   ├── link_predictor.py          # Chiheb: Link prediction
│   ├── job_classifier.py          # Chiheb: Job classification
│   ├── graph_enrichment.py        # Chiheb: Graph enrichment
│   │
│   ├── graph_builder.py           # Shared: Base graph construction
│   ├── metrics_computer.py        # Shared: Centrality metrics
│   └── visualizer.py              # Shared: Visualization utils
│
├── results/                       # Analysis results
│   ├── figures/                   # Visualizations
│   ├── metrics/                   # Computed metrics (CSV/JSON)
│   └── analysis/                  # Analysis outputs
│
├── models/                        # Trained models
├── tests/                         # Unit tests
├── notebooks/                     # Jupyter notebooks
├── requirements.txt               # Python dependencies
└── README.md                      # This file
```

## Getting Started

### Prerequisites
```bash
Python >= 3.8
```

### Installation
```bash
# Clone the repository
git clone <repository-url>
cd cv-job-graph-project

# Install dependencies
pip install -r requirements.txt
```

## Workflow

### Ramy's Workflow (CV Pipeline)
```bash
# Run the main pipeline
python ramy_main_pipeline.py

# This will execute:
# - Phase 1: Generate CVs and build initial graph
# - Phase 2: Detect skill communities
# - (Phases 3-6 after Chiheb's handoff)
```

### Chiheb's Workflow (Job & Link Pipeline)
```bash
# After Ramy completes Phase 1-2:
# 1. Load graph from data/cv_job_graph.pkl
# 2. Add Job nodes
# 3. Generate embeddings
# 4. Run link prediction
```

## Git Workflow

### Branch Strategy
- **main**: Stable integration branch
- **ramy/cv-development**: Ramy's CV-related work
- **chiheb/job-link-development**: Chiheb's Job and Link work

### Ramy's Git Commands
```bash
# Create and switch to your branch
git checkout -b ramy/cv-development

# Work on your tasks
# ... edit files ...

# Stage and commit
git add src/cv_*.py data/cvs*.json
git commit -m "feat: CV generation and skill graphs"

# Push to remote
git push origin ramy/cv-development

# When ready to integrate (after testing)
git checkout main
git merge ramy/cv-development
git push origin main
```

## Key Handoff Points

### Ramy → Chiheb
**Files to share:**
- `data/cv_job_graph.pkl` - Graph with CV nodes
- `data/cvs.json` - CV data for embeddings

**What Chiheb needs to do:**
1. Add 80 Job nodes to graph
2. Generate CV and Job embeddings
3. Return updated graph

### Chiheb → Ramy
**Files to share:**
- `data/cv_job_graph.pkl` - Updated graph with Jobs
- `data/cv_embeddings.npy` - CV embeddings

**What Ramy needs to do:**
1. Run global community detection
2. Complete CV classification with embeddings
3. Generate analysis reports

## Performance Targets
- ✅ Modularity Q > 0.3
- ✅ Link Prediction AUC > 0.75
- ✅ CV Classification Accuracy > 0.70
- ✅ Job Classification Accuracy > 0.70

## Contact
- **Ramy**: CV and Community Detection
- **Chiheb**: Jobs and Link Prediction

## License
Academic Project - 2026