"""
Interactive Graph Visualizations - Neo4j style with HTML dashboards

Creates interactive, explorable visualizations of the CV-Job graph.
"""

import json
import networkx as nx
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, Set, Tuple
import webbrowser

# Pyvis for interactive network graphs
try:
    from pyvis.network import Network
except ImportError as e:
    raise ImportError("pyvis not installed. Install with: pip install pyvis") from e


class InteractiveGraphVisualizer:
    """Create interactive HTML visualizations of CV-Job graphs."""
    
    def __init__(self, output_dir: str = 'results/interactive'):
        """Initialize visualizer."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def _inject_edge_highlight_js(self, html_path: Path):
        """Inject JavaScript to highlight edges when a node is clicked (for smaller graphs)."""
        highlight_js = '''
        <script type="text/javascript">
        // Store original edge colors/widths
        var originalEdgeStyles = {};
        edges.get().forEach(function(edge) {
            originalEdgeStyles[edge.id] = {color: edge.color, width: edge.width};
        });
        
        // Edge highlight on node click
        network.on("click", function(params) {
            // Reset all edges to original style first
            edges.get().forEach(function(edge) {
                var orig = originalEdgeStyles[edge.id];
                edges.update({id: edge.id, color: orig.color, width: orig.width});
            });
            
            if (params.nodes.length > 0) {
                var selectedNode = params.nodes[0];
                var connectedEdges = network.getConnectedEdges(selectedNode);
                
                // Highlight connected edges in gold
                connectedEdges.forEach(function(edgeId) {
                    edges.update({id: edgeId, color: '#FFD700', width: 4});
                });
            }
        });
        </script>
        '''
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Insert before closing body tag
        content = content.replace('</body>', highlight_js + '\n</body>')
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def _inject_edge_highlight_js_optimized(self, html_path: Path):
        """Optimized edge highlight for large graphs - only updates changed edges."""
        highlight_js = '''
        <script type="text/javascript">
        // Store original edge colors/widths
        var originalEdgeStyles = {};
        var lastHighlighted = [];
        
        edges.get().forEach(function(edge) {
            originalEdgeStyles[edge.id] = {color: edge.color, width: edge.width};
        });
        
        // Edge highlight on node click - optimized version
        network.on("click", function(params) {
            // Reset only previously highlighted edges
            if (lastHighlighted.length > 0) {
                var updates = [];
                lastHighlighted.forEach(function(edgeId) {
                    var orig = originalEdgeStyles[edgeId];
                    if (orig) updates.push({id: edgeId, color: orig.color, width: orig.width});
                });
                edges.update(updates);
                lastHighlighted = [];
            }
            
            if (params.nodes.length > 0) {
                var selectedNode = params.nodes[0];
                var connectedEdges = network.getConnectedEdges(selectedNode);
                
                // Highlight connected edges in gold - batch update
                var updates = [];
                connectedEdges.forEach(function(edgeId) {
                    updates.push({id: edgeId, color: '#FFD700', width: 4});
                });
                edges.update(updates);
                lastHighlighted = connectedEdges;
            }
        });
        </script>
        '''
        
        with open(html_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = content.replace('</body>', highlight_js + '\n</body>')
        
        with open(html_path, 'w', encoding='utf-8') as f:
            f.write(content)
    
    def visualize_bipartite_graph(self,
                                 graph: nx.Graph,
                                 cv_nodes: Set[str],
                                 job_nodes: Set[str],
                                 communities: Dict[str, int],
                                 cv_data: list,
                                 job_data: list,
                                 title: str = "CV-Job Graph",
                                 filename: str = "bipartite_graph.html"):
        """
        Create simple interactive graph visualization.
        """
        print(f"🌐 Creating interactive graph...")
        
        # Create network - simple force-directed
        net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white')
        
        # Create lookup dicts
        cv_lookup = {cv['cv_id']: cv for cv in cv_data}
        job_lookup = {job['job_id']: job for job in job_data}
        
        # Simple colors - blue for CVs, orange for jobs
        cv_color = '#4A90D9'  # Blue
        job_color = '#F5A623'  # Orange
        
        # Calculate degrees for sizing
        cv_degrees = {cv_id: graph.degree(cv_id) for cv_id in cv_nodes if cv_id in graph}
        job_degrees = {job_id: graph.degree(job_id) for job_id in job_nodes if job_id in graph}
        
        max_cv_degree = max(cv_degrees.values()) if cv_degrees else 1
        max_job_degree = max(job_degrees.values()) if job_degrees else 1
        
        # Add CV nodes
        for cv_id in cv_nodes:
            if cv_id not in graph:
                continue
            cv = cv_lookup.get(cv_id, {})
            degree = cv_degrees.get(cv_id, 1)
            size = 10 + (degree / max_cv_degree) * 20
            
            title_text = f"<b>{cv_id}</b><br>Domain: {cv.get('domain', 'N/A')}<br>Seniority: {cv.get('seniority_level', 'N/A')}<br>Connections: {degree}"
            
            net.add_node(cv_id, label=cv_id, title=title_text, color=cv_color, size=size)
        
        # Add job nodes
        for job_id in job_nodes:
            if job_id not in graph:
                continue
            job = job_lookup.get(job_id, {})
            degree = job_degrees.get(job_id, 1)
            size = 10 + (degree / max_job_degree) * 20
            
            title_text = f"<b>{job_id}</b><br>Title: {job.get('title', 'N/A')}<br>Domain: {job.get('domain', 'N/A')}<br>Connections: {degree}"
            
            net.add_node(job_id, label=job_id, title=title_text, color=job_color, size=size)
        
        # Add edges - subtle gray
        edge_count = 0
        for u, v, data in graph.edges(data=True):
            weight = data.get('weight', 1.0)
            net.add_edge(u, v, color='#555555', width=1, title=f"Score: {weight:.2f}")
            edge_count += 1
        
        # Physics settings - stable after settling
        options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 150}
            },
            "interaction": {"hover": True, "tooltipDelay": 100}
        }
        net.set_options(json.dumps(options))
        
        output_path = self.output_dir / filename
        net.write_html(str(output_path), open_browser=False)
        self._inject_edge_highlight_js_optimized(output_path)
        
        print(f"✓ Saved: {filename} ({len(cv_nodes)} CVs, {len(job_nodes)} jobs, {edge_count} edges)")
        
        return output_path
    
    def visualize_predicted_links(self,
                                 original_graph: nx.Graph,
                                 predicted_links: list,
                                 cv_nodes: Set[str],
                                 job_nodes: Set[str],
                                 communities: Dict[str, int],
                                 cv_data: list,
                                 job_data: list,
                                 title: str = "Predicted Links",
                                 filename: str = "predicted_links.html"):
        """
        Create simple interactive graph showing predicted links.
        """
        print(f"🌐 Creating interactive predicted links graph...")
        
        net = Network(height='800px', width='100%', bgcolor='#222222', font_color='white', directed=True)
        
        cv_lookup = {cv['cv_id']: cv for cv in cv_data}
        job_lookup = {job['job_id']: job for job in job_data}
        
        cv_color = '#4A90D9'  # Blue
        job_color = '#F5A623'  # Orange
        
        # Get nodes involved in predictions
        predicted_cvs = set()
        predicted_jobs = set()
        for cv_id, job_id, prob in predicted_links:
            predicted_cvs.add(cv_id)
            predicted_jobs.add(job_id)
        
        # Add CV nodes
        for cv_id in predicted_cvs:
            cv = cv_lookup.get(cv_id, {})
            n_pred = sum(1 for c, j, _ in predicted_links if c == cv_id)
            size = 12 + (n_pred * 2)
            
            title_text = f"<b>{cv_id}</b><br>Domain: {cv.get('domain', 'N/A')}<br>Predictions: {n_pred}"
            net.add_node(cv_id, label=cv_id, title=title_text, color=cv_color, size=size)
        
        # Add job nodes
        for job_id in predicted_jobs:
            job = job_lookup.get(job_id, {})
            n_pred = sum(1 for c, j, _ in predicted_links if j == job_id)
            size = 12 + (n_pred * 2)
            
            title_text = f"<b>{job_id}</b><br>Title: {job.get('title', 'N/A')}<br>Predictions: {n_pred}"
            net.add_node(job_id, label=job_id, title=title_text, color=job_color, size=size)
        
        # Add edges with green color for high probability
        for cv_id, job_id, prob in predicted_links:
            green_intensity = int(100 + (prob * 155))
            color = f'rgb(50, {green_intensity}, 80)'
            width = 1 + (prob * 3)
            
            net.add_edge(cv_id, job_id, color=color, width=width, 
                        title=f"Match: {prob:.1%}",
                        arrows={'to': {'enabled': True, 'scaleFactor': 0.5}})
        
        # Physics settings
        options = {
            "physics": {
                "forceAtlas2Based": {
                    "gravitationalConstant": -50,
                    "centralGravity": 0.01,
                    "springLength": 100,
                    "springConstant": 0.08
                },
                "maxVelocity": 50,
                "solver": "forceAtlas2Based",
                "stabilization": {"iterations": 150}
            },
            "interaction": {"hover": True, "tooltipDelay": 100}
        }
        net.set_options(json.dumps(options))
        
        output_path = self.output_dir / filename
        net.write_html(str(output_path), open_browser=False)
        self._inject_edge_highlight_js(output_path)
        
        print(f"✓ Saved: {filename} ({len(predicted_cvs)} CVs, {len(predicted_jobs)} jobs, {len(predicted_links)} links)")
        
        return output_path
    
    def create_dashboard(self,
                        original_graph: nx.Graph,
                        enriched_graph: nx.Graph,
                        predicted_links: list,
                        cv_nodes: Set[str],
                        job_nodes: Set[str],
                        communities: Dict[str, int],
                        modularity: float,
                        link_metrics: Dict,
                        cv_data: list,
                        job_data: list,
                        filename: str = "dashboard.html"):
        """
        Create comprehensive interactive dashboard.
        
        Args:
            original_graph: Original graph
            enriched_graph: Enriched graph with predictions
            predicted_links: List of predicted links
            cv_nodes: Set of CV node IDs
            job_nodes: Set of job node IDs
            communities: Community assignments
            modularity: Modularity score
            link_metrics: Link prediction metrics
            cv_data: List of CV data dicts
            job_data: List of job data dicts
            filename: Output HTML filename
        """
        print(f"📊 Creating interactive dashboard...")
        
        # Calculate statistics
        n_communities = len(set(communities.values()))
        original_edges = original_graph.number_of_edges()
        enriched_edges = enriched_graph.number_of_edges()
        new_edges = len(predicted_links)
        
        # HTML content
        html = f"""
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>CV-Job Graph Mining - Interactive Dashboard</title>
            <style>
                * {{
                    margin: 0;
                    padding: 0;
                    box-sizing: border-box;
                }}
                
                body {{
                    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: #333;
                    padding: 20px;
                    min-height: 100vh;
                }}
                
                .container {{
                    max-width: 1400px;
                    margin: 0 auto;
                }}
                
                .header {{
                    background: white;
                    padding: 30px;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                    margin-bottom: 30px;
                    text-align: center;
                }}
                
                .header h1 {{
                    color: #667eea;
                    font-size: 32px;
                    margin-bottom: 10px;
                }}
                
                .header p {{
                    color: #666;
                    font-size: 16px;
                }}
                
                .metrics-grid {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                    gap: 20px;
                    margin-bottom: 30px;
                }}
                
                .metric-card {{
                    background: white;
                    padding: 25px;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    text-align: center;
                    transition: transform 0.3s, box-shadow 0.3s;
                }}
                
                .metric-card:hover {{
                    transform: translateY(-5px);
                    box-shadow: 0 10px 25px rgba(0,0,0,0.2);
                }}
                
                .metric-label {{
                    color: #999;
                    font-size: 14px;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                    margin-bottom: 10px;
                    font-weight: 600;
                }}
                
                .metric-value {{
                    color: #667eea;
                    font-size: 36px;
                    font-weight: bold;
                    margin-bottom: 5px;
                }}
                
                .metric-subtitle {{
                    color: #666;
                    font-size: 12px;
                }}
                
                .stat-box {{
                    background: white;
                    padding: 20px;
                    border-radius: 10px;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.1);
                    margin-bottom: 20px;
                }}
                
                .stat-box h3 {{
                    color: #667eea;
                    margin-bottom: 15px;
                    font-size: 18px;
                    border-left: 4px solid #667eea;
                    padding-left: 12px;
                }}
                
                .stat-item {{
                    display: flex;
                    justify-content: space-between;
                    padding: 10px 0;
                    border-bottom: 1px solid #eee;
                }}
                
                .stat-item:last-child {{
                    border-bottom: none;
                }}
                
                .stat-label {{
                    font-weight: 500;
                    color: #333;
                }}
                
                .stat-value {{
                    color: #667eea;
                    font-weight: bold;
                }}
                
                .links-section {{
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 20px;
                    margin-top: 30px;
                }}
                
                .link-button {{
                    display: block;
                    padding: 20px;
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    text-decoration: none;
                    border-radius: 10px;
                    text-align: center;
                    font-weight: bold;
                    font-size: 16px;
                    transition: transform 0.3s, box-shadow 0.3s;
                    box-shadow: 0 5px 15px rgba(0,0,0,0.2);
                    cursor: pointer;
                }}
                
                .link-button:hover {{
                    transform: translateY(-3px);
                    box-shadow: 0 8px 20px rgba(0,0,0,0.3);
                }}
                
                .link-button span {{
                    display: block;
                    font-size: 12px;
                    opacity: 0.9;
                    margin-top: 8px;
                }}
                
                .community-info {{
                    display: grid;
                    grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
                    gap: 15px;
                    margin-top: 20px;
                }}
                
                .community-badge {{
                    background: #f8f9fa;
                    padding: 15px;
                    border-radius: 8px;
                    text-align: center;
                    border-left: 4px solid #667eea;
                }}
                
                .community-badge .label {{
                    font-size: 12px;
                    color: #999;
                    text-transform: uppercase;
                    margin-bottom: 5px;
                    font-weight: 600;
                }}
                
                .community-badge .value {{
                    font-size: 20px;
                    font-weight: bold;
                    color: #667eea;
                }}
                
                .footer {{
                    text-align: center;
                    color: white;
                    margin-top: 40px;
                    font-size: 14px;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <div class="header">
                    <h1>🔗 CV-Job Graph Mining</h1>
                    <p>Interactive Dashboard - Explore the Bipartite Graph</p>
                </div>
                
                <div class="metrics-grid">
                    <div class="metric-card">
                        <div class="metric-label">Total CVs</div>
                        <div class="metric-value">{len(cv_nodes)}</div>
                        <div class="metric-subtitle">Candidates</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Total Jobs</div>
                        <div class="metric-value">{len(job_nodes)}</div>
                        <div class="metric-subtitle">Positions</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Communities</div>
                        <div class="metric-value">{n_communities}</div>
                        <div class="metric-subtitle">Modularity: {modularity:.3f}</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Original Edges</div>
                        <div class="metric-value">{original_edges}</div>
                        <div class="metric-subtitle">Initial Matches</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Predicted Links</div>
                        <div class="metric-value">{new_edges}</div>
                        <div class="metric-subtitle">New Matches</div>
                    </div>
                    <div class="metric-card">
                        <div class="metric-label">Link Precision</div>
                        <div class="metric-value">{link_metrics.get('precision', 0):.1%}</div>
                        <div class="metric-subtitle">Prediction Accuracy</div>
                    </div>
                </div>
                
                <div class="links-section">
                    <a href="bipartite_graph.html" class="link-button">
                        📊 Full Bipartite Graph
                        <span>Explore all CV-Job connections with communities</span>
                    </a>
                    <a href="predicted_links.html" class="link-button">
                        🎯 Predicted Links
                        <span>Interactive view of new predicted matches</span>
                    </a>
                </div>
                
                <div class="stat-box">
                    <h3>📈 Graph Statistics</h3>
                    <div class="stat-item">
                        <span class="stat-label">Total Nodes:</span>
                        <span class="stat-value">{len(cv_nodes) + len(job_nodes)}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Original Edges:</span>
                        <span class="stat-value">{original_edges}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Enriched Edges:</span>
                        <span class="stat-value">{enriched_edges}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">New Predictions:</span>
                        <span class="stat-value">{new_edges}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Density (Original):</span>
                        <span class="stat-value">{nx.density(original_graph):.4f}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Density (Enriched):</span>
                        <span class="stat-value">{nx.density(enriched_graph):.4f}</span>
                    </div>
                </div>
                
                <div class="stat-box">
                    <h3>🎯 Link Prediction Performance</h3>
                    <div class="stat-item">
                        <span class="stat-label">Precision:</span>
                        <span class="stat-value">{link_metrics.get('precision', 0):.1%}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Recall:</span>
                        <span class="stat-value">{link_metrics.get('recall', 0):.1%}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">Accuracy:</span>
                        <span class="stat-value">{link_metrics.get('accuracy', 0):.1%}</span>
                    </div>
                    <div class="stat-item">
                        <span class="stat-label">AUC-ROC:</span>
                        <span class="stat-value">{link_metrics.get('auc_roc', 0):.3f}</span>
                    </div>
                </div>
                
                <div class="footer">
                    <p>Generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
                </div>
            </div>
        </body>
        </html>
        """
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Saved: {filename}")
        
        return output_path
