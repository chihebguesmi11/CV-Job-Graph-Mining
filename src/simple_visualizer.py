"""Simple, reliable HTML-based graph visualizer without pyvis."""

import json
from pathlib import Path
from typing import Dict, Set


class SimpleGraphVisualizer:
    """Generate clean HTML for network visualization."""
    
    def __init__(self, output_dir: str = 'results/interactive'):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def visualize_bipartite_graph(self, graph, cv_nodes, job_nodes, communities,
                                 cv_data, job_data, title="Bipartite Graph",
                                 filename="bipartite_graph.html"):
        """Create bipartite graph visualization."""
        
        cv_lookup = {cv['cv_id']: cv for cv in cv_data}
        job_lookup = {job['job_id']: job for job in job_data}
        
        community_colors = {
            0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1',
            3: '#FFA07A', 4: '#98D8C8', 5: '#F7DC6F',
        }
        
        cv_degrees = {cv_id: graph.degree(cv_id) for cv_id in cv_nodes}
        job_degrees = {job_id: graph.degree(job_id) for job_id in job_nodes}
        
        max_cv_degree = max(cv_degrees.values()) if cv_degrees else 1
        max_job_degree = max(job_degrees.values()) if job_degrees else 1
        
        sorted_cvs = sorted(list(cv_nodes))
        sorted_jobs = sorted(list(job_nodes))
        
        n_cv = len(sorted_cvs)
        n_job = len(sorted_jobs)
        
        nodes = []
        edges = []
        edge_count = 0
        
        # CV nodes (left)
        for i, cv_id in enumerate(sorted_cvs):
            y = (i / (n_cv - 1) if n_cv > 1 else 0.5) * 1000 - 500
            cv = cv_lookup.get(cv_id, {})
            comm = communities.get(cv_id, 0)
            color = community_colors.get(comm, '#95A5A6')
            degree = cv_degrees[cv_id]
            size = 15 + (degree / max_cv_degree) * 35
            
            nodes.append({
                'id': cv_id,
                'label': cv_id[:10],
                'x': -600,
                'y': y,
                'color': color,
                'shape': 'box',
                'size': size,
                'title': f"{cv_id}: {cv.get('domain', 'N/A')}"
            })
        
        # Job nodes (right)
        for i, job_id in enumerate(sorted_jobs):
            y = (i / (n_job - 1) if n_job > 1 else 0.5) * 1000 - 500
            job = job_lookup.get(job_id, {})
            comm = communities.get(job_id, 0)
            color = community_colors.get(comm, '#95A5A6')
            degree = job_degrees[job_id]
            size = 15 + (degree / max_job_degree) * 35
            
            nodes.append({
                'id': job_id,
                'label': job_id[:10],
                'x': 600,
                'y': y,
                'color': color,
                'shape': 'dot',
                'size': size,
                'title': f"{job_id}: {job.get('title', 'N/A')}"
            })
        
        # Edges
        for cv, job, data in graph.edges(data=True):
            weight = data.get('weight', 1.0)
            edges.append({
                'from': cv,
                'to': job,
                'weight': weight,
                'opacity': min(0.5, 0.1 + weight * 0.4)
            })
            edge_count += 1
        
        html = self._generate_html(nodes, edges, title, n_cv, n_job, edge_count)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Saved (STABLE LAYOUT): {filename}")
        print(f"  Layout: {n_cv} CVs (left) ↔ {n_job} jobs (right)")
        print(f"  Edges: {edge_count}")
        print(f"  Physics: DISABLED (fixed positions)")
        
        return output_path
    
    def visualize_predicted_links(self, original_graph, predicted_links, cv_nodes,
                                 job_nodes, communities, cv_data, job_data,
                                 title="Predicted Links", filename="predicted_links.html"):
        """Create predicted links visualization."""
        
        cv_lookup = {cv['cv_id']: cv for cv in cv_data}
        job_lookup = {job['job_id']: job for job in job_data}
        
        community_colors = {
            0: '#FF6B6B', 1: '#4ECDC4', 2: '#45B7D1',
            3: '#FFA07A', 4: '#98D8C8', 5: '#F7DC6F',
        }
        
        predicted_cvs = set()
        predicted_jobs = set()
        
        for cv_id, job_id, prob in predicted_links:
            predicted_cvs.add(cv_id)
            predicted_jobs.add(job_id)
        
        sorted_cvs = sorted(list(predicted_cvs))
        sorted_jobs = sorted(list(predicted_jobs))
        
        n_cv = len(sorted_cvs)
        n_job = len(sorted_jobs)
        
        nodes = []
        edges = []
        
        # CV nodes
        for i, cv_id in enumerate(sorted_cvs):
            y = (i / (n_cv - 1) if n_cv > 1 else 0.5) * 800 - 400
            cv = cv_lookup.get(cv_id, {})
            comm = communities.get(cv_id, 0)
            color = community_colors.get(comm, '#95A5A6')
            n_pred = sum(1 for c, j, _ in predicted_links if c == cv_id)
            
            nodes.append({
                'id': cv_id,
                'label': cv_id[:10],
                'x': -500,
                'y': y,
                'color': color,
                'shape': 'box',
                'size': 20 + n_pred * 3,
                'title': f"{cv_id}: {n_pred} predictions"
            })
        
        # Job nodes
        for i, job_id in enumerate(sorted_jobs):
            y = (i / (n_job - 1) if n_job > 1 else 0.5) * 800 - 400
            job = job_lookup.get(job_id, {})
            comm = communities.get(job_id, 0)
            color = community_colors.get(comm, '#95A5A6')
            n_pred = sum(1 for c, j, _ in predicted_links if j == job_id)
            
            nodes.append({
                'id': job_id,
                'label': job_id[:10],
                'x': 500,
                'y': y,
                'color': color,
                'shape': 'dot',
                'size': 20 + n_pred * 3,
                'title': f"{job_id}: {n_pred} predictions"
            })
        
        # Edges
        for cv_id, job_id, prob in predicted_links:
            r = int(255 * (1 - prob))
            g = int(76 + 179 * prob)
            b = 80
            edges.append({
                'from': cv_id,
                'to': job_id,
                'prob': prob,
                'color': f'rgba({r},{g},{b},{0.4 + prob * 0.6}):'
            })
        
        html = self._generate_html(nodes, edges, title, n_cv, n_job, len(predicted_links), directed=True)
        
        output_path = self.output_dir / filename
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        
        print(f"✓ Saved (STABLE LAYOUT): {filename}")
        print(f"  Predicted links: {len(predicted_links)}")
        print(f"  CVs involved: {n_cv}")
        print(f"  Jobs involved: {n_job}")
        print(f"  Physics: DISABLED (fixed positions)")
        
        return output_path
    
    def _generate_html(self, nodes, edges, title, n_cvs, n_jobs, n_edges, directed=False):
        """Generate HTML from nodes and edges."""
        
        nodes_json = json.dumps(nodes)
        edges_json = json.dumps(edges)
        
        return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/vis-network.min.js"></script>
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/vis-network/9.1.2/dist/dist/vis-network.min.css">
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; }}
        #network {{ width: 100%; height: 100vh; background: #f5f5f5; }}
        #info {{ position: absolute; top: 15px; left: 15px; background: white; padding: 15px 20px; border-radius: 8px; box-shadow: 0 2px 8px rgba(0,0,0,0.15); z-index: 100; font-size: 13px; line-height: 1.6; }}
        .info-item {{ margin: 3px 0; }}
        .info-label {{ font-weight: 600; color: #333; }}
    </style>
</head>
<body>
    <div id="network"></div>
    <div id="info">
        <div class="info-item"><span class="info-label">Network:</span> {title}</div>
        <div class="info-item"><span class="info-label">CVs:</span> {n_cvs}</div>
        <div class="info-item"><span class="info-label">Jobs:</span> {n_jobs}</div>
        <div class="info-item"><span class="info-label">Edges:</span> {n_edges}</div>
        <div class="info-item"><span class="info-label">Layout:</span> Fixed Position</div>
    </div>
    
    <script type="text/javascript">
        var nodes = new vis.DataSet({nodes_json});
        var edges = new vis.DataSet({edges_json});
        
        var edgesArray = [];
        edges.forEach(function(edge) {{
            edgesArray.push({{
                from: edge.from,
                to: edge.to,
                width: edge.weight || 1,
                color: edge.color || 'rgba(100,100,100,0.3)',
                arrows: {{'to': {{'enabled': {'true' if directed else 'false'}, 'scaleFactor': 0.5}}}}
            }});
        }});
        
        var data = {{
            nodes: nodes,
            edges: new vis.DataSet(edgesArray)
        }};
        
        var options = {{
            physics: {{ enabled: false }},
            nodes: {{
                font: {{ size: 11, bold: {{ color: 'white' }} }},
                borderWidth: 2,
                margin: 5
            }},
            edges: {{
                smooth: {{ type: 'continuous' }},
                shadow: true
            }},
            interaction: {{
                hover: true,
                navigationButtons: true,
                keyboard: true
            }}
        }};
        
        var container = document.getElementById('network');
        var network = new vis.Network(container, data, options);
        
        window.addEventListener('resize', function() {{ network.fit(); }});
        setTimeout(function() {{ network.fit({{animation: false}}); }}, 200);
    </script>
</body>
</html>"""
