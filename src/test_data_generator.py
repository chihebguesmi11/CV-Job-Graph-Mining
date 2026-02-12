"""
Test Data Generator

Creates synthetic CV data for testing and demonstration.
Used when Ramy's CV data is not yet available.
"""

import json
import random
import numpy as np
from typing import List, Dict, Any


class TestCVGenerator:
    """Generates test CV data for pipeline testing."""
    
    def __init__(self, seed: int = 42):
        """Initialize with random seed."""
        random.seed(seed)
        np.random.seed(seed)
        
        self.seniority_levels = ['junior', 'mid', 'senior']
        
        self.skill_pool = {
            'engineering': [
                'Python', 'Java', 'C++', 'JavaScript', 'Go', 'Rust',
                'Django', 'FastAPI', 'Spring', 'React', 'Vue.js', 'Angular',
                'Docker', 'Kubernetes', 'AWS', 'GCP', 'Azure',
                'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch',
                'Git', 'CI/CD', 'Microservices', 'REST API', 'GraphQL'
            ],
            'data_science': [
                'Python', 'R', 'SQL', 'Machine Learning', 'Deep Learning',
                'TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy',
                'Matplotlib', 'Seaborn', 'Spark', 'AWS', 'GCP',
                'Statistics', 'A/B Testing', 'Data Mining', 'Feature Engineering'
            ],
            'management': [
                'Project Management', 'Agile', 'Scrum', 'Team Leadership',
                'Budget Management', 'Strategic Planning', 'Business Analysis',
                'JIRA', 'Confluence', 'Excel', 'Communication', 'Problem Solving'
            ]
        }
        
        self.companies = ['Google', 'Meta', 'Amazon', 'Microsoft', 'Apple', 'Netflix']
    
    def generate_single_cv(self, cv_id: str) -> Dict[str, Any]:
        """Generate a single CV."""
        domain = random.choice(list(self.skill_pool.keys()))
        experience_years = random.randint(1, 15)
        
        # Determine seniority based on experience
        if experience_years < 3:
            seniority = 'junior'
        elif experience_years < 8:
            seniority = 'mid'
        else:
            seniority = 'senior'
        
        skills = random.sample(self.skill_pool[domain], random.randint(5, 10))
        
        description = f"""
Professional with {experience_years} years of experience in {domain}.
Skilled in: {', '.join(skills[:5])}.
Based at {random.choice(self.companies)}."""
        
        return {
            'cv_id': cv_id,
            'skills': skills,
            'experience_years': experience_years,
            'seniority': seniority,
            'domain': domain,
            'raw_text': description
        }
    
    def generate_cvs(self, n_cvs: int = 100) -> List[Dict[str, Any]]:
        """Generate multiple CVs."""
        cvs = []
        for i in range(n_cvs):
            cv = self.generate_single_cv(f"cv_{i:03d}")
            cvs.append(cv)
        return cvs
    
    def save_cvs(self, cvs: List[Dict[str, Any]], filepath: str) -> None:
        """Save CVs to JSON."""
        with open(filepath, 'w') as f:
            json.dump(cvs, f, indent=2)
        print(f"Saved {len(cvs)} test CVs to {filepath}")


if __name__ == '__main__':
    generator = TestCVGenerator()
    cvs = generator.generate_cvs(n_cvs=100)
    generator.save_cvs(cvs, 'data/test_cvs.json')
    print(f"Generated {len(cvs)} test CVs")
