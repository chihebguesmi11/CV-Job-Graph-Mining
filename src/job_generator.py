"""
Job Data Generator Module

Generates synthetic job data with realistic attributes for CV-Job matching.
Owner: Chiheb
Dependencies: None (independent)
"""

import json
import random
from typing import List, Dict, Any
import numpy as np


class JobGenerator:
    """
    Generates synthetic job postings with comprehensive attributes.
    
    Attributes:
        job_domains (List[str]): Available job domains
        job_levels (List[str]): Available job levels
        skill_pool (Dict[str, List[str]]): Skills organized by domain
        companies (List[str]): Sample company names
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize job generator with random seed.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        self.job_domains = ['engineering', 'data_science', 'management']
        self.job_levels = ['entry', 'mid', 'senior', 'lead']
        
        # Comprehensive skill pool organized by domain
        self.skill_pool = {
            'engineering': [
                'Python', 'Java', 'C++', 'JavaScript', 'Go', 'Rust',
                'Django', 'FastAPI', 'Spring', 'React', 'Vue.js', 'Angular',
                'Docker', 'Kubernetes', 'AWS', 'GCP', 'Azure',
                'PostgreSQL', 'MongoDB', 'Redis', 'Elasticsearch',
                'Git', 'CI/CD', 'Microservices', 'REST API', 'GraphQL',
                'System Design', 'OOP', 'Design Patterns'
            ],
            'data_science': [
                'Python', 'R', 'SQL', 'Scala', 'Julia',
                'Machine Learning', 'Deep Learning', 'NLP', 'Computer Vision',
                'TensorFlow', 'PyTorch', 'Scikit-learn', 'Keras',
                'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Plotly',
                'Spark', 'Hadoop', 'Hive', 'Airflow',
                'AWS', 'GCP', 'Azure', 'Databricks',
                'Statistics', 'A/B Testing', 'Data Mining', 'Feature Engineering',
                'XGBoost', 'LightGBM', 'Tableau', 'Power BI'
            ],
            'management': [
                'Project Management', 'Agile', 'Scrum', 'Kanban',
                'Budget Management', 'Team Leadership', 'Mentoring',
                'Stakeholder Management', 'Strategic Planning',
                'Business Analysis', 'Requirements Gathering',
                'Risk Management', 'JIRA', 'Confluence', 'Excel',
                'Communication', 'Problem Solving', 'Decision Making',
                'Product Planning', 'Roadmap Development', 'Cross-functional Collaboration'
            ]
        }
        
        self.companies = [
            'TechCorp', 'DataFlow Systems', 'CloudNine', 'InnovateLabs',
            'FutureTech', 'Digital Solutions Inc', 'AI Innovations', 'ByteWorks',
            'DataDrive', 'CloudMaster', 'TechForward', 'Quantum Leap',
            'NexGen Tech', 'Alpha Systems', 'Sigma Analytics', 'Omega Solutions'
        ]
        
        self.job_titles = {
            'engineering': [
                'Software Engineer', 'Senior Software Engineer', 'Tech Lead',
                'Backend Engineer', 'Frontend Engineer', 'Full Stack Engineer',
                'DevOps Engineer', 'Solutions Architect', 'Staff Engineer',
                'Engineering Manager', 'Principal Engineer'
            ],
            'data_science': [
                'Data Scientist', 'Senior Data Scientist', 'ML Engineer',
                'AI Engineer', 'Analytics Engineer', 'Data Engineer',
                'Machine Learning Engineer', 'Lead Data Scientist',
                'Principal ML Engineer', 'Analytics Manager'
            ],
            'management': [
                'Product Manager', 'Project Manager', 'Engineering Manager',
                'VP Engineering', 'Director of Engineering', 'VP Product',
                'Scrum Master', 'Technical Project Manager', 'Program Manager',
                'Operations Manager'
            ]
        }
        
        self.job_descriptions = {
            'engineering': [
                "We're looking for a talented engineer to build scalable systems.",
                "Join our team to develop cutting-edge cloud-based solutions.",
                "Help us build the infrastructure for tomorrow's applications.",
                "Work on mission-critical systems serving millions of users.",
                "Design and implement robust, maintainable code in a collaborative environment.",
                "Contribute to our open-source initiatives and shape the future of software.",
                "Lead technical initiatives and mentor junior engineers.",
                "Build APIs and microservices in a high-performance environment."
            ],
            'data_science': [
                "Apply advanced ML techniques to solve complex business problems.",
                "Build predictive models that drive real business impact.",
                "Work with large-scale datasets and implement efficient algorithms.",
                "Lead data-driven decision making across the organization.",
                "Develop and deploy machine learning models in production.",
                "Create insights from complex data to guide strategy.",
                "Research and implement cutting-edge deep learning approaches.",
                "Build recommendation systems serving millions of users."
            ],
            'management': [
                "Lead a high-performing team of talented engineers.",
                "Manage product strategy and roadmap execution.",
                "Drive cross-functional collaboration and deliver results.",
                "Build and mentor a world-class engineering team.",
                "Drive innovation through effective project management.",
                "Lead strategic initiatives and manage stakeholder expectations.",
                "Create a culture of excellence and continuous improvement.",
                "Guide the team to deliver on ambitious goals."
            ]
        }
    
    def generate_single_job(self, job_id: str) -> Dict[str, Any]:
        """
        Generate a single job posting with all attributes.
        
        Args:
            job_id (str): Unique identifier for the job
        
        Returns:
            Dict[str, Any]: Job data with all attributes
        """
        domain = random.choice(self.job_domains)
        level = random.choice(self.job_levels)
        
        # Determine required experience based on level
        experience_map = {
            'entry': (0, 2),
            'mid': (2, 5),
            'senior': (5, 10),
            'lead': (8, 15)
        }
        min_exp, max_exp = experience_map[level]
        experience_required = random.randint(min_exp, max_exp)
        
        # Select skills based on domain and count
        domain_skills = self.skill_pool[domain]
        skill_count = random.randint(4, 8)
        required_skills = random.sample(domain_skills, min(skill_count, len(domain_skills)))
        
        # Generate description
        description = random.choice(self.job_descriptions[domain])
        skills_text = ', '.join(required_skills)
        full_description = f"{description}\n\nRequired Skills: {skills_text}\nExperience Required: {experience_required} years"
        
        job = {
            'job_id': job_id,
            'company': random.choice(self.companies),
            'title': random.choice(self.job_titles[domain]),
            'domain': domain,
            'level': level,
            'required_skills': required_skills,
            'experience_required': experience_required,
            'description': full_description,
            'raw_text': full_description,
            'salary_range': self._generate_salary(level, domain),
            'location': random.choice(['Remote', 'On-site', 'Hybrid']),
            'benefits': self._generate_benefits()
        }
        
        return job
    
    def _generate_salary(self, level: str, domain: str) -> Dict[str, int]:
        """
        Generate realistic salary range based on level and domain.
        
        Args:
            level (str): Job level
            domain (str): Job domain
        
        Returns:
            Dict[str, int]: Salary range with min and max
        """
        base_salary = {
            'engineering': {'entry': 80000, 'mid': 120000, 'senior': 160000, 'lead': 200000},
            'data_science': {'entry': 90000, 'mid': 130000, 'senior': 170000, 'lead': 210000},
            'management': {'entry': 100000, 'mid': 140000, 'senior': 180000, 'lead': 250000}
        }
        
        min_sal = base_salary[domain][level]
        max_sal = min_sal + random.randint(20000, 50000)
        
        return {'min': min_sal, 'max': max_sal}
    
    def _generate_benefits(self) -> List[str]:
        """
        Generate list of job benefits.
        
        Returns:
            List[str]: List of benefits
        """
        all_benefits = [
            'Health Insurance', '401(k) Match', 'Stock Options',
            'Flexible Hours', 'Remote Work', 'Professional Development',
            'Gym Membership', 'Free Lunch', 'Mentorship Program',
            'Paid Time Off', 'Parental Leave', 'Mental Health Support'
        ]
        return random.sample(all_benefits, random.randint(3, 7))
    
    def generate_jobs(self, n_jobs: int = 80) -> List[Dict[str, Any]]:
        """
        Generate multiple jobs.
        
        Args:
            n_jobs (int): Number of jobs to generate (default: 80)
        
        Returns:
            List[Dict[str, Any]]: List of job data
        """
        jobs = []
        for i in range(n_jobs):
            job_id = f"job_{i:03d}"
            job = self.generate_single_job(job_id)
            jobs.append(job)
        
        return jobs
    
    def save_jobs(self, jobs: List[Dict[str, Any]], filepath: str) -> None:
        """
        Save jobs to JSON file.
        
        Args:
            jobs (List[Dict[str, Any]]): List of jobs to save
            filepath (str): Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(jobs, f, indent=2)
        print(f"Saved {len(jobs)} jobs to {filepath}")
    
    def load_jobs(self, filepath: str) -> List[Dict[str, Any]]:
        """
        Load jobs from JSON file.
        
        Args:
            filepath (str): Input file path
        
        Returns:
            List[Dict[str, Any]]: List of jobs
        """
        with open(filepath, 'r') as f:
            jobs = json.load(f)
        print(f"Loaded {len(jobs)} jobs from {filepath}")
        return jobs


def main():
    """Generate and save job data."""
    generator = JobGenerator(seed=42)
    jobs = generator.generate_jobs(n_jobs=80)
    generator.save_jobs(jobs, 'data/jobs.json')
    
    # Print summary
    print("\n=== Job Generation Summary ===")
    print(f"Total jobs: {len(jobs)}")
    
    domain_counts = {}
    level_counts = {}
    skill_counts = []
    
    for job in jobs:
        domain = job['domain']
        level = job['level']
        domain_counts[domain] = domain_counts.get(domain, 0) + 1
        level_counts[level] = level_counts.get(level, 0) + 1
        skill_counts.append(len(job['required_skills']))
    
    print("\nDomain Distribution:")
    for domain, count in sorted(domain_counts.items()):
        print(f"  {domain}: {count}")
    
    print("\nLevel Distribution:")
    for level, count in sorted(level_counts.items()):
        print(f"  {level}: {count}")
    
    print(f"\nAverage skills per job: {np.mean(skill_counts):.1f}")
    print(f"Skill range: {min(skill_counts)}-{max(skill_counts)}")


if __name__ == '__main__':
    main()
