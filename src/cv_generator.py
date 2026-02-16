"""
CV Data Generator Module

This module generates synthetic CV data for the CV-Job graph mining project.
Creates realistic CVs with skills, experience, education, and other attributes.

Owner: Ramy
Dependencies: None (fully independent)
"""

import json
import random
import numpy as np
from typing import List, Dict, Any


class CVGenerator:
    """
    Generates synthetic CV data with realistic attributes and distributions.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the CV generator.
        
        Args:
            seed (int): Random seed for reproducibility
        """
        random.seed(seed)
        np.random.seed(seed)
        
        # Define skill pools by domain
        self.tech_skills = [
            'Python', 'Java', 'JavaScript', 'C++', 'SQL', 'R', 'Go', 'Rust',
            'Machine Learning', 'Deep Learning', 'Data Analysis', 'NLP',
            'Computer Vision', 'TensorFlow', 'PyTorch', 'Scikit-learn',
            'Django', 'Flask', 'React', 'Node.js', 'AWS', 'Docker', 'Kubernetes',
            'Git', 'CI/CD', 'Agile', 'REST APIs', 'GraphQL', 'MongoDB', 'PostgreSQL'
        ]
        
        self.business_skills = [
            'Project Management', 'Business Analysis', 'Strategic Planning',
            'Financial Analysis', 'Market Research', 'Sales', 'Marketing',
            'Customer Relations', 'Negotiation', 'Leadership', 'Communication',
            'Excel', 'PowerPoint', 'Tableau', 'Power BI', 'CRM', 'ERP',
            'Risk Management', 'Budgeting', 'Forecasting', 'Process Improvement'
        ]
        
        self.research_skills = [
            'Research Design', 'Statistical Analysis', 'SPSS', 'R', 'Python',
            'Academic Writing', 'Literature Review', 'Data Collection',
            'Experiment Design', 'Hypothesis Testing', 'Regression Analysis',
            'Qualitative Research', 'Quantitative Research', 'Survey Design',
            'Publication', 'Grant Writing', 'Peer Review', 'MATLAB', 'LaTeX'
        ]
        
        # Education levels
        self.education_options = {
            'junior': [
                ['BS Computer Science'],
                ['BS Engineering'],
                ['BS Mathematics'],
                ['BS Data Science'],
                ['BS Information Technology']
            ],
            'mid': [
                ['BS Computer Science', 'MS Data Science'],
                ['BS Engineering', 'MS Computer Science'],
                ['BS Mathematics', 'MS Statistics'],
                ['BS Business', 'MBA'],
                ['BS Physics', 'MS Machine Learning']
            ],
            'senior': [
                ['BS Computer Science', 'MS AI', 'PhD Computer Science'],
                ['BS Engineering', 'MS Engineering', 'MBA'],
                ['BS Mathematics', 'MS Statistics', 'PhD Statistics'],
                ['BS Physics', 'PhD Physics'],
                ['BS Computer Science', 'MS Data Science', 'Executive MBA']
            ]
        }
        
        # Domain definitions
        self.domains = ['tech', 'business', 'research']
        
    def generate_cvs(self, n_cvs: int = 100) -> List[Dict[str, Any]]:
        """
        Generate synthetic CVs with diverse attributes.
        
        Args:
            n_cvs (int): Number of CVs to generate (default: 100)
        
        Returns:
            List[Dict]: List of CV dictionaries with all attributes
        """
        cvs = []
        
        for i in range(n_cvs):
            cv = self._generate_single_cv(i)
            cvs.append(cv)
        
        print(f"✅ Generated {len(cvs)} CVs")
        self._print_statistics(cvs)
        
        return cvs
    
    def _generate_single_cv(self, index: int) -> Dict[str, Any]:
        """
        Generate a single CV with all attributes.
        
        Args:
            index (int): CV index for unique ID
        
        Returns:
            Dict: CV dictionary with all attributes
        """
        cv_id = f"CV_{index:03d}"
        
        # Randomly assign domain (40% tech, 35% business, 25% research)
        domain = np.random.choice(
            self.domains, 
            p=[0.40, 0.35, 0.25]
        )
        
        # Assign seniority level (30% junior, 45% mid, 25% senior)
        seniority = np.random.choice(
            ['junior', 'mid', 'senior'],
            p=[0.30, 0.45, 0.25]
        )
        
        # Generate experience years based on seniority
        experience_years = self._generate_experience(seniority)
        
        # Select education based on seniority
        education = random.choice(self.education_options[seniority])
        
        # Generate skills based on domain (5-10 skills)
        skills = self._generate_skills(domain, seniority)
        
        # Generate experiences (projects/jobs where skills were used)
        experiences = self._generate_experiences(skills, experience_years)
        
        # Generate raw text description
        raw_text = self._generate_cv_text(
            cv_id, skills, experience_years, education, seniority, domain, experiences
        )
        
        cv = {
            'cv_id': cv_id,
            'skills': skills,
            'experience_years': experience_years,
            'education': education,
            'seniority': seniority,
            'domain': domain,
            'experiences': experiences,
            'raw_text': raw_text
        }
        
        return cv
    
    def _generate_experience(self, seniority: str) -> int:
        """Generate experience years based on seniority level."""
        if seniority == 'junior':
            return int(np.random.normal(2, 1))  # 0-4 years
        elif seniority == 'mid':
            return int(np.random.normal(6, 2))  # 3-10 years
        else:  # senior
            return int(np.random.normal(12, 3))  # 8-18 years
    
    def _generate_skills(self, domain: str, seniority: str) -> List[str]:
        """
        Generate skills based on domain and seniority.
        
        Args:
            domain (str): CV domain (tech/business/research)
            seniority (str): Seniority level
        
        Returns:
            List[str]: List of skills (5-10 skills)
        """
        # Number of skills based on seniority
        if seniority == 'junior':
            n_skills = random.randint(5, 7)
        elif seniority == 'mid':
            n_skills = random.randint(7, 9)
        else:  # senior
            n_skills = random.randint(8, 10)
        
        # Select primary skills from domain
        if domain == 'tech':
            primary_pool = self.tech_skills
            secondary_pool = self.business_skills
        elif domain == 'business':
            primary_pool = self.business_skills
            secondary_pool = self.tech_skills
        else:  # research
            primary_pool = self.research_skills
            secondary_pool = self.tech_skills
        
        # 70-80% from primary domain, rest from secondary (for polyvalent profiles)
        n_primary = int(n_skills * random.uniform(0.7, 0.8))
        n_secondary = n_skills - n_primary
        
        skills = (
            random.sample(primary_pool, min(n_primary, len(primary_pool))) +
            random.sample(secondary_pool, min(n_secondary, len(secondary_pool)))
        )
        
        return skills[:n_skills]
    
    def _generate_experiences(self, skills: List[str], years: int) -> List[Dict[str, Any]]:
        """
        Generate work experiences/projects where skills co-occur.
        
        Args:
            skills (List[str]): CV skills
            years (int): Total experience years
        
        Returns:
            List[Dict]: List of experiences with associated skills
        """
        n_experiences = max(2, min(years // 2, 6))  # 2-6 experiences
        experiences = []
        
        # Ensure all skills appear in at least one experience
        skills_copy = skills.copy()
        random.shuffle(skills_copy)
        
        for i in range(n_experiences):
            # Each experience uses 2-5 skills
            n_exp_skills = random.randint(2, min(5, len(skills)))
            
            if i < len(skills_copy):
                # Ensure coverage of all skills
                exp_skills = [skills_copy[i]]
                remaining = random.sample(
                    [s for s in skills if s != skills_copy[i]], 
                    n_exp_skills - 1
                )
                exp_skills.extend(remaining)
            else:
                exp_skills = random.sample(skills, n_exp_skills)
            
            experience = {
                'title': f'Project/Position {i+1}',
                'duration_years': round(random.uniform(0.5, 3), 1),
                'skills_used': exp_skills,
                'description': f'Worked on {", ".join(exp_skills[:2])} and related technologies'
            }
            experiences.append(experience)
        
        return experiences
    
    def _generate_cv_text(
        self, 
        cv_id: str, 
        skills: List[str], 
        years: int, 
        education: List[str], 
        seniority: str,
        domain: str,
        experiences: List[Dict]
    ) -> str:
        """
        Generate full CV text description.
        
        Args:
            cv_id (str): CV identifier
            skills (str): List of skills
            years (int): Experience years
            education (List[str]): Education degrees
            seniority (str): Seniority level
            domain (str): Domain
            experiences (List[Dict]): Work experiences
        
        Returns:
            str: Full CV text description
        """
        text = f"""CV ID: {cv_id}

PROFESSIONAL SUMMARY:
{seniority.capitalize()} professional with {years} years of experience in {domain}. 
Expertise in {', '.join(skills[:3])} and related technologies.

EDUCATION:
{chr(10).join(['- ' + edu for edu in education])}

SKILLS:
{', '.join(skills)}

EXPERIENCE:
"""
        for exp in experiences:
            text += f"""
- {exp['title']} ({exp['duration_years']} years)
  Skills: {', '.join(exp['skills_used'])}
  {exp['description']}
"""
        
        return text
    
    def save_cvs(self, cvs: List[Dict[str, Any]], filepath: str = 'data/cvs.json'):
        """
        Save generated CVs to JSON file.
        
        Args:
            cvs (List[Dict]): List of CV dictionaries
            filepath (str): Output file path
        """
        with open(filepath, 'w') as f:
            json.dump(cvs, f, indent=2)
        
        print(f"✅ Saved {len(cvs)} CVs to {filepath}")
    
    def _print_statistics(self, cvs: List[Dict[str, Any]]):
        """Print statistics about generated CVs."""
        print("\n📊 CV Generation Statistics:")
        print(f"   Total CVs: {len(cvs)}")
        
        # Seniority distribution
        seniority_counts = {}
        for cv in cvs:
            seniority_counts[cv['seniority']] = seniority_counts.get(cv['seniority'], 0) + 1
        print(f"   Seniority: {seniority_counts}")
        
        # Domain distribution
        domain_counts = {}
        for cv in cvs:
            domain_counts[cv['domain']] = domain_counts.get(cv['domain'], 0) + 1
        print(f"   Domains: {domain_counts}")
        
        # Experience range
        exp_years = [cv['experience_years'] for cv in cvs]
        print(f"   Experience: {min(exp_years)}-{max(exp_years)} years (avg: {np.mean(exp_years):.1f})")
        
        # Skills range
        skill_counts = [len(cv['skills']) for cv in cvs]
        print(f"   Skills per CV: {min(skill_counts)}-{max(skill_counts)} (avg: {np.mean(skill_counts):.1f})")


if __name__ == "__main__":
    # Generate CVs
    generator = CVGenerator(seed=42)
    cvs = generator.generate_cvs(n_cvs=100)
    
    # Save to file
    generator.save_cvs(cvs, 'data/cvs.json')
    
    print("\n✅ CV generation complete!")