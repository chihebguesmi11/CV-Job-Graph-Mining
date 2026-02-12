"""
LLM Utilities Module

Handles embeddings generation and LLM-based similarity scoring using Groq API.
Owner: Chiheb
Dependencies: sentence-transformers, groq, numpy
"""

import os
import json
import numpy as np
from typing import List, Dict, Tuple, Optional, Union
from pathlib import Path
import warnings

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    warnings.warn("sentence-transformers not available. Install it for embedding support.")

try:
    from groq import Groq
    GROQ_AVAILABLE = True
except ImportError:
    GROQ_AVAILABLE = False
    warnings.warn("groq not available. Install it for LLM support.")


class EmbeddingGenerator:
    """
    Generates embeddings for texts using sentence-transformers.
    
    Uses 'all-MiniLM-L6-v2' model by default (384-dimensional embeddings).
    
    Attributes:
        model_name (str): Name of the sentence-transformer model
        model: Loaded SentenceTransformer model
        embedding_dim (int): Dimension of embeddings
    """
    
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        """
        Initialize embedding generator.
        
        Args:
            model_name (str): Name of sentence-transformer model
                Default: 'all-MiniLM-L6-v2' (384 dimensions, fast, lightweight)
                Alternative: 'all-mpnet-base-v2' (768 dimensions, more powerful)
        
        Raises:
            ImportError: If sentence-transformers not available
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError("sentence-transformers is required. Install with: pip install sentence-transformers")
        
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        
        # Determine embedding dimension
        test_embedding = self.model.encode("test")
        self.embedding_dim = len(test_embedding)
        
        print(f"Loaded {model_name} model ({self.embedding_dim} dimensions)")
    
    def encode(self, texts: Union[str, List[str]], 
               batch_size: int = 32,
               show_progress_bar: bool = False) -> np.ndarray:
        """
        Encode texts to embeddings.
        
        Args:
            texts (str or List[str]): Text(s) to encode
            batch_size (int): Batch size for encoding (default: 32)
            show_progress_bar (bool): Whether to show progress bar
        
        Returns:
            np.ndarray: Embeddings of shape (n_samples, embedding_dim)
        """
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress_bar,
            convert_to_numpy=True
        )
        
        return embeddings
    
    def encode_batch(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encode a batch of texts to embeddings.
        
        Args:
            texts (List[str]): Texts to encode
            batch_size (int): Batch size for processing
        
        Returns:
            np.ndarray: Embeddings of shape (n_texts, embedding_dim)
        """
        return self.encode(texts, batch_size=batch_size, show_progress_bar=True)
    
    def encode_documents(self, documents: Dict[str, str]) -> Dict[str, np.ndarray]:
        """
        Encode a dictionary of documents.
        
        Args:
            documents (Dict[str, str]): Dictionary mapping doc_id to text
        
        Returns:
            Dict[str, np.ndarray]: Dictionary mapping doc_id to embedding
        """
        doc_ids = list(documents.keys())
        texts = list(documents.values())
        
        embeddings = self.encode_batch(texts)
        
        return {doc_id: emb for doc_id, emb in zip(doc_ids, embeddings)}
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], filepath: str) -> None:
        """
        Save embeddings to NPY file.
        
        Args:
            embeddings (Dict[str, np.ndarray]): Dictionary of embeddings
            filepath (str): Output file path
        """
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save as NPZ format to preserve dictionary structure
        np.savez(filepath, **embeddings)
        print(f"Saved {len(embeddings)} embeddings to {filepath}")
    
    def load_embeddings(self, filepath: str) -> Dict[str, np.ndarray]:
        """
        Load embeddings from NPZ file.
        
        Args:
            filepath (str): Input file path
        
        Returns:
            Dict[str, np.ndarray]: Dictionary of embeddings
        """
        data = np.load(filepath, allow_pickle=True)
        embeddings = {key: data[key] for key in data.files}
        print(f"Loaded {len(embeddings)} embeddings from {filepath}")
        return embeddings


class LLMSimilarityScorer:
    """
    Scores CV-Job match using Groq API with Llama models.
    
    Provides semantic similarity scoring for link prediction.
    
    Attributes:
        client: Groq API client
        model (str): Groq model to use
    """
    
    def __init__(self, api_key: Optional[str] = None, 
                 model: str = "llama-3.1-70b-versatile"):
        """
        Initialize LLM similarity scorer.
        
        Args:
            api_key (str): Groq API key (defaults to GROQ_API_KEY env var)
            model (str): Groq model name (default: llama-3.1-70b-versatile)
        
        Raises:
            ImportError: If groq not available
            ValueError: If API key not provided and GROQ_API_KEY not set
        """
        if not GROQ_AVAILABLE:
            raise ImportError("groq is required. Install with: pip install groq")
        
        if api_key is None:
            api_key = os.environ.get("GROQ_API_KEY")
        
        if not api_key:
            raise ValueError(
                "GROQ_API_KEY not provided. Set as argument or environment variable."
            )
        
        self.client = Groq(api_key=api_key)
        self.model = model
        print(f"Initialized Groq client with model: {model}")
    
    def score_match(self, cv_text: str, job_text: str, max_tokens: int = 10) -> float:
        """
        Score how well a CV matches a job using LLM.
        
        Args:
            cv_text (str): CV text content
            job_text (str): Job description text
            max_tokens (int): Max tokens in response
        
        Returns:
            float: Match score in [0, 1]
        
        Raises:
            Exception: If API call fails
        """
        # Truncate texts to avoid token limits
        cv_text = cv_text[:500]
        job_text = job_text[:500]
        
        prompt = f"""You are an expert recruiter. Rate how well this CV matches this Job on a scale of 0-10.

CV Summary:
{cv_text}

Job Description:
{job_text}

Consider:
- Skill overlap
- Experience level match
- Domain alignment
- Gap analysis

Respond with ONLY a number between 0-10, nothing else."""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=max_tokens
            )
            
            # Parse score
            score_text = response.choices[0].message.content.strip()
            score = float(score_text)
            
            # Normalize to [0, 1]
            return min(1.0, max(0.0, score / 10.0))
        
        except Exception as e:
            print(f"Error scoring match: {e}")
            return 0.5  # Default score on error
    
    def batch_score_matches(self, cv_job_pairs: List[Tuple[str, str]],
                           verbose: bool = True) -> List[float]:
        """
        Score multiple CV-Job pairs.
        
        Args:
            cv_job_pairs (List[Tuple[str, str]]): List of (cv_text, job_text) tuples
            verbose (bool): Whether to print progress
        
        Returns:
            List[float]: List of match scores
        """
        scores = []
        
        for i, (cv_text, job_text) in enumerate(cv_job_pairs):
            if verbose and (i + 1) % 10 == 0:
                print(f"Scored {i + 1}/{len(cv_job_pairs)}")
            
            score = self.score_match(cv_text, job_text)
            scores.append(score)
        
        return scores


class SemanticSimilarityMatrix:
    """
    Computes semantic similarity matrices between CVs and Jobs.
    
    Uses cosine similarity on embeddings.
    """
    
    @staticmethod
    def cosine_similarity(embedding1: np.ndarray, 
                         embedding2: np.ndarray) -> float:
        """
        Compute cosine similarity between two embeddings.
        
        Args:
            embedding1 (np.ndarray): First embedding
            embedding2 (np.ndarray): Second embedding
        
        Returns:
            float: Cosine similarity in [0, 1]
        """
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(embedding1, embedding2) / (norm1 * norm2))
    
    @staticmethod
    def compute_matrix(cv_embeddings: Dict[str, np.ndarray],
                      job_embeddings: Dict[str, np.ndarray]) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Compute semantic similarity matrix between CVs and Jobs.
        
        Args:
            cv_embeddings (Dict[str, np.ndarray]): CV embeddings
            job_embeddings (Dict[str, np.ndarray]): Job embeddings
        
        Returns:
            Tuple: (similarity_matrix, cv_ids, job_ids)
                - similarity_matrix: (n_cvs, n_jobs) matrix
                - cv_ids: List of CV IDs (rows)
                - job_ids: List of job IDs (columns)
        """
        cv_ids = sorted(cv_embeddings.keys())
        job_ids = sorted(job_embeddings.keys())
        
        n_cvs = len(cv_ids)
        n_jobs = len(job_ids)
        
        similarity_matrix = np.zeros((n_cvs, n_jobs))
        
        for i, cv_id in enumerate(cv_ids):
            for j, job_id in enumerate(job_ids):
                similarity = SemanticSimilarityMatrix.cosine_similarity(
                    cv_embeddings[cv_id],
                    job_embeddings[job_id]
                )
                similarity_matrix[i, j] = similarity
        
        return similarity_matrix, cv_ids, job_ids
    
    @staticmethod
    def save_matrix(similarity_matrix: np.ndarray,
                   cv_ids: List[str],
                   job_ids: List[str],
                   filepath: str) -> None:
        """
        Save similarity matrix to file.
        
        Args:
            similarity_matrix (np.ndarray): Similarity matrix
            cv_ids (List[str]): CV IDs (row labels)
            job_ids (List[str]): Job IDs (column labels)
            filepath (str): Output file path
        """
        # Create directory if needed
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        
        # Save matrix and metadata
        data = {
            'matrix': similarity_matrix,
            'cv_ids': np.array(cv_ids),
            'job_ids': np.array(job_ids)
        }
        
        np.savez(filepath, **data)
        print(f"Saved similarity matrix ({similarity_matrix.shape}) to {filepath}")
    
    @staticmethod
    def load_matrix(filepath: str) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load similarity matrix from file.
        
        Args:
            filepath (str): Input file path
        
        Returns:
            Tuple: (similarity_matrix, cv_ids, job_ids)
        """
        data = np.load(filepath)
        
        similarity_matrix = data['matrix']
        cv_ids = data['cv_ids'].tolist()
        job_ids = data['job_ids'].tolist()
        
        print(f"Loaded similarity matrix {similarity_matrix.shape} from {filepath}")
        return similarity_matrix, cv_ids, job_ids


def get_embedding_generator(model_name: str = 'all-MiniLM-L6-v2') -> EmbeddingGenerator:
    """
    Factory function to get embedding generator.
    
    Args:
        model_name (str): Model name for embeddings
    
    Returns:
        EmbeddingGenerator: Initialized generator
    """
    return EmbeddingGenerator(model_name)


def get_llm_scorer(api_key: Optional[str] = None) -> LLMSimilarityScorer:
    """
    Factory function to get LLM similarity scorer.
    
    Args:
        api_key (str): Optional Groq API key
    
    Returns:
        LLMSimilarityScorer: Initialized scorer
    """
    return LLMSimilarityScorer(api_key=api_key)


if __name__ == '__main__':
    # Test embedding generation
    print("Testing EmbeddingGenerator...")
    generator = EmbeddingGenerator()
    
    texts = [
        "Senior Python developer with 8 years of experience in cloud architecture",
        "Looking for a full-stack engineer to join our startup"
    ]
    
    embeddings = generator.encode(texts)
    print(f"Generated embeddings shape: {embeddings.shape}")
    
    # Test semantic similarity
    print("\nTesting SemanticSimilarityMatrix...")
    similarity = SemanticSimilarityMatrix.cosine_similarity(embeddings[0], embeddings[1])
    print(f"Similarity between texts: {similarity:.3f}")
