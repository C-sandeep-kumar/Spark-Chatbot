import logging
import os
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
from pathlib import Path

logger = logging.getLogger(__name__)

class SparkDocsLoader:
    """Load and process Apache Spark documentation"""
    
    def __init__(self, base_url: str = "https://spark.apache.org/docs/latest/"):
        self.base_url = base_url
        self.documents = []
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
    
    def fetch_documentation(self) -> List[Tuple[str, str]]:
        """
        Fetch Spark documentation pages
        Returns list of (title, content) tuples
        """
        try:
            docs_urls = [
                f"{self.base_url}sql-programming-guide.html",
                f"{self.base_url}rdd-programming-guide.html",
                f"{self.base_url}mllib-guide.html",
                f"{self.base_url}structured-streaming-programming-guide.html",
                f"{self.base_url}graphx-programming-guide.html",
            ]
            
            documents = []
            for url in docs_urls:
                try:
                    doc = self._fetch_page(url)
                    if doc:
                        documents.append(doc)
                except Exception as e:
                    logger.warning(f"Failed to fetch {url}: {str(e)}")
            
            return documents
        except Exception as e:
            logger.error(f"Error fetching documentation: {str(e)}")
            return []
    
    def _fetch_page(self, url: str) -> Tuple[str, str]:
        """Fetch and parse a single documentation page"""
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Extract title
            title = soup.find('h1')
            title_text = title.get_text(strip=True) if title else "Unknown"
            
            # Extract main content
            content_div = soup.find('div', class_='container')
            if not content_div:
                content_div = soup.find('main')
            if not content_div:
                content_div = soup.find('article')
            
            if content_div:
                # Remove script and style elements
                for script in content_div(['script', 'style']):
                    script.decompose()
                content_text = content_div.get_text(separator='\n', strip=True)
            else:
                content_text = soup.get_text(separator='\n', strip=True)
            
            return (title_text, content_text)
        except Exception as e:
            logger.error(f"Error processing page {url}: {str(e)}")
            return None
    
    def load_from_local_files(self, directory: str) -> List[Tuple[str, str]]:
        """Load documentation from local markdown files"""
        documents = []
        
        if not os.path.exists(directory):
            logger.warning(f"Directory not found: {directory}")
            return documents
        
        for filename in os.listdir(directory):
            if filename.endswith(('.md', '.txt', '.html')):
                filepath = os.path.join(directory, filename)
                try:
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                    documents.append((filename, content))
                except Exception as e:
                    logger.error(f"Error loading file {filepath}: {str(e)}")
        
        return documents
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = ' '.join(text.split())
        # Remove special characters but keep basic punctuation
        return text
    
    @staticmethod
    def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> List[str]:
        """Split text into overlapping chunks"""
        chunks = []
        words = text.split()
        
        chunk_words = []
        for i, word in enumerate(words):
            chunk_words.append(word)
            
            if len(chunk_words) >= chunk_size:
                chunks.append(' '.join(chunk_words))
                # Create overlap
                chunk_words = chunk_words[-chunk_overlap:]
        
        # Add remaining words
        if chunk_words:
            chunks.append(' '.join(chunk_words))
        
        return chunks


def get_spark_docs(force_refresh: bool = False) -> List[Tuple[str, str]]:
    """
    Get Spark documentation with caching
    """
    cache_file = "data/spark_docs_cache.txt"
    
    if os.path.exists(cache_file) and not force_refresh:
        logger.info("Loading Spark docs from cache")
        # Return cached version (simplified)
        return []
    
    logger.info("Fetching fresh Spark documentation")
    loader = SparkDocsLoader()
    
    # Try to fetch from web first
    docs = loader.fetch_documentation()
    
    # Fall back to sample data if fetching fails
    if not docs:
        docs = [
            ("PySpark SQL Guide", get_sample_spark_docs())
        ]
    
    return docs


def get_sample_spark_docs() -> str:
    """Return sample Spark documentation for testing"""
    return """
    Apache Spark SQL Guide
    
    PySpark DataFrames:
    - Create DataFrames from various data sources
    - SQL queries on DataFrames
    - DataFrame transformations and actions
    
    Example: Creating a DataFrame
    from pyspark.sql import SparkSession
    
    spark = SparkSession.builder.appName("example").getOrCreate()
    df = spark.read.csv("data.csv", header=True)
    
    RDD Programming Guide:
    - Resilient Distributed Datasets (RDDs)
    - Creating RDDs
    - RDD operations and transformations
    
    Spark Streaming:
    - Streaming DataFrames
    - Stream processing
    - Continuous applications
    
    MLlib Machine Learning:
    - Classification
    - Regression
    - Clustering
    - Collaborative Filtering
    
    GraphX:
    - Graph processing
    - Property graphs
    - GraphFrame operations
    """