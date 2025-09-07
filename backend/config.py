"""
Configuration management for Advanced Document Search
"""

import yaml
import os
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)

@dataclass
class ServerConfig:
    host: str = "0.0.0.0"
    port: int = 8000
    debug: bool = False
    reload: bool = False

@dataclass
class StorageConfig:
    base_dir: str = "./data"
    indexes_dir: str = "./data/indexes"
    models_dir: str = "./data/models"
    cache_dir: str = "./data/cache"
    temp_dir: str = "./data/temp"

@dataclass
class BM25Config:
    k1: float = 1.6
    b: float = 0.75
    epsilon: float = 0.25
    min_doc_freq: int = 3
    max_doc_freq: float = 0.8
    hash_features: int = 262144
    weight_precision: str = "float16"
    use_idf: bool = True

@dataclass
class DenseRetrievalConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dim: int = 384
    batch_size: int = 32
    normalize_embeddings: bool = True
    device: str = "cpu"
    faiss_index_type: str = "IndexFlatIP"

@dataclass
class CrossEncoderConfig:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    max_candidates: int = 100
    batch_size: int = 16
    device: str = "cpu"

@dataclass
class HybridConfig:
    use_rrv_fusion: bool = True
    rrv_k: int = 60
    bm25_weight: float = 0.5
    dense_weight: float = 0.5
    final_rerank_size: int = 50
    min_score_threshold: float = 0.1

@dataclass
class ChunkingConfig:
    strategy: str = "structure_aware"
    base_chunk_size: int = 1024
    overlap_percentage: int = 25
    min_chunk_size: int = 50
    max_chunk_size: int = 2048
    dynamic_overlap: bool = True
    preserve_headers: bool = True
    sentence_boundary: bool = True
    dedup_threshold: float = 0.9

@dataclass
class QueryProcessingConfig:
    enable_acronym_expansion: bool = True
    enable_synonym_expansion: bool = True
    enable_query_rewrite: bool = False
    max_query_length: int = 512
    min_query_length: int = 3
    stopwords_removal: bool = True
    lemmatization: bool = True
    case_sensitive: bool = False

class Config:
    """Main configuration class"""
    
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            # Default config path
            config_path = Path(__file__).parent.parent / "configs" / "config.yaml"
        
        self.config_path = Path(config_path)
        self._config_data = self._load_config()
        
        # Initialize configuration sections
        self.server = self._create_config_object(ServerConfig, "server")
        self.storage = self._create_config_object(StorageConfig, "storage")
        self.bm25 = self._create_config_object(BM25Config, "bm25")
        self.dense_retrieval = self._create_config_object(DenseRetrievalConfig, "dense_retrieval")
        self.cross_encoder = self._create_config_object(CrossEncoderConfig, "cross_encoder")
        self.hybrid = self._create_config_object(HybridConfig, "hybrid")
        self.chunking = self._create_config_object(ChunkingConfig, "chunking")
        self.query_processing = self._create_config_object(QueryProcessingConfig, "query_processing")
        
        # Ensure directories exist
        self._create_directories()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r', encoding='utf-8') as f:
                    return yaml.safe_load(f) or {}
            else:
                logger.warning(f"Config file not found: {self.config_path}. Using defaults.")
                return {}
        except Exception as e:
            logger.error(f"Error loading config: {e}. Using defaults.")
            return {}
    
    def _create_config_object(self, config_class, section_name: str):
        """Create configuration object from YAML section"""
        section_data = self._config_data.get(section_name, {})
        try:
            return config_class(**section_data)
        except TypeError as e:
            logger.warning(f"Invalid config for {section_name}: {e}. Using defaults.")
            return config_class()
    
    def _create_directories(self):
        """Create necessary directories"""
        directories = [
            self.storage.base_dir,
            self.storage.indexes_dir,
            self.storage.models_dir,
            self.storage.cache_dir,
            self.storage.temp_dir,
            "logs"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value by dot notation key"""
        keys = key.split('.')
        value = self._config_data
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def update_config(self, updates: Dict[str, Any]):
        """Update configuration with new values"""
        def deep_update(base_dict, update_dict):
            for key, value in update_dict.items():
                if key in base_dict and isinstance(base_dict[key], dict) and isinstance(value, dict):
                    deep_update(base_dict[key], value)
                else:
                    base_dict[key] = value
        
        deep_update(self._config_data, updates)
        
        # Recreate config objects
        self.server = self._create_config_object(ServerConfig, "server")
        self.storage = self._create_config_object(StorageConfig, "storage")
        self.bm25 = self._create_config_object(BM25Config, "bm25")
        self.dense_retrieval = self._create_config_object(DenseRetrievalConfig, "dense_retrieval")
        self.cross_encoder = self._create_config_object(CrossEncoderConfig, "cross_encoder")
        self.hybrid = self._create_config_object(HybridConfig, "hybrid")
        self.chunking = self._create_config_object(ChunkingConfig, "chunking")
        self.query_processing = self._create_config_object(QueryProcessingConfig, "query_processing")

# Global config instance
config = Config()

def get_config() -> Config:
    """Get global configuration instance"""
    return config

def reload_config():
    """Reload configuration from file"""
    global config
    config = Config()
    return config