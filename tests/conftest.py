"""
Pytest configuration and fixtures
"""

import pytest
import tempfile
import shutil
from pathlib import Path

def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (may take longer to run)"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )

@pytest.fixture(scope="session")
def sample_documents():
    """Fixture providing sample documents for testing"""
    return [
        {
            'doc_id': 'ai_overview',
            'content': '''Artificial Intelligence Overview
            
Artificial intelligence (AI) is a branch of computer science that aims to create 
intelligent machines capable of performing tasks that typically require human intelligence. 
These tasks include visual perception, speech recognition, decision-making, and language 
translation.

Machine learning is a subset of AI that provides systems the ability to automatically 
learn and improve from experience without being explicitly programmed. Deep learning 
is a subset of machine learning that uses neural networks with multiple layers.

Applications of AI include:
- Natural language processing for chatbots and virtual assistants
- Computer vision for image recognition and autonomous vehicles
- Robotics for manufacturing and service industries
- Healthcare for medical diagnosis and drug discovery''',
            'metadata': {
                'category': 'Technology',
                'topic': 'Artificial Intelligence',
                'length': 'medium',
                'language': 'en'
            }
        },
        {
            'doc_id': 'ml_algorithms',
            'content': '''Machine Learning Algorithms Guide

Supervised Learning:
Supervised learning algorithms learn from labeled training data to make predictions 
on new, unseen data. Common algorithms include:

- Linear Regression: Predicts continuous values
- Logistic Regression: Binary classification
- Decision Trees: Tree-like model of decisions
- Random Forest: Ensemble of decision trees
- Support Vector Machines: Finds optimal decision boundary
- Neural Networks: Interconnected nodes mimicking brain neurons

Unsupervised Learning:
Unsupervised learning finds hidden patterns in data without labels:

- K-Means Clustering: Groups similar data points
- Hierarchical Clustering: Creates tree of clusters
- Principal Component Analysis (PCA): Reduces dimensionality
- Association Rules: Finds relationships between variables

Reinforcement Learning:
Agents learn through interaction with environment, receiving rewards or penalties 
for actions taken. Used in game playing, robotics, and autonomous systems.''',
            'metadata': {
                'category': 'Technology',
                'topic': 'Machine Learning',
                'length': 'long',
                'language': 'en'
            }
        },
        {
            'doc_id': 'data_science',
            'content': '''Introduction to Data Science

Data science is an interdisciplinary field that uses scientific methods, processes, 
algorithms, and systems to extract knowledge and insights from structured and 
unstructured data.

The data science process typically involves:

1. Data Collection: Gathering relevant data from various sources
2. Data Cleaning: Handling missing values, outliers, and inconsistencies  
3. Exploratory Data Analysis: Understanding data patterns and relationships
4. Feature Engineering: Creating meaningful variables for analysis
5. Model Building: Applying statistical and machine learning techniques
6. Model Evaluation: Assessing performance using appropriate metrics
7. Deployment: Implementing models in production environments

Key tools include Python, R, SQL, Tableau, and various machine learning libraries 
like scikit-learn, TensorFlow, and PyTorch.

Data scientists work across industries including finance, healthcare, retail, 
and technology to solve complex business problems using data-driven approaches.''',
            'metadata': {
                'category': 'Technology',
                'topic': 'Data Science',
                'length': 'medium',
                'language': 'en'
            }
        },
        {
            'doc_id': 'nlp_guide',
            'content': '''Natural Language Processing Fundamentals

Natural Language Processing (NLP) is a subfield of artificial intelligence 
that focuses on the interaction between computers and human language.

Core NLP Tasks:
- Tokenization: Breaking text into individual words or tokens
- Part-of-speech tagging: Identifying grammatical roles of words
- Named entity recognition: Identifying people, places, organizations
- Sentiment analysis: Determining emotional tone of text
- Machine translation: Converting text between languages
- Question answering: Understanding and responding to queries
- Text summarization: Creating concise summaries of longer texts

Modern NLP approaches use deep learning models like:
- Recurrent Neural Networks (RNNs) for sequential data
- Long Short-Term Memory (LSTM) networks for long sequences
- Transformer models like BERT and GPT for advanced understanding
- Attention mechanisms to focus on relevant parts of input

Applications include chatbots, virtual assistants, search engines, 
content moderation, and automated customer service.''',
            'metadata': {
                'category': 'Technology',
                'topic': 'Natural Language Processing',
                'length': 'medium',
                'language': 'en'
            }
        },
        {
            'doc_id': 'computer_vision',
            'content': '''Computer Vision and Image Processing

Computer vision is a field of artificial intelligence that trains computers 
to interpret and understand visual information from the world.

Key Computer Vision Tasks:
- Image classification: Categorizing images into predefined classes
- Object detection: Identifying and locating objects within images
- Semantic segmentation: Labeling each pixel with corresponding class
- Instance segmentation: Distinguishing individual object instances
- Facial recognition: Identifying or verifying person identity
- Optical character recognition (OCR): Extracting text from images

Common architectures:
- Convolutional Neural Networks (CNNs) for feature extraction
- ResNet for very deep networks with skip connections
- YOLO (You Only Look Once) for real-time object detection
- U-Net for image segmentation tasks
- GANs (Generative Adversarial Networks) for image generation

Applications span autonomous vehicles, medical imaging, surveillance systems, 
augmented reality, and quality control in manufacturing.''',
            'metadata': {
                'category': 'Technology',
                'topic': 'Computer Vision',
                'length': 'medium',
                'language': 'en'
            }
        }
    ]

@pytest.fixture
def temp_directory():
    """Fixture providing a temporary directory for tests"""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    shutil.rmtree(temp_dir)

@pytest.fixture
def sample_files(temp_directory, sample_documents):
    """Fixture creating sample files in temporary directory"""
    temp_path = Path(temp_directory)
    file_paths = []
    
    for i, doc in enumerate(sample_documents):
        file_path = temp_path / f"{doc['doc_id']}.txt"
        file_path.write_text(doc['content'])
        file_paths.append(str(file_path))
    
    return file_paths

@pytest.fixture
def test_config():
    """Fixture providing test configuration"""
    return {
        'bm25': {
            'k1': 1.2,
            'b': 0.75,
            'min_doc_freq': 1,
            'max_doc_freq': 0.9
        },
        'dense_retrieval': {
            'model_name': 'sentence-transformers/all-MiniLM-L6-v2',
            'embedding_dim': 384,
            'batch_size': 2,
            'device': 'cpu'
        },
        'cross_encoder': {
            'model_name': 'cross-encoder/ms-marco-MiniLM-L-2-v2',
            'max_candidates': 10,
            'batch_size': 2,
            'device': 'cpu'
        },
        'hybrid': {
            'use_rrv_fusion': True,
            'rrv_k': 60,
            'bm25_weight': 0.5,
            'dense_weight': 0.5,
            'final_rerank_size': 5,
            'min_score_threshold': 0.0
        },
        'chunking': {
            'strategy': 'adaptive',
            'base_chunk_size': 512,
            'overlap_percentage': 0.1,
            'min_chunk_size': 100,
            'max_chunk_size': 1000,
            'dynamic_overlap': True,
            'preserve_headers': True,
            'sentence_boundary': True,
            'dedup_threshold': 0.8
        }
    }