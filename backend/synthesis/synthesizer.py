"""
Document Synthesis Agent
Reads multiple document chunks and synthesizes coherent, complete answers
"""

import os
from typing import List, Dict, Optional
from dataclasses import dataclass
import logging
from openai import OpenAI

logger = logging.getLogger(__name__)

@dataclass
class SynthesisResult:
    query: str
    synthesized_answer: str
    sources_used: List[str]
    chunks_analyzed: int
    confidence_score: float

class DocumentSynthesizer:
    """
    Intelligent agent that synthesizes information from multiple document chunks
    """
    
    def __init__(self, api_key: Optional[str] = None, model: str = "gpt-4-turbo-preview"):
        """Initialize the synthesizer with OpenAI API"""
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model
        
        if self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            self.enabled = True
            logger.info(f"Document Synthesizer initialized with model: {model}")
        else:
            self.client = None
            self.enabled = False
            logger.warning("OpenAI API key not found. Synthesis features disabled.")
    
    def synthesize(
        self,
        query: str,
        chunks: List[Dict],
        max_chunks: int = 20,
        include_sources: bool = True
    ) -> SynthesisResult:
        """
        Synthesize a comprehensive answer from multiple document chunks
        
        Args:
            query: The user's question
            chunks: List of document chunks with content and metadata
            max_chunks: Maximum number of chunks to analyze
            include_sources: Whether to include source references
            
        Returns:
            SynthesisResult with complete, coherent answer
        """
        if not self.enabled:
            return self._fallback_synthesis(query, chunks)
        
        # Limit chunks to prevent token overflow
        relevant_chunks = chunks[:max_chunks]
        
        # Prepare context from chunks
        context = self._prepare_context(relevant_chunks)
        
        # Create synthesis prompt
        prompt = self._create_synthesis_prompt(query, context, include_sources)
        
        try:
            # Call OpenAI API for synthesis
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": self._get_system_prompt()},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            synthesized_answer = response.choices[0].message.content
            
            # Extract sources used
            sources = self._extract_sources(relevant_chunks)
            
            return SynthesisResult(
                query=query,
                synthesized_answer=synthesized_answer,
                sources_used=sources,
                chunks_analyzed=len(relevant_chunks),
                confidence_score=self._calculate_confidence(relevant_chunks)
            )
            
        except Exception as e:
            logger.error(f"Synthesis error: {e}")
            return self._fallback_synthesis(query, chunks)
    
    def _prepare_context(self, chunks: List[Dict]) -> str:
        """Prepare context from document chunks"""
        context_parts = []
        
        for i, chunk in enumerate(chunks, 1):
            content = chunk.get('content', '')
            doc_id = chunk.get('doc_id', f'doc_{i}')
            score = chunk.get('score', 0)
            
            # Add chunk with reference
            context_parts.append(f"[Chunk {i} - Score: {score:.2f} - ID: {doc_id}]")
            context_parts.append(content)
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def _create_synthesis_prompt(self, query: str, context: str, include_sources: bool) -> str:
        """Create the synthesis prompt"""
        prompt = f"""Based on the following document chunks, provide a comprehensive and complete answer to the question.

IMPORTANT: 
- Synthesize information from ALL relevant chunks
- Provide a COMPLETE answer, especially if information continues across chunks
- If steps or lists are mentioned, include ALL of them
- Maintain accuracy while ensuring completeness
- If chunks contain incomplete information that continues in other chunks, merge them coherently

Question: {query}

Document Chunks:
{context}

Instructions:
1. Read through ALL chunks carefully
2. Identify information that continues across chunks
3. Synthesize a complete, coherent answer
4. Include all relevant details, especially numbered steps or lists
"""

        if include_sources:
            prompt += "\n5. Reference which chunks (by number) contained key information"
        
        prompt += "\n\nProvide your comprehensive answer:"
        
        return prompt
    
    def _get_system_prompt(self) -> str:
        """Get the system prompt for the synthesis agent"""
        return """You are an expert document analyst and information synthesizer. Your role is to:
1. Read multiple document chunks that may contain fragmented information
2. Identify when information continues across chunk boundaries
3. Synthesize complete, coherent answers that merge information from multiple sources
4. Ensure no important details are lost, especially in lists, steps, or procedures
5. Maintain factual accuracy while providing comprehensive responses

You excel at recognizing when text like "This process consists of several steps:" in one chunk
is followed by the actual steps in another chunk, and you seamlessly combine them."""
    
    def _extract_sources(self, chunks: List[Dict]) -> List[str]:
        """Extract unique source documents from chunks"""
        sources = set()
        
        for chunk in chunks:
            if 'metadata' in chunk and chunk['metadata']:
                if 'file_path' in chunk['metadata']:
                    # Extract filename from path
                    file_path = chunk['metadata']['file_path']
                    filename = file_path.split('\\')[-1].split('/')[-1]
                    sources.add(filename)
                elif 'source' in chunk['metadata']:
                    sources.add(chunk['metadata']['source'])
        
        return list(sources)
    
    def _calculate_confidence(self, chunks: List[Dict]) -> float:
        """Calculate confidence score based on chunk relevance"""
        if not chunks:
            return 0.0
        
        # Average of top chunk scores
        scores = [chunk.get('score', 0) for chunk in chunks[:5]]
        return sum(scores) / len(scores) if scores else 0.0
    
    def _fallback_synthesis(self, query: str, chunks: List[Dict]) -> SynthesisResult:
        """Fallback synthesis when OpenAI is not available"""
        # Simple concatenation of top chunks
        combined_content = "\n\n---\n\n".join([
            f"[From: {chunk.get('doc_id', 'Unknown')}]\n{chunk.get('content', '')}"
            for chunk in chunks[:5]
        ])
        
        return SynthesisResult(
            query=query,
            synthesized_answer=f"Here are the most relevant sections for your query:\n\n{combined_content}",
            sources_used=self._extract_sources(chunks[:5]),
            chunks_analyzed=min(5, len(chunks)),
            confidence_score=self._calculate_confidence(chunks[:5])
        )