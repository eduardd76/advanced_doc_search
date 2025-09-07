"""
Advanced structure-aware chunking system
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

@dataclass
class DocumentChunk:
    """Represents a document chunk with metadata"""
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    start_position: int
    end_position: int
    section_header: Optional[str] = None
    page_number: Optional[int] = None
    word_count: int = 0
    overlap_with: List[str] = None
    
    def __post_init__(self):
        if self.overlap_with is None:
            self.overlap_with = []
        if self.word_count == 0:
            self.word_count = len(self.content.split())

@dataclass
class DocumentStructure:
    """Represents detected document structure"""
    sections: List[Dict[str, Any]]
    headers: List[Dict[str, Any]]
    page_breaks: List[int]
    table_of_contents: List[Dict[str, Any]]
    footnotes: List[Dict[str, Any]]

class StructureDetector:
    """Detects document structure for better chunking"""
    
    def __init__(self):
        # Regex patterns for structure detection
        self.header_patterns = [
            r'^(#{1,6})\s+(.+)$',  # Markdown headers
            r'^([IVX]+\.|[\d]+\.)\s+(.+)$',  # Numbered sections
            r'^([A-Z][A-Z\s]+)$',  # ALL CAPS headers
            r'^(.{1,50})\n[=-]{3,}$',  # Underlined headers
            r'^\s*(\d+\.)+\s*(.+)$',  # Nested numbering (1.1, 1.1.1)
        ]
        
        self.page_break_patterns = [
            r'\n\s*\f\s*\n',  # Form feed
            r'\n\s*-{3,}\s*Page\s*\d+\s*-{3,}\s*\n',  # Page headers
            r'\n\s*Page\s*\d+\s*\n',  # Simple page numbers
        ]
        
        self.footnote_patterns = [
            r'^\[\d+\]\s+(.+)$',  # [1] footnote
            r'^\d+\.\s+(.+)$',    # 1. footnote (at document end)
        ]
    
    def detect_structure(self, text: str, filename: Optional[str] = None) -> DocumentStructure:
        """Detect document structure"""
        sections = []
        headers = []
        page_breaks = []
        table_of_contents = []
        footnotes = []
        
        lines = text.split('\n')
        current_position = 0
        current_page = 1
        
        for i, line in enumerate(lines):
            line_start = current_position
            line_end = current_position + len(line) + 1  # +1 for newline
            current_position = line_end
            
            # Detect headers
            header_info = self._detect_header(line, i, line_start, line_end)
            if header_info:
                headers.append(header_info)
                
                # Create section from previous header to this one
                if len(headers) > 1:
                    prev_header = headers[-2]
                    section = {
                        'title': prev_header['text'],
                        'level': prev_header['level'],
                        'start_position': prev_header['end_position'],
                        'end_position': line_start,
                        'start_line': prev_header['line_number'],
                        'end_line': i - 1,
                        'page_number': prev_header.get('page_number', current_page)
                    }
                    sections.append(section)
            
            # Detect page breaks
            if self._is_page_break(line):
                page_breaks.append(current_position)
                current_page += 1
            
            # Detect footnotes
            footnote_info = self._detect_footnote(line, i, line_start, line_end)
            if footnote_info:
                footnotes.append(footnote_info)
            
            # Detect TOC entries
            toc_info = self._detect_toc_entry(line, i)
            if toc_info:
                table_of_contents.append(toc_info)
        
        # Add final section if there are headers
        if headers:
            last_header = headers[-1]
            final_section = {
                'title': last_header['text'],
                'level': last_header['level'],
                'start_position': last_header['end_position'],
                'end_position': len(text),
                'start_line': last_header['line_number'],
                'end_line': len(lines) - 1,
                'page_number': last_header.get('page_number', current_page)
            }
            sections.append(final_section)
        
        return DocumentStructure(
            sections=sections,
            headers=headers,
            page_breaks=page_breaks,
            table_of_contents=table_of_contents,
            footnotes=footnotes
        )
    
    def _detect_header(self, line: str, line_number: int, start_pos: int, end_pos: int) -> Optional[Dict[str, Any]]:
        """Detect if line is a header"""
        line_stripped = line.strip()
        
        if not line_stripped or len(line_stripped) > 200:
            return None
        
        for pattern in self.header_patterns:
            match = re.match(pattern, line_stripped, re.MULTILINE)
            if match:
                groups = match.groups()
                
                if pattern.startswith('^(#{1,6})'):  # Markdown header
                    level = len(groups[0])
                    text = groups[1]
                elif pattern.startswith('^([IVX]+\\.|[\\d]+\\.)'):  # Numbered section
                    level = 1
                    text = groups[1]
                elif pattern.startswith('^([A-Z][A-Z\\s]+)'):  # ALL CAPS
                    level = 1
                    text = groups[0]
                elif 'underlined' in pattern:  # Underlined header
                    level = 2
                    text = groups[0]
                else:  # Nested numbering
                    level = groups[0].count('.') + 1
                    text = groups[1] if len(groups) > 1 else groups[0]
                
                return {
                    'text': text.strip(),
                    'level': min(level, 6),  # Cap at level 6
                    'line_number': line_number,
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'pattern_type': pattern,
                    'original_line': line
                }
        
        # Heuristic: Short lines that look like headers
        if (len(line_stripped) < 100 and 
            line_stripped.isupper() and 
            len(line_stripped.split()) <= 8):
            return {
                'text': line_stripped,
                'level': 2,
                'line_number': line_number,
                'start_position': start_pos,
                'end_position': end_pos,
                'pattern_type': 'heuristic_caps',
                'original_line': line
            }
        
        return None
    
    def _is_page_break(self, line: str) -> bool:
        """Check if line indicates a page break"""
        for pattern in self.page_break_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                return True
        return False
    
    def _detect_footnote(self, line: str, line_number: int, start_pos: int, end_pos: int) -> Optional[Dict[str, Any]]:
        """Detect footnote"""
        line_stripped = line.strip()
        
        for pattern in self.footnote_patterns:
            match = re.match(pattern, line_stripped)
            if match:
                return {
                    'text': match.group(1) if len(match.groups()) > 0 else line_stripped,
                    'line_number': line_number,
                    'start_position': start_pos,
                    'end_position': end_pos,
                    'pattern_type': pattern
                }
        return None
    
    def _detect_toc_entry(self, line: str, line_number: int) -> Optional[Dict[str, Any]]:
        """Detect table of contents entry"""
        # Simple TOC detection (dots followed by page number)
        toc_pattern = r'^(.+?)\.{3,}\s*(\d+)$'
        match = re.match(toc_pattern, line.strip())
        
        if match:
            return {
                'title': match.group(1).strip(),
                'page_number': int(match.group(2)),
                'line_number': line_number
            }
        return None

class AdvancedChunker:
    """
    Advanced structure-aware chunking system that:
    - Respects document structure
    - Uses dynamic overlapping
    - Preserves context
    - Handles various document types
    """
    
    def __init__(
        self,
        strategy: str = "structure_aware",
        base_chunk_size: int = 512,
        overlap_percentage: int = 15,
        min_chunk_size: int = 50,
        max_chunk_size: int = 1024,
        dynamic_overlap: bool = True,
        preserve_headers: bool = True,
        sentence_boundary: bool = True,
        dedup_threshold: float = 0.9
    ):
        self.strategy = strategy
        self.base_chunk_size = base_chunk_size
        self.overlap_percentage = overlap_percentage
        self.min_chunk_size = min_chunk_size
        self.max_chunk_size = max_chunk_size
        self.dynamic_overlap = dynamic_overlap
        self.preserve_headers = preserve_headers
        self.sentence_boundary = sentence_boundary
        self.dedup_threshold = dedup_threshold
        
        # Initialize structure detector
        self.structure_detector = StructureDetector()
        
        # Sentence boundary patterns
        self.sentence_endings = re.compile(r'[.!?]+\s+')
        
        # Cache for duplicate detection
        self._content_hashes = set()
    
    def chunk_document(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any] = None
    ) -> List[DocumentChunk]:
        """Chunk document using the specified strategy"""
        if metadata is None:
            metadata = {}
        
        if not text.strip():
            logger.warning(f"Empty text for document {doc_id}")
            return []
        
        # Detect document structure
        structure = self.structure_detector.detect_structure(text, doc_id)
        
        # Choose chunking strategy
        if self.strategy == "structure_aware":
            chunks = self._chunk_structure_aware(text, doc_id, metadata, structure)
        elif self.strategy == "fixed_size":
            chunks = self._chunk_fixed_size(text, doc_id, metadata)
        elif self.strategy == "sentence_boundary":
            chunks = self._chunk_sentence_boundary(text, doc_id, metadata)
        elif self.strategy == "paragraph_based":
            chunks = self._chunk_paragraph_based(text, doc_id, metadata)
        else:
            logger.warning(f"Unknown chunking strategy: {self.strategy}. Using structure_aware.")
            chunks = self._chunk_structure_aware(text, doc_id, metadata, structure)
        
        # Apply deduplication
        chunks = self._deduplicate_chunks(chunks)
        
        # Add overlap information
        chunks = self._add_overlap_info(chunks)
        
        logger.debug(f"Chunked document {doc_id} into {len(chunks)} chunks")
        return chunks
    
    def _chunk_structure_aware(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any],
        structure: DocumentStructure
    ) -> List[DocumentChunk]:
        """Chunk document based on detected structure"""
        chunks = []
        
        if not structure.sections:
            # No structure detected, fall back to sentence boundary chunking
            return self._chunk_sentence_boundary(text, doc_id, metadata)
        
        for i, section in enumerate(structure.sections):
            section_text = text[section['start_position']:section['end_position']]
            section_header = section.get('title', '')
            page_number = section.get('page_number')
            
            if len(section_text.strip()) < self.min_chunk_size:
                continue
            
            # If section is small enough, use as single chunk
            if len(section_text.split()) <= self.base_chunk_size:
                chunk_content = section_text.strip()
                if self.preserve_headers and section_header:
                    chunk_content = f"{section_header}\n\n{chunk_content}"
                
                chunk = self._create_chunk(
                    doc_id=doc_id,
                    chunk_index=len(chunks),
                    content=chunk_content,
                    metadata={
                        **metadata,
                        'section_title': section_header,
                        'section_level': section.get('level', 1),
                        'page_number': page_number
                    },
                    start_position=section['start_position'],
                    end_position=section['end_position'],
                    section_header=section_header,
                    page_number=page_number
                )
                chunks.append(chunk)
            else:
                # Split large section into smaller chunks
                section_chunks = self._split_large_section(
                    section_text, doc_id, len(chunks), metadata, 
                    section_header, page_number, section['start_position']
                )
                chunks.extend(section_chunks)
        
        return chunks
    
    def _split_large_section(
        self,
        section_text: str,
        doc_id: str,
        start_chunk_index: int,
        metadata: Dict[str, Any],
        section_header: Optional[str],
        page_number: Optional[int],
        section_start_pos: int
    ) -> List[DocumentChunk]:
        """Split a large section into smaller chunks"""
        chunks = []
        
        # Split by sentences first
        sentences = self._split_into_sentences(section_text)
        
        current_chunk = ""
        current_position = 0
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            potential_chunk = current_chunk + " " + sentence if current_chunk else sentence
            potential_word_count = len(potential_chunk.split())
            
            if potential_word_count <= self.base_chunk_size:
                current_chunk = potential_chunk
            else:
                # Save current chunk if it meets minimum size
                if len(current_chunk.split()) >= self.min_chunk_size:
                    chunk_content = current_chunk.strip()
                    if self.preserve_headers and section_header and chunks == []:
                        # Add header only to first chunk of section
                        chunk_content = f"{section_header}\n\n{chunk_content}"
                    
                    chunk = self._create_chunk(
                        doc_id=doc_id,
                        chunk_index=start_chunk_index + len(chunks),
                        content=chunk_content,
                        metadata={
                            **metadata,
                            'section_title': section_header,
                            'page_number': page_number,
                            'chunk_in_section': len(chunks)
                        },
                        start_position=section_start_pos + current_position,
                        end_position=section_start_pos + current_position + len(current_chunk),
                        section_header=section_header,
                        page_number=page_number
                    )
                    chunks.append(chunk)
                
                # Start new chunk with current sentence
                current_chunk = sentence
                current_position += len(potential_chunk) - len(sentence)
        
        # Add remaining content as final chunk
        if current_chunk.strip() and len(current_chunk.split()) >= self.min_chunk_size:
            chunk_content = current_chunk.strip()
            
            chunk = self._create_chunk(
                doc_id=doc_id,
                chunk_index=start_chunk_index + len(chunks),
                content=chunk_content,
                metadata={
                    **metadata,
                    'section_title': section_header,
                    'page_number': page_number,
                    'chunk_in_section': len(chunks)
                },
                start_position=section_start_pos + current_position,
                end_position=section_start_pos + current_position + len(current_chunk),
                section_header=section_header,
                page_number=page_number
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_fixed_size(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk document into fixed-size chunks"""
        chunks = []
        words = text.split()
        
        overlap_size = int(self.base_chunk_size * self.overlap_percentage / 100)
        
        for i in range(0, len(words), self.base_chunk_size - overlap_size):
            chunk_words = words[i:i + self.base_chunk_size]
            chunk_content = " ".join(chunk_words)
            
            if len(chunk_words) < self.min_chunk_size:
                continue
            
            chunk = self._create_chunk(
                doc_id=doc_id,
                chunk_index=len(chunks),
                content=chunk_content,
                metadata=metadata,
                start_position=0,  # Approximate
                end_position=0     # Approximate
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_sentence_boundary(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk document respecting sentence boundaries"""
        chunks = []
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_word_count = 0
        
        for sentence in sentences:
            sentence_word_count = len(sentence.split())
            
            # Check if adding this sentence would exceed chunk size
            if current_word_count + sentence_word_count <= self.base_chunk_size:
                current_chunk += " " + sentence if current_chunk else sentence
                current_word_count += sentence_word_count
            else:
                # Save current chunk
                if current_word_count >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        doc_id=doc_id,
                        chunk_index=len(chunks),
                        content=current_chunk.strip(),
                        metadata=metadata,
                        start_position=0,  # Approximate
                        end_position=0     # Approximate
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk = sentence
                current_word_count = sentence_word_count
        
        # Add remaining content
        if current_chunk.strip() and current_word_count >= self.min_chunk_size:
            chunk = self._create_chunk(
                doc_id=doc_id,
                chunk_index=len(chunks),
                content=current_chunk.strip(),
                metadata=metadata,
                start_position=0,  # Approximate
                end_position=0     # Approximate
            )
            chunks.append(chunk)
        
        return chunks
    
    def _chunk_paragraph_based(
        self,
        text: str,
        doc_id: str,
        metadata: Dict[str, Any]
    ) -> List[DocumentChunk]:
        """Chunk document by paragraphs, combining small ones"""
        chunks = []
        paragraphs = re.split(r'\n\s*\n', text)
        
        current_chunk = ""
        current_word_count = 0
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
            
            paragraph_word_count = len(paragraph.split())
            
            # Check if adding this paragraph would exceed chunk size
            if current_word_count + paragraph_word_count <= self.base_chunk_size:
                current_chunk += "\n\n" + paragraph if current_chunk else paragraph
                current_word_count += paragraph_word_count
            else:
                # Save current chunk
                if current_word_count >= self.min_chunk_size:
                    chunk = self._create_chunk(
                        doc_id=doc_id,
                        chunk_index=len(chunks),
                        content=current_chunk.strip(),
                        metadata=metadata,
                        start_position=0,  # Approximate
                        end_position=0     # Approximate
                    )
                    chunks.append(chunk)
                
                # Start new chunk
                current_chunk = paragraph
                current_word_count = paragraph_word_count
        
        # Add remaining content
        if current_chunk.strip() and current_word_count >= self.min_chunk_size:
            chunk = self._create_chunk(
                doc_id=doc_id,
                chunk_index=len(chunks),
                content=current_chunk.strip(),
                metadata=metadata,
                start_position=0,  # Approximate
                end_position=0     # Approximate
            )
            chunks.append(chunk)
        
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences"""
        if not self.sentence_boundary:
            return [text]
        
        # Simple sentence splitting
        sentences = self.sentence_endings.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        return sentences
    
    def _create_chunk(
        self,
        doc_id: str,
        chunk_index: int,
        content: str,
        metadata: Dict[str, Any],
        start_position: int,
        end_position: int,
        section_header: Optional[str] = None,
        page_number: Optional[int] = None
    ) -> DocumentChunk:
        """Create a DocumentChunk object"""
        chunk_id = f"{doc_id}#{chunk_index}"
        
        return DocumentChunk(
            chunk_id=chunk_id,
            content=content,
            metadata=metadata,
            start_position=start_position,
            end_position=end_position,
            section_header=section_header,
            page_number=page_number,
            word_count=len(content.split())
        )
    
    def _deduplicate_chunks(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Remove near-duplicate chunks"""
        if self.dedup_threshold >= 1.0:
            return chunks
        
        deduplicated = []
        content_hashes = set()
        
        for chunk in chunks:
            # Create content hash
            content_hash = hashlib.md5(chunk.content.encode()).hexdigest()
            
            # Check for exact duplicates
            if content_hash in content_hashes:
                continue
            
            # Check for near-duplicates
            is_duplicate = False
            normalized_content = self._normalize_for_dedup(chunk.content)
            
            for existing_chunk in deduplicated:
                existing_normalized = self._normalize_for_dedup(existing_chunk.content)
                similarity = self._calculate_similarity(normalized_content, existing_normalized)
                
                if similarity > self.dedup_threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                deduplicated.append(chunk)
                content_hashes.add(content_hash)
        
        logger.debug(f"Deduplicated {len(chunks)} -> {len(deduplicated)} chunks")
        return deduplicated
    
    def _normalize_for_dedup(self, text: str) -> str:
        """Normalize text for deduplication comparison"""
        # Remove extra whitespace and convert to lowercase
        normalized = re.sub(r'\s+', ' ', text.lower().strip())
        return normalized
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        if not text1 or not text2:
            return 0.0
        
        # Simple Jaccard similarity on words
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        if not union:
            return 0.0
        
        return len(intersection) / len(union)
    
    def _add_overlap_info(self, chunks: List[DocumentChunk]) -> List[DocumentChunk]:
        """Add overlap information between chunks"""
        if not self.dynamic_overlap:
            return chunks
        
        for i, chunk in enumerate(chunks):
            overlaps = []
            
            # Check overlap with previous chunk
            if i > 0:
                prev_chunk = chunks[i-1]
                overlap_ratio = self._calculate_overlap_ratio(chunk.content, prev_chunk.content)
                if overlap_ratio > 0.1:  # 10% threshold
                    overlaps.append(prev_chunk.chunk_id)
            
            # Check overlap with next chunk
            if i < len(chunks) - 1:
                next_chunk = chunks[i+1]
                overlap_ratio = self._calculate_overlap_ratio(chunk.content, next_chunk.content)
                if overlap_ratio > 0.1:  # 10% threshold
                    overlaps.append(next_chunk.chunk_id)
            
            chunk.overlap_with = overlaps
        
        return chunks
    
    def _calculate_overlap_ratio(self, text1: str, text2: str) -> float:
        """Calculate overlap ratio between two texts"""
        words1 = set(text1.split())
        words2 = set(text2.split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        return len(intersection) / min(len(words1), len(words2))
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get chunker statistics"""
        return {
            "strategy": self.strategy,
            "base_chunk_size": self.base_chunk_size,
            "overlap_percentage": self.overlap_percentage,
            "min_chunk_size": self.min_chunk_size,
            "max_chunk_size": self.max_chunk_size,
            "dynamic_overlap": self.dynamic_overlap,
            "preserve_headers": self.preserve_headers,
            "sentence_boundary": self.sentence_boundary,
            "dedup_threshold": self.dedup_threshold
        }