"""
Tests for Advanced Document Processor
"""

import pytest
import tempfile
import os
from pathlib import Path
import sys

# Add backend to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'backend'))

from processing.document_processor import AdvancedDocumentProcessor

class TestAdvancedDocumentProcessor:
    def setup_method(self):
        """Setup test fixtures"""
        self.processor = AdvancedDocumentProcessor(
            enable_ocr=False,  # Disable OCR for testing
            remove_boilerplate=True,
            detect_language=True
        )

    def test_processor_initialization(self):
        """Test document processor initialization"""
        assert self.processor.enable_ocr == False
        assert self.processor.remove_boilerplate == True
        assert self.processor.detect_language == True

    def test_supported_formats(self):
        """Test getting supported file formats"""
        formats = self.processor.get_supported_formats()
        
        expected_formats = ['.txt', '.md', '.pdf', '.docx', '.epub', '.rtf', '.html']
        for fmt in expected_formats:
            assert fmt in formats

    def test_text_file_processing(self):
        """Test processing plain text files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            test_content = "This is a test document.\n\nIt contains multiple paragraphs.\nAnd various formatting."
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                doc = self.processor.process_file(temp_file.name)
                
                assert doc is not None
                assert doc.content.strip() != ""
                assert doc.file_path == temp_file.name
                assert doc.doc_id == Path(temp_file.name).name
                assert 'file_type' in doc.metadata
                assert doc.metadata['file_type'] == 'txt'
                
            finally:
                os.unlink(temp_file.name)

    def test_markdown_file_processing(self):
        """Test processing Markdown files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as temp_file:
            test_content = """# Test Document

This is a **markdown** document with:

- Lists
- *Formatting*
- `Code blocks`

## Section 2

More content here."""
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                doc = self.processor.process_file(temp_file.name)
                
                assert doc is not None
                assert doc.content.strip() != ""
                assert 'Test Document' in doc.content
                assert doc.metadata['file_type'] == 'md'
                
            finally:
                os.unlink(temp_file.name)

    def test_html_file_processing(self):
        """Test processing HTML files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as temp_file:
            test_content = """<!DOCTYPE html>
<html>
<head>
    <title>Test Document</title>
</head>
<body>
    <h1>Main Title</h1>
    <p>This is a paragraph with <strong>bold text</strong>.</p>
    <ul>
        <li>Item 1</li>
        <li>Item 2</li>
    </ul>
</body>
</html>"""
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                doc = self.processor.process_file(temp_file.name)
                
                assert doc is not None
                assert doc.content.strip() != ""
                assert 'Main Title' in doc.content
                assert 'Item 1' in doc.content
                assert doc.metadata['file_type'] == 'html'
                
            finally:
                os.unlink(temp_file.name)

    def test_directory_processing(self):
        """Test processing a directory of files"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / 'test1.txt').write_text('Content of test file 1.')
            (temp_path / 'test2.md').write_text('# Test File 2\n\nContent here.')
            (temp_path / 'test3.html').write_text('<html><body><h1>Test 3</h1></body></html>')
            
            # Create subdirectory
            sub_dir = temp_path / 'subdir'
            sub_dir.mkdir()
            (sub_dir / 'test4.txt').write_text('Content of test file 4.')
            
            # Process directory recursively
            docs = self.processor.process_directory(
                str(temp_path),
                recursive=True,
                file_extensions=['.txt', '.md', '.html']
            )
            
            assert len(docs) == 4  # Should find all 4 files
            
            # Check that all files were processed
            file_names = [Path(doc.file_path).name for doc in docs]
            assert 'test1.txt' in file_names
            assert 'test2.md' in file_names
            assert 'test3.html' in file_names
            assert 'test4.txt' in file_names

    def test_directory_processing_non_recursive(self):
        """Test processing a directory without recursion"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create test files
            (temp_path / 'test1.txt').write_text('Content of test file 1.')
            (temp_path / 'test2.txt').write_text('Content of test file 2.')
            
            # Create subdirectory with file
            sub_dir = temp_path / 'subdir'
            sub_dir.mkdir()
            (sub_dir / 'test3.txt').write_text('Content of test file 3.')
            
            # Process directory non-recursively
            docs = self.processor.process_directory(
                str(temp_path),
                recursive=False,
                file_extensions=['.txt']
            )
            
            assert len(docs) == 2  # Should only find files in root directory

    def test_file_extension_filtering(self):
        """Test filtering by file extensions"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Create files with different extensions
            (temp_path / 'test.txt').write_text('Text content')
            (temp_path / 'test.md').write_text('# Markdown content')
            (temp_path / 'test.log').write_text('Log content')
            (temp_path / 'test.html').write_text('<html>HTML content</html>')
            
            # Process with specific extensions
            docs = self.processor.process_directory(
                str(temp_path),
                file_extensions=['.txt', '.md']
            )
            
            assert len(docs) == 2
            extensions = [Path(doc.file_path).suffix for doc in docs]
            assert '.txt' in extensions
            assert '.md' in extensions
            assert '.log' not in extensions
            assert '.html' not in extensions

    def test_content_cleaning(self):
        """Test content cleaning functionality"""
        # Test with content that has extra whitespace and formatting
        test_content = """


        This    is   a   test   document.


        It has extra    whitespace and     formatting.

        Multiple     consecutive    spaces.


        """
        
        cleaned = self.processor._clean_content(test_content)
        
        # Should remove extra whitespace and normalize
        assert cleaned.count('\n\n') <= 1  # No more than one consecutive empty line
        assert '    ' not in cleaned  # No multiple consecutive spaces
        assert cleaned.strip() != ""
        assert 'This is a test document.' in cleaned

    def test_language_detection(self):
        """Test language detection functionality"""
        processor_with_lang = AdvancedDocumentProcessor(detect_language=True)
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            # English content
            test_content = "This is an English document with multiple sentences. It contains various words and phrases."
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                doc = processor_with_lang.process_file(temp_file.name)
                
                assert 'language' in doc.metadata
                # Language detection might not be 100% accurate for short texts
                assert doc.metadata['language'] is not None
                
            finally:
                os.unlink(temp_file.name)

    def test_error_handling_invalid_file(self):
        """Test error handling for invalid files"""
        # Try to process a non-existent file
        doc = self.processor.process_file("non_existent_file.txt")
        assert doc is None

    def test_error_handling_unsupported_format(self):
        """Test error handling for unsupported file formats"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.unknown', delete=False) as temp_file:
            temp_file.write("Some content")
            temp_file.flush()
            
            try:
                doc = self.processor.process_file(temp_file.name)
                # Should either return None or handle gracefully
                if doc is not None:
                    assert doc.content is not None
                    
            finally:
                os.unlink(temp_file.name)

    def test_metadata_extraction(self):
        """Test metadata extraction from documents"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            test_content = "Sample document content for metadata testing."
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                doc = self.processor.process_file(temp_file.name)
                
                # Check standard metadata fields
                assert 'file_type' in doc.metadata
                assert 'file_size' in doc.metadata
                assert 'processing_time' in doc.metadata
                assert 'word_count' in doc.metadata
                assert 'character_count' in doc.metadata
                
                # Check values
                assert doc.metadata['file_type'] == 'txt'
                assert doc.metadata['file_size'] > 0
                assert doc.metadata['word_count'] > 0
                assert doc.metadata['character_count'] > 0
                
            finally:
                os.unlink(temp_file.name)

    def test_empty_file_handling(self):
        """Test handling of empty files"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            # Write empty content
            temp_file.write("")
            temp_file.flush()
            
            try:
                doc = self.processor.process_file(temp_file.name)
                
                # Should handle empty files gracefully
                if doc is not None:
                    assert doc.content == "" or doc.content.strip() == ""
                    assert doc.metadata['word_count'] == 0
                    
            finally:
                os.unlink(temp_file.name)

    def test_large_content_handling(self):
        """Test handling of large content"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            # Create large content
            large_content = "This is a test sentence. " * 10000  # ~250KB of text
            temp_file.write(large_content)
            temp_file.flush()
            
            try:
                doc = self.processor.process_file(temp_file.name)
                
                assert doc is not None
                assert len(doc.content) > 0
                assert doc.metadata['word_count'] > 10000
                
            finally:
                os.unlink(temp_file.name)

    def test_custom_doc_id_generation(self):
        """Test custom document ID generation"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as temp_file:
            test_content = "Test content for doc ID generation."
            temp_file.write(test_content)
            temp_file.flush()
            
            try:
                doc = self.processor.process_file(temp_file.name)
                
                # Default doc_id should be the filename
                assert doc.doc_id == Path(temp_file.name).name
                
                # Test with custom doc_id
                custom_doc = self.processor.process_file(temp_file.name, doc_id="custom_id_123")
                assert custom_doc.doc_id == "custom_id_123"
                
            finally:
                os.unlink(temp_file.name)

if __name__ == '__main__':
    pytest.main([__file__, '-v'])