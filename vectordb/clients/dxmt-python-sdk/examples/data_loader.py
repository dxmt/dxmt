"""
Data Loader Utility for DXMT

Batch data loading utility for ingesting large datasets into DXMT.
Supports various data formats and provides progress tracking.

Use cases:
- Initial data migration
- Periodic data updates
- Document ingestion for RAG
- Large-scale data imports

Requirements:
    pip install dxmt sentence-transformers
"""

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional, Iterator
from dxmt import Client
from common_utils import create_test_client, setup_test_table, print_section

# Optional: Sentence transformers for embeddings
try:
    from sentence_transformers import SentenceTransformer
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False
    print("⚠️  sentence-transformers not installed. Install with: pip install sentence-transformers")


class DataLoader:
    """
    Batch data loader for DXMT with progress tracking and error handling.
    """

    def __init__(
        self,
        client: Client,
        table_name: str,
        embedding_model: Optional[str] = "all-MiniLM-L6-v2",
        batch_size: int = 100,
        embedding_dimension: Optional[int] = None
    ):
        """
        Initialize data loader.

        Args:
            client: DXMT client
            table_name: Target table name
            embedding_model: Sentence transformer model name (None for no embeddings)
            batch_size: Number of records per batch
            embedding_dimension: Custom dimension for random embeddings (when embedding_model=None)
        """
        self.client = client
        self.table_name = table_name
        self.batch_size = batch_size

        # Initialize embedding model
        if embedding_model and HAS_SENTENCE_TRANSFORMERS:
            print(f"Loading embedding model: {embedding_model}")
            self.embedder = SentenceTransformer(embedding_model)
            self.dimension = self.embedder.get_sentence_embedding_dimension()
            print(f"✓ Embedding model loaded (dim={self.dimension})")
        else:
            self.embedder = None
            self.dimension = embedding_dimension or 384  # Custom or default dimension

        self.stats = {
            'total_processed': 0,
            'total_inserted': 0,
            'total_errors': 0,
            'start_time': None,
            'end_time': None,
        }

    def load_from_texts(
        self,
        texts: List[str],
        metadatas: Optional[List[Dict[str, Any]]] = None,
        ids: Optional[List[str]] = None
    ) -> None:
        """
        Load documents from list of texts.

        Args:
            texts: List of text strings
            metadatas: Optional list of metadata dicts (same length as texts)
            ids: Optional list of IDs (same length as texts)
        """
        self.stats['start_time'] = time.time()

        if metadatas is None:
            metadatas = [{}] * len(texts)

        if ids is None:
            ids = [str(i) for i in range(len(texts))]

        print(f"\nLoading {len(texts)} documents...")
        print(f"Batch size: {self.batch_size}")

        # Process in batches
        for batch_start in range(0, len(texts), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(texts))
            batch_texts = texts[batch_start:batch_end]
            batch_metadatas = metadatas[batch_start:batch_end]
            batch_ids = ids[batch_start:batch_end]

            self._process_batch(batch_texts, batch_metadatas, batch_ids)

            # Progress update
            progress = (batch_end / len(texts)) * 100
            print(f"Progress: {batch_end}/{len(texts)} ({progress:.1f}%)")

        self.stats['end_time'] = time.time()
        self._print_stats()

    def load_from_directory(
        self,
        directory: str,
        file_pattern: str = "*.txt",
        encoding: str = "utf-8",
        extract_metadata: bool = True
    ) -> None:
        """
        Load documents from directory of text files.

        Args:
            directory: Directory path
            file_pattern: Glob pattern for files (e.g., "*.txt", "**/*.md")
            encoding: File encoding
            extract_metadata: Whether to extract metadata from filename/path
        """
        self.stats['start_time'] = time.time()

        # Find all matching files
        dir_path = Path(directory)
        files = list(dir_path.glob(file_pattern))
        print(f"\nFound {len(files)} files matching '{file_pattern}'")

        if not files:
            print("No files found")
            return

        # Process files in batches
        batch_texts = []
        batch_metadatas = []
        batch_ids = []

        for file_path in files:
            try:
                # Read file
                with open(file_path, 'r', encoding=encoding) as f:
                    content = f.read()

                # Extract metadata
                metadata = {}
                if extract_metadata:
                    metadata['filename'] = file_path.name
                    metadata['filepath'] = str(file_path)
                    metadata['extension'] = file_path.suffix
                    metadata['size'] = os.path.getsize(file_path)

                # Generate ID from filename
                file_id = file_path.stem

                batch_texts.append(content)
                batch_metadatas.append(metadata)
                batch_ids.append(file_id)

                # Process batch if full
                if len(batch_texts) >= self.batch_size:
                    self._process_batch(batch_texts, batch_metadatas, batch_ids)
                    batch_texts = []
                    batch_metadatas = []
                    batch_ids = []

                    # Progress update
                    progress = (self.stats['total_processed'] / len(files)) * 100
                    print(f"Progress: {self.stats['total_processed']}/{len(files)} ({progress:.1f}%)")

            except Exception as e:
                print(f"Error processing {file_path}: {e}")
                self.stats['total_errors'] += 1

        # Process remaining batch
        if batch_texts:
            self._process_batch(batch_texts, batch_metadatas, batch_ids)

        self.stats['end_time'] = time.time()
        self._print_stats()

    def load_from_jsonl(
        self,
        jsonl_file: str,
        text_field: str = "text",
        id_field: Optional[str] = "id",
        metadata_fields: Optional[List[str]] = None
    ) -> None:
        """
        Load documents from JSONL file.

        Args:
            jsonl_file: Path to JSONL file
            text_field: Field name containing text content
            id_field: Field name for ID (None to auto-generate)
            metadata_fields: List of fields to include as metadata (None for all except text)
        """
        self.stats['start_time'] = time.time()

        print(f"\nLoading from JSONL: {jsonl_file}")

        batch_texts = []
        batch_metadatas = []
        batch_ids = []

        try:
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                for line_num, line in enumerate(f, 1):
                    try:
                        data = json.loads(line)

                        # Extract text
                        if text_field not in data:
                            print(f"Warning: Line {line_num} missing '{text_field}' field")
                            continue

                        text = data[text_field]

                        # Extract ID
                        if id_field and id_field in data:
                            doc_id = str(data[id_field])
                        else:
                            doc_id = str(line_num)

                        # Extract metadata
                        if metadata_fields:
                            metadata = {k: data[k] for k in metadata_fields if k in data}
                        else:
                            metadata = {k: v for k, v in data.items() if k != text_field}

                        batch_texts.append(text)
                        batch_metadatas.append(metadata)
                        batch_ids.append(doc_id)

                        # Process batch if full
                        if len(batch_texts) >= self.batch_size:
                            self._process_batch(batch_texts, batch_metadatas, batch_ids)
                            batch_texts = []
                            batch_metadatas = []
                            batch_ids = []

                            # Progress update
                            if line_num % 1000 == 0:
                                print(f"Processed {line_num} lines...")

                    except json.JSONDecodeError as e:
                        print(f"Error parsing line {line_num}: {e}")
                        self.stats['total_errors'] += 1

            # Process remaining batch
            if batch_texts:
                self._process_batch(batch_texts, batch_metadatas, batch_ids)

        except FileNotFoundError:
            print(f"Error: File not found: {jsonl_file}")
            return

        self.stats['end_time'] = time.time()
        self._print_stats()

    def _process_batch(
        self,
        texts: List[str],
        metadatas: List[Dict[str, Any]],
        ids: List[str]
    ) -> None:
        """
        Process a batch of documents.

        Args:
            texts: List of text strings
            metadatas: List of metadata dicts
            ids: List of IDs
        """
        try:
            # Generate embeddings
            if self.embedder:
                embeddings = self.embedder.encode(texts, show_progress_bar=False).tolist()
            else:
                # Random embeddings fallback
                import random
                embeddings = [[random.random() for _ in range(self.dimension)] for _ in texts]

            # Build records
            records = []
            for i, (text, metadata, doc_id) in enumerate(zip(texts, metadatas, ids)):
                # Store text in metadata
                metadata_copy = metadata.copy()
                metadata_copy['text'] = text

                record = {
                    "id": doc_id,
                    "vector": embeddings[i],
                    "metadata": metadata_copy
                }
                records.append(record)

            # Insert batch
            self.client.insert_records(self.table_name, records)

            self.stats['total_processed'] += len(records)
            self.stats['total_inserted'] += len(records)

        except Exception as e:
            print(f"Error processing batch: {e}")
            self.stats['total_errors'] += len(texts)

    def _print_stats(self) -> None:
        """Print loading statistics."""
        duration = self.stats['end_time'] - self.stats['start_time']

        print("\n" + "=" * 80)
        print("Loading Statistics")
        print("=" * 80)
        print(f"Total processed:   {self.stats['total_processed']}")
        print(f"Successfully inserted: {self.stats['total_inserted']}")
        print(f"Errors:           {self.stats['total_errors']}")
        print(f"Duration:         {duration:.2f}s")

        if duration > 0:
            throughput = self.stats['total_processed'] / duration
            print(f"Throughput:       {throughput:.1f} docs/sec")

        print("=" * 80)


def create_sample_files(directory: str, num_files: int = 10) -> None:
    """
    Create sample text files for demo.

    Args:
        directory: Directory to create files in
        num_files: Number of files to create
    """
    os.makedirs(directory, exist_ok=True)

    topics = ["AI", "ML", "Data Science", "Cloud", "Security"]
    for i in range(num_files):
        filename = f"doc_{i:03d}.txt"
        filepath = os.path.join(directory, filename)

        topic = topics[i % len(topics)]
        content = f"This is document {i} about {topic}.\n"
        content += f"It contains information related to {topic} and its applications.\n"
        content += f"Document ID: {i}, Topic: {topic}\n"

        with open(filepath, 'w') as f:
            f.write(content)

    print(f"✓ Created {num_files} sample files in {directory}")


def main():
    print_section("Data Loader Utility Demo")

    # Check prerequisites
    if not HAS_SENTENCE_TRANSFORMERS:
        print("⚠️  sentence-transformers not installed")
        print("   Install with: pip install sentence-transformers")
        print("   Continuing with random embeddings...")

    # Create client
    client = create_test_client()

    # Setup table
    table_name = "data_loader_demo"
    dimension = 384
    setup_test_table(client, table_name, dimension)

    # Initialize data loader
    loader = DataLoader(
        client=client,
        table_name=table_name,
        embedding_model="all-MiniLM-L6-v2" if HAS_SENTENCE_TRANSFORMERS else None,
        batch_size=50
    )

    # ==========================================================================
    # Example 1: Load from text list
    # ==========================================================================
    print_section("Example 1: Load from Text List")

    texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Deep learning uses neural networks with multiple layers.",
        "Natural language processing enables computers to understand human language.",
        "Computer vision allows machines to interpret visual information.",
        "Reinforcement learning trains agents through rewards and penalties.",
    ]

    metadatas = [
        {"category": "ML", "difficulty": "beginner"},
        {"category": "DL", "difficulty": "intermediate"},
        {"category": "NLP", "difficulty": "intermediate"},
        {"category": "CV", "difficulty": "advanced"},
        {"category": "RL", "difficulty": "advanced"},
    ]

    loader.load_from_texts(texts, metadatas=metadatas)

    # ==========================================================================
    # Example 2: Load from directory
    # ==========================================================================
    print_section("Example 2: Load from Directory")

    # Create sample files
    sample_dir = "/tmp/dxmt_sample_docs"
    create_sample_files(sample_dir, num_files=20)

    loader.load_from_directory(
        directory=sample_dir,
        file_pattern="*.txt",
        extract_metadata=True
    )

    # ==========================================================================
    # Example 3: Load from JSONL
    # ==========================================================================
    print_section("Example 3: Load from JSONL File")

    # Create sample JSONL file
    jsonl_file = "/tmp/dxmt_sample.jsonl"
    with open(jsonl_file, 'w') as f:
        for i in range(15):
            doc = {
                "id": f"jsonl_{i}",
                "text": f"This is JSONL document {i} with structured data.",
                "category": f"category_{i % 3}",
                "score": i * 10,
                "active": i % 2 == 0
            }
            f.write(json.dumps(doc) + '\n')

    print(f"✓ Created sample JSONL file: {jsonl_file}")

    loader.load_from_jsonl(
        jsonl_file=jsonl_file,
        text_field="text",
        id_field="id"
    )

    # Verify loaded data
    print_section("Verification")

    # Count total records
    results = client.search(
        table_name=table_name,
        vector=[0.1] * dimension,
        k=100
    )
    print(f"✓ Total records in table: {len(results)}")
    print(f"  Expected: ~40 (5 + 20 + 15)")

    # Cleanup
    print_section("Cleanup")
    try:
        client.delete_table(table_name)
        print(f"✓ Deleted table '{table_name}'")

        # Clean up sample files
        import shutil
        if os.path.exists(sample_dir):
            shutil.rmtree(sample_dir)
            print(f"✓ Deleted sample directory: {sample_dir}")

        if os.path.exists(jsonl_file):
            os.remove(jsonl_file)
            print(f"✓ Deleted sample JSONL: {jsonl_file}")

    except Exception as e:
        print(f"Error during cleanup: {e}")

    print("\n✓ Data loader demo complete!")


if __name__ == "__main__":
    main()
