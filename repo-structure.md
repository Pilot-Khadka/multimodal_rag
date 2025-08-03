## Multimodal-RAG Repository Structure

## Overview
This repository contains a Multimodal Retrieval-Augmented Generation (RAG) system capable of retrieving information from YouTube videos and images. It combines video captioning, vector embeddings, and a hierarchical retriever to answer user queries using both text and visual content.

## Directory Structure

```
├── 1veritasium_videos.csv         # CSV containing video IDs and processing status
├── configs                        # Configuration files for models and RAG pipeline
│   └── config.yaml
├── data
│   ├── captions/                  # Video captions organized by ID
│   └── videos/                    # Downloaded YouTube videos
├── download_youtube_videos.py     # Script to fetch videos, captions
├── images/                        # Test images for image-based retrieval
├── main.py                        # Entry point to run the full pipeline
├── models/                        # Embedding model (e.g., CLIP)
│   └── embeddings.py
├── pipeline.py                    # Main orchestration pipeline
├── processing/
│   └── text_splitter.py           # Utility for splitting captions into chunks
├── prompts/
│   └── prompts.py                 # Prompt templates for generation
├── pyproject.toml                 # Project metadata and dependencies
├── query_decomposition/
│   └── query.py                   # Breaks down complex queries
├── rag.py                         # High-level RAG orchestration
├── rag_interface.py               # Interface layer for using the RAG system
├── repo-structure.md              # This file
├── requirements.txt               # Runtime dependencies
├── retreival/                     # Retrieval logic
│   ├── hierarchical_retreiver.py
│   └── retreiver.py
├── test_images/                   # Additional test images
├── test.py                        # Unit and integration tests
├── utils/                         # Utility functions
│   ├── generate_video_frame.py
│   ├── process_captions.py
│   └── helper.py
├── uv.lock                        # Lockfile for dependencies
└── vectorstore/
    ├── chroma_store/              # Vector database storage (e.g., Chroma)
    └── manager.py                 # Vector store manager logic
```

## Core Features
**Multimodal Retrieval**
- Supports text-based, image-based, and video caption-based querying.
- Uses CLIP embeddings to connect visual and textual data spaces.

**Hierarchical Retriever**

- Two-stage retrieval: high-level filtering followed by fine-grained reranking.
- Compatible with Chroma or other vector databases.

**YouTube Integration**

- Downloads and processes video + captions using download_youtube_videos.py.
- Captions are chunked and embedded for retrieval.Core Features

# 
