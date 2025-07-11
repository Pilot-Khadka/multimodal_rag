<div align="center">
<img src="images/title.png">
</div>

## About the project

**Status: Incomplete** – under active development

## Implemented 

- Text integration working
- Download and clean YouTube captions
- Text embedding with recursive splitting
- Image embedding using CLIP + histogram-based frame selection from videos
- Multirepresentational chunking
- CLI app
- Query decomposition - multi-query, step-back, HyDE

## Remaining 
- **Image Integration** –  core embeddings are done, needs adding it to interface
- complete rag interface (add image query), pass rag through the interface to cli
- **Hierarchical Chunking** –  add tree-like metadata or context windows  
- **Reranking / Rank Fusion** – planned-COLBERT,  
- **Query fusion** - probably?

## Usage

Run main.py
```
python main.py
```