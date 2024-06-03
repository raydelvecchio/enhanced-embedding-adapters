# Embedding Adapter Training
Method to train an embedding adapter with no prior labelling or synthetic generation necessary. Using an LLM-in-the-loop to evaluate relevant of retrieved data 
from the adapter over time. Inspired by / building tangentially to [Chroma's Embedding Adapter Report](https://research.trychroma.com/embedding-adapters) and this [Arxiv paper on LLM-improved text embeddings](https://arxiv.org/abs/2401.00368) by Wang et al.

# Data
* We use the questions and contexts from the SQuAD dataset
* However, we do not use any labels or positive / negative matches from these!
* All contexts are individually loaded into a local vector database for retrieval and vector comparison
