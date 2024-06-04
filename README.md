# Embedding Adapter Training
Method to train an embedding adapter with no prior labelling or synthetic generation necessary. Using an LLM-in-the-loop to evaluate relevant of retrieved data 
from the adapter over time. Inspired by / building tangentially to [Chroma's Embedding Adapter Report](https://research.trychroma.com/embedding-adapters) and this [Arxiv paper on LLM-improved text embeddings](https://arxiv.org/abs/2401.00368) by Wang et al.

# Data
* We use the questions and contexts from the SQuAD dataset
* However, we do not use any labels or positive / negative matches from these!
* All contexts are individually loaded into a local vector database for retrieval and vector comparison

# Notes
* Training code for Chroma's Embedding Adapter found [here](https://github.com/suvansh/ChromaAdaptEmbed/blob/main/adapt_embed/models/nn/nn.py)
    * Appears that they use Triplet, MSE, or BCE loss to train it
    * The `.forward` method is used to apply the adapters naturally
    * Entire training pipeline found [here](https://github.com/suvansh/ChromaAdaptEmbed/blob/main/adapt_embed/models/nn/run_nn.py)

# TODO
* Figure out why this isn't training properly lmfao
