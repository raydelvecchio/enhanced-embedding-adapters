
import torch
from transformers import AutoModel, AutoTokenizer
from model import MiniLMWithAdapter
import os
from vlite2 import VLite2
from constants import VDB_NAME

def load_latest_checkpoint(model, checkpoint_dir='checkpoints') -> MiniLMWithAdapter:
    checkpoints = [os.path.join(checkpoint_dir, f) for f in os.listdir(checkpoint_dir) if f.endswith('.pt')]
    latest_checkpoint = max(checkpoints, key=os.path.getctime)
    print(f"Latest model checkpoint: {latest_checkpoint}")
    model.load_state_dict(torch.load(latest_checkpoint))
    return model

def retrieve_with_original_minilm_model(query, vdb, top_k=10):
    original_model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    original_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    inputs = original_tokenizer(query, return_tensors='pt')
    outputs = original_model(**inputs)
    query_vector = outputs.last_hidden_state[:, 0].detach().numpy().flatten()
    return vdb.retrieve(vector=query_vector, top_k=top_k)['texts']

def retrieve_with_minilm_adapter_model(model: MiniLMWithAdapter, query, vdb, top_k=10):
    query_vector_pt = model.forward(query)
    query_vector_copy = query_vector_pt.clone()  # make a copy to avoid detachment from the autodiff graph
    query_vector_np = query_vector_copy.detach().numpy().flatten()
    return vdb.retrieve(vector=query_vector_np, top_k=top_k)['texts']

def main(query: str):
    vdb = VLite2(vdb_name=VDB_NAME)

    # Load untrained model with adapter
    untrained_model = MiniLMWithAdapter()
    untrained_tokenizer = untrained_model.tokenizer

    # Load latest checkpoint
    trained_model = MiniLMWithAdapter()
    trained_model = load_latest_checkpoint(trained_model)

    # Perform retrievals
    top_k = 1
    untrained_results = retrieve_with_minilm_adapter_model(untrained_model, query, vdb, top_k=top_k)
    trained_results = retrieve_with_minilm_adapter_model(trained_model, query, vdb, top_k=top_k)
    original_results = retrieve_with_original_minilm_model(query, vdb, top_k=top_k)

    # Print results
    print("Untrained Adapter Results:", untrained_results)
    print("Checkpoint Adapter Results:", trained_results)
    print("MiniLM Model Results:", original_results)

if __name__ == "__main__":
    q = "where did the virgin mary appear bro"
    main(query=q)
