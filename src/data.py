from datasets import load_dataset
from vlite2 import VLite2, Ingestor
from constants import VDB_NAME
import json

def download_squad_dataset():
    """
    Downloads the SQuAD dataset for question-answering tasks.
    """
    dataset = load_dataset("squad")
    questions = []
    contexts = []

    for item in dataset['train']:
        questions.append({"Question": item['question']})
        contexts.append({"Context": item['context']})

    for item in dataset['validation']:
        questions.append({"Question": item['question']})
        contexts.append({"Context": item['context']})

    unique_contexts = []
    seen_contexts = set()

    # de-duping contexts
    for context in contexts:
        context_text = context["Context"]
        if context_text not in seen_contexts:
            unique_contexts.append(context)
            seen_contexts.add(context_text)

    contexts = unique_contexts

    with open("squad_questions.json", "w", encoding="utf-8") as q_file:
        json.dump(questions, q_file, ensure_ascii=False, indent=4)

    with open("squad_contexts.json", "w", encoding="utf-8") as c_file:
        json.dump(contexts, c_file, ensure_ascii=False, indent=4)

    print("SQuAD dataset downloaded successfully.")

def build_database(max_size: int = 50):
    """
    Loads the text data from wikipedia_corpus.txt and processes it for database insertion using VLite2.
    """
    vdb = VLite2(vdb_name=VDB_NAME, embedding_model='all-MiniLM-L6-v2')
    vdb.clear()

    with open("squad_contexts.json", "r", encoding="utf-8") as file:
        contexts = json.load(file)
    
    ingestor = Ingestor(vdb)
    num_entries = len(contexts)
    
    for i, context in enumerate(contexts[:max_size]):
        context_text = context["Context"].strip()
        if context_text:
            print(f"Processing context {i+1} of {min(num_entries, max_size)}")
            ingestor.processString(context_text)
    
    vdb.save()
    print(f"Database built and saved successfully with a maximum of {max_size} entries.")

if __name__ == "__main__":
    # download_squad_dataset()
    # build_database()
