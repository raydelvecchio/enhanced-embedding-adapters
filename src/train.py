from model import MiniLMWithAdapter
import torch
import torch.nn as nn
from torch.optim import Adam
import random
import asyncio
from vlite2 import VLite2
from constants import VDB_NAME
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

class AdapterTrainer:
    def __init__(self, top_k: int = 10, learning_rate: float = 1e-4, epochs: int = 3) -> None:
        self.client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.embedding_with_adapter = MiniLMWithAdapter()
        self.top_k = top_k
        self.lr = learning_rate
        self.epochs = epochs
        self.questions = self.__load_sample_questions()
        self.vdb = VLite2(vdb_name=VDB_NAME)

    def __load_sample_questions(self) -> list:
        """
        Loads all questions from the squad_questions.json file and returns them as a list.
        """
        file_path = 'squad_questions.json'
        with open(file_path, 'r') as f:
            data = json.load(f)
        questions = [item['Question'] for item in data]
        random.shuffle(questions)
        return questions

    async def evaluate(self, question: str, context: str) -> tuple[int, str]:
        """
        Uses LLM to evaluate if the context answers the question. Outputs a tuple of (context, 1 or -1), denoting if the context answers the question or not.
        """
        output_format = {"answers": "1 OR -1"}
        prompt = f"""
        Tell me if the given context ansewrs the question. If it DOES answer the question, output 1. If it does
        NOT answer the question, output -1. Answer in the following JSON format: {output_format}.

        Put nothing else in the JSON object except for the int.

        QUESTION:
        {question}

        CONTEXT:
        {context}

        JSON output:
        """

        messages = [{'role': 'system', 'content': ''}, {'role': 'user', 'content': prompt}]

        try:
            completion = await self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                response_format={"type": "json_object"},
            )
            output = completion.choices[0].message.content
            output = json.loads(output)
            return (int(output['answers']), context)
        except:
            return ()
    
    async def train(self):
        optimizer = Adam(self.embedding_with_adapter.dense_adapter.parameters(), lr=self.lr)
        loss_fn = nn.CosineEmbeddingLoss(margin=0.5)

        for epoch in range(self.epochs):
            print(f"\nStarting epoch {epoch+1}/{self.epochs}\n")
            for i, q in enumerate(self.questions):
                query_vector_pt = self.embedding_with_adapter.forward(q)

                query_vector_copy = query_vector_pt.clone()  # make a copy to avoid detachment from the autodiff graph
                query_vector_np = query_vector_copy.detach().numpy().flatten()

                top_contexts = self.vdb.retrieve(vector=query_vector_np, top_k=self.top_k)['texts']

                if not top_contexts:
                    continue

                evaluation_tasks = [self.evaluate(q, context) for context in top_contexts]
                evaluations = [evaluation for evaluation in await asyncio.gather(*evaluation_tasks) if evaluation]

                total_loss_per_question = 0.0
                for context, evaluation in zip(top_contexts, evaluations):
                    print(f"Question: {q}")
                    print(f"Context: {context}")
                    print(f"Evaluation: {evaluation[0]}")

                    optimizer.zero_grad()
                    query_vector_pt = self.embedding_with_adapter.forward(q)  # re-create the query vector so the backprop graph stays intact
                    context_tokenization = self.embedding_with_adapter.tokenizer(text=context, return_tensors='pt')
                    context_embedding_output = self.embedding_with_adapter.embedding_model(**context_tokenization)  # creating the embedding of the context with the original embedding model, since we only apply a transform via adapter to the query
                    context_vector_pt = context_embedding_output.last_hidden_state[:, 0]
                    target = torch.tensor([evaluation[0] * -1], dtype=torch.float32)
                    loss = loss_fn(query_vector_pt, context_vector_pt, target)
                    loss.backward()
                    optimizer.step()
                    total_loss_per_question += loss.item()
                
                average_loss = total_loss_per_question / len(evaluations) if evaluations else 0.0
                print(f"Average Loss for Question: {average_loss}\n")
                    
                if i % 50 == 0 and i != 0:
                    checkpoint_dir = 'checkpoints'
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    checkpoint_path = f'{checkpoint_dir}/checkpoint_question_{(i+1) * (epoch + 1)}.pt'
                    torch.save(self.embedding_with_adapter.state_dict(), checkpoint_path)
                    print(f"Checkpoint saved at {checkpoint_path}")
                    
            print(f"\nEpoch {epoch+1} completed\n")
    
if __name__ == "__main__":
    t = AdapterTrainer()
    asyncio.run(t.train())
