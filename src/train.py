from model import MiniLMWithAdapter
import asyncio
from vlite2 import VLite2
from constants import VDB_NAME
import json
from openai import AsyncOpenAI
from dotenv import load_dotenv
import os
load_dotenv()

class AdapterTrainer:
    def __init__(self) -> None:
        self.client = AsyncOpenAI(api_key=os.environ['OPENAI_API_KEY'])
        self.embedding_with_adapter = MiniLMWithAdapter()
        self.top_k = 10
        self.questions = self.__load_sample_questions()
        self.vdb = VLite2(vdb_name=VDB_NAME)

    def __load_sample_questions(self) -> list:
        """
        Loads all questions from the squad_questions.json file and returns them as a list.
        """
        file_path = 'squad_questions.json'
        with open(file_path, 'r') as file:
            data = json.load(file)
            questions = [item['Question'] for item in data]
        return questions

    async def evaluate(self, question: str, context: str) -> int:
        """
        Uses LLM to evaluate if the context answers the question. Outputs a tuple of (context, 1 or 0), denoting if the context answers the question or not.
        """
        output_format = {"answers": "1 OR 0"}
        prompt = f"""
        Tell me if the given context ansewrs the question. If it DOES answer the question, output 1. If it does
        NOT answer the question, output 0. Answer in the following JSON format: {output_format}.

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
    
    def retrieve(self, vector):
        """
        Retrieve texts from the vector database using the provided vector. Must be a 1D np.ndarray.
        """
        results = self.vdb.retrieve(vector=vector, top_k=self.top_k)
        return results['texts']
    
    async def train(self):
        for q in self.questions:
            vector_pt = self.embedding_with_adapter.forward(q)
            vector_np = vector_pt.detach().numpy().flatten()
            top_texts = self.retrieve(vector_np)
            evaluation_tasks = [self.evaluate(q, context) for context in top_texts]
            evaluations = await asyncio.gather(*evaluation_tasks)
            evaluations = [evaluation for evaluation in evaluations if evaluation]

            # TODO: weight update and shit here
    
if __name__ == "__main__":
    t = AdapterTrainer()
    asyncio.run(t.train())
