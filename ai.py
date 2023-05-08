import openai
from typing import List

def embedding_of_many(texts: List[str], embed_model: str) -> List[str]:
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            print("Rate limit error, waiting 5 seconds...")
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    return [record['embedding'] for record in res['data']]

def embedding_of(text: str, embed_model: str) -> List[float]:
    res = openai.Embedding.create(
        input=[text],
        engine=embed_model
    )
    return res['data'][0]['embedding']

def complete(prompt):
    # query text-davinci-003
    res = openai.Completion.create(
        engine='text-davinci-003',
        prompt=prompt,
        temperature=0,
        max_tokens=400,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None
    )
    return res['choices'][0]['text'].strip()
