import functools
import os
from typing import List
import openai

class DB:
    def __init__(self, source, **kwargs):
        import pinecone
        pinecone.init(**kwargs)
        if source not in pinecone.list_indexes():
            # if does not exist, create index
            pinecone.create_index(
                source,
                dimension=len(res['data'][0]['embedding']),
                metric='cosine',
                metadata_config={'indexed': ['channel_id', 'published']}
            )
        self.index = pinecone.Index(source)

    def upsert_batch(self, meta_batch, embeds):
        ids_batch = [x['id'] for x in meta_batch]
        # cleanup metadata
        meta_batch = [{
            'start': x['start'],
            'end': x['end'],
            'title': x['title'],
            'text': x['text'],
            'url': x['url'],
            'published': x['published'],
            'channel_id': x['channel_id']
        } for x in meta_batch]
        to_upsert = list(zip(ids_batch, embeds, meta_batch))
        # upsert to Pinecone
        self.index.upsert(vectors=to_upsert)

    def query(self, vector, top_k) -> List[str]:
        res = self.index.query(vector, top_k=top_k, include_metadata=True)
        print(res)
        return [x['metadata']['text'] for x in res['matches']]


def embedding_of(vector):
    res = openai.Embedding.create(
        input=[query],
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


window = 20  # number of sentences to combine
def process_data(data, i):
    i_end = min(len(data)-1, i+window)
    if data[i]['title'] != data[i_end]['title']:
        # in this case we skip this entry as we have start/end of two videos
        return None
    text = ' '.join(data[i:i_end]['text'])
    # create the new merged dataset
    return {
        'start': data[i]['start'],
        'end': data[i_end]['end'],
        'title': data[i]['title'],
        'text': text,
        'id': data[i]['id'],
        'url': data[i]['url'],
        'published': data[i]['published'],
        'channel_id': data[i]['channel_id']
    }

if __name__ == '__main__':
    openai.api_key = open('openai.key', 'r').readlines()[0].strip()

    openai.Engine.list()  # check we have authenticated
    from datasets import load_dataset

    data = load_dataset('jamescalam/youtube-transcriptions', split='train')

    from tqdm.auto import tqdm
    import concurrent.futures

    stride = 4  # number of sentences to 'stride' over, used to create overlap
    print('loading data')
    new_data = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        process_data_with_data = functools.partial(process_data, data)
        for result in tqdm(executor.map(process_data_with_data, range(0, len(data), stride))):
            if result is not None:
                new_data.append(result)

    embed_model = "text-embedding-ada-002"
    api_key = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
    env = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"
    index_name = 'openai-youtube-transcriptions'
    db = DB(index_name, api_key, env)

    from tqdm.auto import tqdm
    from time import sleep

    batch_size = 100  # how many embeddings we create and insert at once

    for i in tqdm(range(0, len(new_data), batch_size)):
        # find end of batch
        i_end = min(len(new_data), i+batch_size)
        meta_batch = new_data[i:i_end]
        # get texts to encode
        texts = [x['text'] for x in meta_batch]
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
        embeds = [record['embedding'] for record in res['data']]
        db.upsert_batch(meta_batch, embeds)

    # Now we search, for this we need to create a _query vector_ `xq`:
    query = (
            "Which training method should I use for sentence transformers when " +
            "I only have pairs of related sentences?"
    )

    # get relevant contexts (including the questions)
    print(db.query(embedding_of(query), 2))
    limit = 3750

    def retrieve(query):
        xq = embedding_of(query)
        contexts = db.query(xq, top_k=3)

        # build our prompt with the retrieved contexts included
        prompt_start = (
            "Answer the question based on the context below.\n\n"+
            "Context:\n"
        )
        prompt_end = (
            f"\n\nQuestion: {query}\nAnswer:"
        )
        # append contexts until hitting limit
        for i in range(1, len(contexts)):
            if len("\n\n---\n\n".join(contexts[:i])) >= limit:
                prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts[:i-1]) +
                    prompt_end
                )
                break
            elif i == len(contexts)-1:
                prompt = (
                    prompt_start +
                    "\n\n---\n\n".join(contexts) +
                    prompt_end
                )
        return prompt

    # first we retrieve relevant items from the database
    query_with_contexts = retrieve(query)
    # then we complete the context-infused query
    complete(query_with_contexts)

