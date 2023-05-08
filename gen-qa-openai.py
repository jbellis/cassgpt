import os
from time import sleep
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
from typing import Any, List, Dict
import openai
from tqdm.auto import tqdm
from datasets import load_dataset
from cassandra.marshal import float_pack, float_unpack


def _pack_bytes(vector: List[float]) -> bytes:
    return b''.join(float_pack(x) for x in vector)

def _unpack_bytes(bytes: bytes) -> List[float]:
    return [float_unpack(bytes[i:i+4]) for i in range(0, len(bytes), 4)]
        
class DB:
    def __init__(self, keyspace: str, table: str, **kwargs):
        self.keyspace = keyspace
        self.table = table
        self.cluster = Cluster(**kwargs)
        self.session = self.cluster.connect()

        # Create keyspace if not exists
        self.session.execute(
            f"""
            CREATE KEYSPACE IF NOT EXISTS {keyspace}
            WITH REPLICATION = {{ 'class': 'SimpleStrategy', 'replication_factor': 1 }}
            """
        )

        # Create table if not exists
        self.session.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {keyspace}.{table} (
            id text PRIMARY KEY,
            start float,
            end float,
            title text,
            text text,
            url text,
            published text,
            channel_id text,
            embedding FLOAT VECTOR[1536]);
            """
        )

        # Create SAI index if not exists
        sai_index_name = f"{table}_embedding_idx"
        self.session.execute(
            f"""
            CREATE CUSTOM INDEX IF NOT EXISTS {sai_index_name} ON {keyspace}.{table} (embedding)
            USING 'org.apache.cassandra.index.sai.StorageAttachedIndex'
            """
        )

    def upsert_batch(self, meta_batch: List[Dict[str, Any]], embeds: List[List[float]]):
        for meta, vector in zip(meta_batch, embeds):
            query = SimpleStatement(
                f"""
                INSERT INTO {self.keyspace}.{self.table}
                (id, start, end, title, text, url, published, channel_id, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            )
            self.session.execute(
                query, (
                    meta['id'],
                    meta['start'],
                    meta['end'],
                    meta['title'],
                    meta['text'],
                    meta['url'],
                    meta['published'],
                    meta['channel_id'],
                    _pack_bytes(vector)
                )
            )

    def query(self, vector: List[float], top_k: int) -> List[str]:
        query = SimpleStatement(
            f"SELECT id, start, end, text FROM {self.keyspace}.{self.table} WHERE embedding ANN OF %s LIMIT %s"
        )
        res = self.session.execute(query, (_pack_bytes(vector), top_k))
        rows = [row for row in res]
        print('\n'.join(repr(row) for row in rows))
        return [row.text for row in rows]


openai.api_key = open('openai.key', 'r').read().splitlines()[0]
embed_model = "text-embedding-ada-002"
db = DB("demo", "youtube_transcriptions")

data = load_dataset('jamescalam/youtube-transcriptions', split='train')

new_data = []
window = 20  # number of sentences to combine
stride = 4  # number of sentences to 'stride' over, used to create overlap

print('loading data')
for i in tqdm(range(0, len(data), stride)):
    i_end = min(len(data)-1, i+window)
    if data[i]['title'] != data[i_end]['title']:
        # in this case we skip this entry as we have start/end of two videos
        continue
    text = ' '.join(data[i:i_end]['text'])
    # create the new merged dataset
    new_data.append({
        'start': data[i]['start'],
        'end': data[i_end]['end'],
        'title': data[i]['title'],
        'text': text,
        'id': data[i]['id'],
        'url': data[i]['url'],
        'published': data[i]['published'],
        'channel_id': data[i]['channel_id']
    })
print(f"Created {len(new_data)} new entries from {len(data)} original entries")

batch_size = 100  # how many embeddings we create and insert at once

print('generating embeddings')
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
def embedding_of(text: str) -> List[float]:
    res = openai.Embedding.create(
        input=[text],
        engine=embed_model
    )
    return res['data'][0]['embedding']

def retrieve(query):
    # get relevant contexts (including the questions) and add them to the openai prompt
    limit = 3750
    xq = embedding_of(query)
    contexts = db.query(xq, top_k=3)
    print(f"Retrieved {len(contexts)} contexts after asking for 3")

    # build our prompt with the retrieved contexts included
    prompt_start = (
        "Answer the question based on the context below.\n\n"+
        "Context:\n"
    )
    prompt_end = (
        f"\n\nQuestion: {query}\nAnswer:"
    )
    # append contexts until hitting limit
    for i in range(1, len(contexts) + 1):
        if len("\n\n---\n\n".join(contexts[:i])) >= limit:
            prompt_middle = "\n\n---\n\n".join(contexts[:i-1])
            break
    else:
        prompt_middle = "\n\n---\n\n".join(contexts)

    return prompt_start + prompt_middle + prompt_end

# first we retrieve relevant items from the database
query_with_contexts = retrieve(query)
# then we complete the context-infused query
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
print(query_with_contexts)
response = complete(query_with_contexts)
print(f'After enriching with youtube context, the answer is: {response}')
