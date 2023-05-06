# Markdown Cell
# [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pinecone-io/examples/blob/master/generation/generative-qa/openai/gen-qa-openai/gen-qa-openai.ipynb) [![Open nbviewer](https://raw.githubusercontent.com/pinecone-io/examples/master/assets/nbviewer-shield.svg)](https://nbviewer.org/github/pinecone-io/examples/blob/master/generation/generative-qa/openai/gen-qa-openai/gen-qa-openai.ipynb)
# Markdown Cell
# # Retrieval Enhanced Generative Question Answering with OpenAI
# 
# #### Fixing LLMs that Hallucinate
# 
# In this notebook we will learn how to query relevant contexts to our queries from Pinecone, and pass these to a generative OpenAI model to generate an answer backed by real data sources. Required installs for this notebook are:
# Code Cell
!pip install -qU openai pinecone-client datasets tqdm
# Code Cell
import os
import openai

# get API key from top-right dropdown on OpenAI website
openai.api_key = os.getenv("OPENAI_API_KEY") or "OPENAI_API_KEY"

openai.Engine.list()  # check we have authenticated
# Markdown Cell
# For many questions *state-of-the-art (SOTA)* LLMs are more than capable of answering correctly.
# Code Cell
query = "who was the 12th person on the moon and when did they land?"

# now query text-davinci-003 WITHOUT context
res = openai.Completion.create(
    engine='text-davinci-003',
    prompt=query,
    temperature=0,
    max_tokens=400,
    top_p=1,
    frequency_penalty=0,
    presence_penalty=0,
    stop=None
)

res['choices'][0]['text'].strip()
# Markdown Cell
# However, that isn't always the case. Let's first rewrite the above into a simple function so we're not rewriting this every time.
# Code Cell
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
# Markdown Cell
# Now let's ask a more specific question about training a specific type of transformer model called a *sentence-transformer*. The ideal answer we'd be looking for is _"Multiple Negatives Ranking (MNR) loss"_.
# 
# Don't worry if this is a new term to you, it isn't required to understand what we're doing or demoing here.
# Code Cell
query = (
    "Which training method should I use for sentence transformers when " +
    "I only have pairs of related sentences?"
)

complete(query)
# Markdown Cell
# One of the common answers I get to this is:
# 
# ```
# The best training method to use for fine-tuning a pre-trained model with sentence transformers is the Masked Language Model (MLM) training. MLM training involves randomly masking some of the words in a sentence and then training the model to predict the masked words. This helps the model to learn the context of the sentence and better understand the relationships between words.
# ```
# 
# This answer seems pretty convincing right? Yet, it's wrong. MLM is typically used in the pretraining step of a transformer model but *cannot* be used to fine-tune a sentence-transformer, and has nothing to do with having _"pairs of related sentences"_.
# 
# An alternative answer I recieve is about `supervised learning approach` being the most suitable. This is completely true, but it's not specific and doesn't answer the question.
# 
# We have two options for enabling our LLM in understanding and correctly answering this question:
# 
# 1. We fine-tune the LLM on text data covering the topic mentioned, likely on articles and papers talking about sentence transformers, semantic search training methods, etc.
# 
# 2. We use **R**etrieval **A**ugmented **G**eneration (RAG), a technique that implements an information retrieval component to the generation process. Allowing us to retrieve relevant information and feed this information into the generation model as a *secondary* source of information.
# 
# We will demonstrate option **2**.
# Markdown Cell
# ---
# 
# ## Building a Knowledge Base
# 
# With open **2** the retrieval of relevant information requires an external _"Knowledge Base"_, a place where we can store and use to efficiently retrieve information. We can think of this as the external _long-term memory_ of our LLM.
# 
# We will need to retrieve information that is semantically related to our queries, to do this we need to use _"dense vector embeddings"_. These can be thought of as numerical representations of the *meaning* behind our sentences.
# 
# There are many options for creating these dense vectors, like open source [sentence transformers](https://pinecone.io/learn/nlp/) or OpenAI's [ada-002 model](https://youtu.be/ocxq84ocYi0). We will use OpenAI's offering in this example.
# 
# We have already authenticated our OpenAI connection, to create an embedding we just do:
# Code Cell
embed_model = "text-embedding-ada-002"

res = openai.Embedding.create(
    input=[
        "Sample document text goes here",
        "there will be several phrases in each batch"
    ], engine=embed_model
)
# Markdown Cell
# In the response `res` we will find a JSON-like object containing our new embeddings within the `'data'` field.
# Code Cell
res.keys()
# Markdown Cell
# Inside `'data'` we will find two records, one for each of the two sentences we just embedded. Each vector embedding contains `1536` dimensions (the output dimensionality of the `text-embedding-ada-002` model.
# Code Cell
len(res['data'])
# Code Cell
len(res['data'][0]['embedding']), len(res['data'][1]['embedding'])
# Markdown Cell
# We will apply this same embedding logic to a dataset containing information relevant to our query (and many other queries on the topics of ML and AI).
# 
# ### Data Preparation
# 
# The dataset we will be using is the `jamescalam/youtube-transcriptions` from Hugging Face _Datasets_. It contains transcribed audio from several ML and tech YouTube channels. We download it with:
# Code Cell
from datasets import load_dataset

data = load_dataset('jamescalam/youtube-transcriptions', split='train')
data
# Code Cell
data[0]
# Markdown Cell
# The dataset contains many small snippets of text data. We will need to merge many snippets from each video to create more substantial chunks of text that contain more information.
# Code Cell
from tqdm.auto import tqdm

new_data = []

window = 20  # number of sentences to combine
stride = 4  # number of sentences to 'stride' over, used to create overlap

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
# Code Cell
new_data[0]
# Markdown Cell
# Now we need a place to store these embeddings and enable a efficient _vector search_ through them all. To do that we use Pinecone, we can get a [free API key](https://app.pinecone.io) and enter it below where we will initialize our connection to Pinecone and create a new index.
# Code Cell
import pinecone

# initialize connection to pinecone (get API key at app.pinecone.io)
api_key = os.getenv("PINECONE_API_KEY") or "PINECONE_API_KEY"
# find your environment next to the api key in pinecone console
env = os.getenv("PINECONE_ENVIRONMENT") or "PINECONE_ENVIRONMENT"

pinecone.init(api_key=api_key, enviroment=env)
pinecone.whoami()
# Code Cell
index_name = 'openai-youtube-transcriptions'
# Code Cell
# check if index already exists (it shouldn't if this is first time)
if index_name not in pinecone.list_indexes():
    # if does not exist, create index
    pinecone.create_index(
        index_name,
        dimension=len(res['data'][0]['embedding']),
        metric='cosine',
        metadata_config={'indexed': ['channel_id', 'published']}
    )
# connect to index
index = pinecone.Index(index_name)
# view index stats
index.describe_index_stats()
# Markdown Cell
# We can see the index is currently empty with a `total_vector_count` of `0`. We can begin populating it with OpenAI `text-embedding-ada-002` built embeddings like so:
# Code Cell
from tqdm.auto import tqdm
from time import sleep

batch_size = 100  # how many embeddings we create and insert at once

for i in tqdm(range(0, len(new_data), batch_size)):
    # find end of batch
    i_end = min(len(new_data), i+batch_size)
    meta_batch = new_data[i:i_end]
    # get ids
    ids_batch = [x['id'] for x in meta_batch]
    # get texts to encode
    texts = [x['text'] for x in meta_batch]
    # create embeddings (try-except added to avoid RateLimitError)
    try:
        res = openai.Embedding.create(input=texts, engine=embed_model)
    except:
        done = False
        while not done:
            sleep(5)
            try:
                res = openai.Embedding.create(input=texts, engine=embed_model)
                done = True
            except:
                pass
    embeds = [record['embedding'] for record in res['data']]
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
    index.upsert(vectors=to_upsert)
# Markdown Cell
# Now we search, for this we need to create a _query vector_ `xq`:
# Code Cell
res = openai.Embedding.create(
    input=[query],
    engine=embed_model
)

# retrieve from Pinecone
xq = res['data'][0]['embedding']

# get relevant contexts (including the questions)
res = index.query(xq, top_k=2, include_metadata=True)
# Code Cell
res
# Code Cell
limit = 3750

def retrieve(query):
    res = openai.Embedding.create(
        input=[query],
        engine=embed_model
    )

    # retrieve from Pinecone
    xq = res['data'][0]['embedding']

    # get relevant contexts
    res = index.query(xq, top_k=3, include_metadata=True)
    contexts = [
        x['metadata']['text'] for x in res['matches']
    ]

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
# Code Cell
# first we retrieve relevant items from Pinecone
query_with_contexts = retrieve(query)
query_with_contexts
# Code Cell
# then we complete the context-infused query
complete(query_with_contexts)
# Markdown Cell
# And we get a pretty great answer straight away, specifying to use _multiple-rankings loss_ (also called _multiple negatives ranking loss_).
# Markdown Cell
# Once we're done with the index we delete it to save resources:
# Code Cell
pinecone.delete_index(index_name)
# Markdown Cell
# ---
