# Q&A with ChatGPT enriched with youtube transcriptions

This program uses OpenAI's GPT-3 to generate answers to your questions based on transcriptions of presentations on AI from YouTube.

## Dependencies

- OpenAI API key, stored in file `openai.key`
- Cassandra database supporting vector search. Currently that means you need to build and run 
this branch: https://github.com/datastax/cassandra/tree/cep-vsearch
  - TLDR `git clone`, `ant jar`, `bin/cassandra -f`
  - !Note! This is a prerelease and uses `FLOAT VECTOR[N]` syntax for declaring a vector column.
    This will change in the near future.
- Python driver with vector support from this branch: https://github.com/datastax/python-driver/tree/cep-vsearch
  - TLDR `git clone`, `python setup.py install`
  - You will be able to run cqlsh with vector support if you additionally
    - `pip install wcwidth` and then
    - `CQLSH_NO_BUNDLED=True bin/cqlsh` from the cassandra source root
- You can install the other Python dependencies for cassgpt by running 
`pip install -r requirements.txt` from this source tree.

## Usage
`python gen-qa-openai.py [--load-data]`

Specifying --load-data will will download the dataset, merge the transcriptions into larger chunks, generate embeddings for each chunk using OpenAI's text-embedding-ada-002 model, and insert the chunks and embeddings into the database.  This will take around twenty minutes and cost about $5 as of May 2023.
This only needs to be done once.

Once the dataset is loaded, the program will prompt you for a question; it will find the most
relevant context from the transcriptions using Cassandra vector search, and feed the resulting
context + question to OpenAI to generate an answer to your query.
