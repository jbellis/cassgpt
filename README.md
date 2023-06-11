# Q&A with ChatGPT enriched with youtube transcriptions

This program uses OpenAI's GPT-3 to generate answers to your questions based on transcriptions of presentations on AI from YouTube.

## Dependencies

- OpenAI API key, stored in file `openai.key`
- Cassandra database supporting vector search. Currently that means you need to build and run 
this branch: https://github.com/datastax/cassandra/tree/vsearch
  - TLDR:
    - `git clone git@github.com:datastax/cassandra.git --branch vsearch`
    - `ant realclean`
    - `ant jar -Duse.jdk11=true`
    - `bin/cassandra -f`
- JDK 11.  _Exactly_ 11.
- You will be able to run cqlsh with vector support if you run `bin/cqlsh` from the cassandra source root
- You can install the Python dependencies for cassgpt by running 
`pip install -r requirements.txt` from this source tree.

## Usage
`python gen-qa-openai.py [--load_data]`

Specifying `--load_data` will will download the dataset, merge the transcriptions into larger chunks, generate embeddings for each chunk using OpenAI's `text-embedding-ada-002` model, and insert the chunks and embeddings into the database.  This will take around twenty minutes and cost about $5 as of May 2023.
This only needs to be done once.

Once the dataset is loaded, the program will prompt you for a question; it will find the most
relevant context from the transcriptions using Cassandra vector search, and feed the resulting
context + question to OpenAI to generate an answer to your query.

Assumes Cassandra is running on localhost, hack the source if it's somewhere else.

## Need to start over?
Instead of rebuilding the embeddings from scratch (slow!), dump them from Cassandra and
re-load them into a fresh database.

`python dump.py`
`python load.py`

Also assumes Cassandra is running on localhost.

---
