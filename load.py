import base64
import json
from cassandra.cluster import Cluster
from db import DB
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
import random
import threading
import time

thread_local_storage = threading.local()

def get_db_handle():
    if not hasattr(thread_local_storage, 'db_handle'):
        thread_local_storage.db_handle = DB('demo', 'youtube_transcriptions')
    return thread_local_storage.db_handle

def upsert_row(row):
    db = get_db_handle()
    row['embedding'] = base64.b64decode(row['embedding'])
    db.upsert_one(row)

def main():
    print("Waiting for Cassandra schema")
    get_db_handle() # let one thread create the table + index
    time.sleep(1)

    print("Reading data")
    with open('youtube_transcriptions.json', 'r') as f:
        data = json.load(f)
    random.shuffle(data);  # avoid saturating a single memtable shard

    print("Importing data")
    num_threads = 64
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(upsert_row, data), total=len(data)))

    print('Data imported from youtube_transcriptions.json')

if __name__ == '__main__':
    main()
