import base64
import json
from cassandra.cluster import Cluster
from db import DB
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
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
    with open('youtube_transcriptions.json', 'r') as f:
        data = json.load(f)

    num_threads = 32
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        list(tqdm(executor.map(upsert_row, data), total=len(data)))

    print('Data imported from youtube_transcriptions.json')

if __name__ == '__main__':
    main()
