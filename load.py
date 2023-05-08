import base64
import json
from cassandra.cluster import Cluster
from db import DB
from tqdm.auto import tqdm


def main():
    db = DB('demo', 'youtube_transcriptions')

    with open('youtube_transcriptions.json', 'r') as f:
        data = json.load(f)

    for row in tqdm(data):
        # Convert base64 back to blob
        row['embedding'] = base64.b64decode(row['embedding'])
        db.upsert_one(row)

    print('Data imported from youtube_transcriptions.json')

if __name__ == '__main__':
    main()
