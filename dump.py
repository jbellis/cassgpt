import base64
import json
from cassandra.cluster import Cluster
from cassandra.query import dict_factory
from tqdm.auto import tqdm

def main():
    cluster = Cluster(['127.0.0.1'])  # replace with your Cassandra hosts
    session = cluster.connect('demo')  # replace with your keyspace name
    session.row_factory = dict_factory

    rows = session.execute('SELECT * FROM youtube_transcriptions')
    data = []
    for row in tqdm(rows):
        # Convert blob to base64
        row['embedding'] = base64.b64encode(row['embedding']).decode()
        data.append(row)

    with open('youtube_transcriptions.json', 'w') as f:
        json.dump(data, f, indent=2)

    print('Data exported to youtube_transcriptions.json')

if __name__ == '__main__':
    main()
