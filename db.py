from typing import Any, Dict, List
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement


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
            embedding vector<float, 1536>);
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

    def upsert_one(self, data):
            query = SimpleStatement(
                f"""
                INSERT INTO {self.keyspace}.{self.table}
                (id, start, end, title, text, url, published, channel_id, embedding)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                """
            )
            self.session.execute(
                query, (
                    data['id'],
                    data['start'],
                    data['end'],
                    data['title'],
                    data['text'],
                    data['url'],
                    data['published'],
                    data['channel_id'],
                    data['embedding']
                )
            )


    def upsert_batch(self, meta_batch: List[Dict[str, Any]], embeds: List[List[float]]):
        for meta, vector in zip(meta_batch, embeds):
            d = meta.copy()
            d['embedding'] = vector
            self.upsert_one(d)

    def query(self, vector: List[float], top_k: int) -> List[str]:
        query = SimpleStatement(
            f"SELECT id, start, end, text FROM {self.keyspace}.{self.table} ORDER BY embedding ANN OF %s LIMIT %s"
        )
        res = self.session.execute(query, (vector, top_k))
        rows = [row for row in res]
        # print('\n'.join(repr(row) for row in rows))
        return [row.text for row in rows]
