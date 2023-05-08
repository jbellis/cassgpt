from typing import Any, Dict, List
from cassandra.cluster import Cluster
from cassandra.query import SimpleStatement
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
        # print('\n'.join(repr(row) for row in rows))
        return [row.text for row in rows]