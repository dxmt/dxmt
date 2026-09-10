"""The fixed corpus every functional test asserts against.

Eight records in four dimensions. Small enough to write the expected answer down, which is the
only way an assertion can catch "the right number of wrong records".

The vectors are spread over the unit sphere rather than along a line. A first draft put them on
a line through 4-space, which reads well and measures nothing: under cosine every point on a
line through the origin is parallel, so all eight scored exactly 1.0 and the "nearest" neighbour
was whatever the tie-break returned. Each record here is nearest to itself and the order away
from it is strict.

The payload carries one numeric field and two string fields on purpose. Since the numeric index
became a sorted array while strings stayed an inverted map (`payload_index.rs`), a filter on
`rank` and a filter on `city` take different code paths, and a test that only ever filters on
one of them covers half the filtering.
"""

from .client import post

DIM = 4

CORPUS = [
    {"id": "r1", "vector": [1.00, 0.00, 0.00, 0.00], "payload": {"rank": 1, "city": "London", "tag": "alpha"}},
    {"id": "r2", "vector": [0.92, 0.39, 0.00, 0.00], "payload": {"rank": 2, "city": "London", "tag": "beta"}},
    {"id": "r3", "vector": [0.71, 0.71, 0.00, 0.00], "payload": {"rank": 3, "city": "Moscow", "tag": "alpha"}},
    {"id": "r4", "vector": [0.39, 0.92, 0.00, 0.00], "payload": {"rank": 4, "city": "Moscow", "tag": "gamma"}},
    {"id": "r5", "vector": [0.00, 1.00, 0.00, 0.00], "payload": {"rank": 5, "city": "Berlin", "tag": "beta"}},
    {"id": "r6", "vector": [0.00, 0.71, 0.71, 0.00], "payload": {"rank": 6, "city": "Berlin", "tag": "alpha"}},
    {"id": "r7", "vector": [0.00, 0.00, 1.00, 0.00], "payload": {"rank": 7, "city": "Paris", "tag": "gamma"}},
    {"id": "r8", "vector": [0.00, 0.00, 0.00, 1.00], "payload": {"rank": 8, "city": "Paris", "tag": "beta"}},
]

#: Nearest-neighbour order from `r1`, by construction: r1, r2, r3, r4 turn away from the first
#: axis a step at a time, r5 is perpendicular to it, and r6-r8 further still. Written down so a
#: test can assert the order rather than only the membership.
ORDER_FROM_R1 = ["r1", "r2", "r3", "r4", "r5"]


BY_ID = {r["id"]: r for r in CORPUS}

#: A query that is exactly `r1`.
QUERY_AT_R1 = list(CORPUS[0]["vector"])


def query_at(record_id):
    """The vector of one record, as a query. Its own record must come back first."""
    return list(BY_ID[record_id]["vector"])


def create_table(name, dim=DIM, metric="cosine"):
    return post(
        "/tables",
        {"name": name, "dimension": dim, "distance_metric": metric},
    )


def insert_corpus(name, records=None):
    return post(f"/tables/{name}/records", {"records": records or CORPUS})


def ids_of(hits):
    """Ids in the order the server returned them. Order is part of the answer: ties are broken
    by id (`5c89817`), so a stable order is a property worth asserting, not an accident."""
    return [h["id"] for h in hits]
