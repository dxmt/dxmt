"""The documented semantics of DXMT, written without reference to the server's code.

Shared by the tests that check a server against a model of it — `tests/stateful` (a random walk)
and `tests/compat` (data written by an earlier release) — so that when a rule changes it changes
in one place, and the two cannot come to test different rules.

  * A condition on a field holding an array is satisfied when any element satisfies it.
  * `NotEqual` is the complement of `Equal` and is true for a missing field; `NotIn` requires the
    field to be present.
  * Numbers compare as numbers: `5 == 5.0`.
  * The key `id` means the record's own id.
  * Scores: cosine similarity, `-(-dot)` for dot product, `1 / (1 + d)` for euclidean distance.
"""

import math


def is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def same(a, b):
    if is_num(a) and is_num(b):
        return float(a) == float(b)
    return type(a) is type(b) and a == b


def elements(v):
    return v if isinstance(v, list) else [v]


def scalar_ok(x, op, t):
    if op == "Equal":
        return same(x, t)
    if op == "In":
        return any(same(x, y) for y in t)
    if op in ("GreaterThan", "GreaterThanOrEqual", "LessThan", "LessThanOrEqual"):
        if not (is_num(x) and is_num(t)):
            return False
        return {"GreaterThan": x > t, "GreaterThanOrEqual": x >= t, "LessThan": x < t, "LessThanOrEqual": x <= t}[op]
    if not isinstance(x, str):
        return False
    return {"Contains": t in x, "StartsWith": x.startswith(t), "EndsWith": x.endswith(t)}[op]


def cond_ok(rid, payload, c):
    op, t = c["operation"]["type"], c["operation"]["value"]
    if c["key"] == "id":
        present, v = True, rid
    else:
        present, v = c["key"] in payload, payload.get(c["key"])
    if op == "NotEqual":
        return not (present and any(same(x, t) for x in elements(v)))
    if op == "NotIn":
        return present and not any(same(x, y) for x in elements(v) for y in t)
    return present and any(scalar_ok(x, op, t) for x in elements(v))


def matches(rid, payload, f):
    if f.get("must") and not all(cond_ok(rid, payload, c) for c in f["must"]):
        return False
    if f.get("should") and not any(cond_ok(rid, payload, c) for c in f["should"]):
        return False
    if f.get("must_not") and any(cond_ok(rid, payload, c) for c in f["must_not"]):
        return False
    return True


def instant(stamp):
    """An RFC 3339 timestamp as something that orders correctly: the fraction padded to nine
    digits, since a serialiser may drop trailing zeros and "…58.2Z" must not sort after "…58.19Z"."""
    head, _, rest = stamp.rstrip("Z").partition(".")
    return head, (rest + "000000000")[:9]


def cosine(a, b):
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return sum(x * y for x, y in zip(a, b)) / (na * nb)


def score(metric, query, vector):
    """The score the API reports for `vector` against `query` under `metric`."""
    if metric == "cosine":
        return cosine(query, vector)
    if metric == "dot":
        return sum(x * y for x, y in zip(query, vector))
    if metric == "euclidean":
        return 1.0 / (1.0 + math.sqrt(sum((x - y) ** 2 for x, y in zip(query, vector))))
    raise ValueError(metric)
