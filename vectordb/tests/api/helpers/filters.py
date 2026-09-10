"""Filter bodies in the shape the server actually reads.

Not the SDK's MongoDB shape (`{"rank": {"$gte": 3}}`) -- that is a client-side convenience the
SDK converts before sending, and a test written in it would be exercising the conversion rather
than the server. This is what goes on the wire:

    {"must": [{"key": "rank", "operation": {"type": "GreaterThanOrEqual", "value": 3}}]}

Operation types, from `FilterOperation` in `src/types/record.rs`: Equal, NotEqual, GreaterThan,
GreaterThanOrEqual, LessThan, LessThanOrEqual, In, NotIn, Contains, StartsWith, EndsWith.
"""


def cond(key, op, value):
    return {"key": key, "operation": {"type": op, "value": value}}


def must(*conditions):
    return {"must": list(conditions)}


def must_not(*conditions):
    return {"must_not": list(conditions)}


def should(*conditions):
    return {"should": list(conditions)}
