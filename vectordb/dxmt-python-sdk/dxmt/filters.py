"""Filters: a typed builder, and a translator from MongoDB-style dicts.

The server takes one shape: three flat lists of conditions,

    {"must": [...], "should": [...], "must_not": [...]}

meaning every `must` holds, at least one `should` holds (when there is any), and no `must_not`
holds. That is not every boolean formula. `(a and b) or c` has no such form, and neither has
`(a or b) and (c or d)`. Both ways of writing a filter here -- `Field` and the dict translator --
build that shape exactly or raise `ValueError` saying what cannot be expressed. They never
approximate: the translator this replaced turned `(a and b) or c` into `a or b or c`, dropped a
key written beside `$and`, and read a nested `$and` as a field named "$and", each of them a
query that ran and returned the wrong records.

    from dxmt import Field

    f = (Field("lang") == "en") & (Field("year") >= 2020) & ~Field("tags").is_in(["spam"])
    client.search("docs", vector, filter=f)

What a condition means is the server's, and is the same on every endpoint:

* a field holding an array matches when any element does, for every operation;
* `!=` / `NotEqual` is true of a record without the field, `not_in` / `NotIn` is not;
* numbers compare as numbers, so 5 matches 5.0, and ranges compare numbers only;
* the key `id` is the record id;
* comparing against an array or object is refused (`tags == ["a", "b"]`).

Two conditions on the same array field are tested independently: `year >= 2020` and
`year <= 2023` holds for `[2019, 2024]`, one element satisfying each.
"""

from typing import Any, Dict, Iterable, List, Optional


# MongoDB operator to DXMT operation type mapping
OPERATOR_MAP = {
    "$eq": "Equal",
    "$ne": "NotEqual",
    "$gt": "GreaterThan",
    "$gte": "GreaterThanOrEqual",
    "$lt": "LessThan",
    "$lte": "LessThanOrEqual",
    "$in": "In",
    "$nin": "NotIn",
    "$contains": "Contains",
    "$startswith": "StartsWith",
    "$endswith": "EndsWith",
}


def _map_operator(mongo_op: str) -> str:
    """Map MongoDB operator to DXMT operation type.

    Args:
        mongo_op: MongoDB operator like $gte, $lt, etc.

    Returns:
        DXMT operation type like GreaterThanOrEqual, LessThan, etc.

    Raises:
        ValueError: If operator is not supported.
    """
    if mongo_op not in OPERATOR_MAP:
        raise ValueError(
            f"Unsupported operator: {mongo_op}. "
            f"Supported operators: {list(OPERATOR_MAP.keys())}"
        )
    return OPERATOR_MAP[mongo_op]



_SCALAR = (str, int, float, bool)


def _condition(key: str, op_type: str, value: Any) -> Dict[str, Any]:
    """One condition in the server's shape, refusing values the server would refuse."""
    if not isinstance(key, str) or not key:
        raise ValueError(f"a filter key must be a non-empty string, not {key!r}")
    if op_type in ("In", "NotIn"):
        if not isinstance(value, (list, tuple)):
            raise ValueError(f"{op_type} on {key!r} takes a list of values, not {value!r}")
        value = list(value)
        bad = [v for v in value if not isinstance(v, _SCALAR)]
    elif op_type in ("Contains", "StartsWith", "EndsWith"):
        bad = [] if isinstance(value, str) else [value]
    elif op_type in ("GreaterThan", "GreaterThanOrEqual", "LessThan", "LessThanOrEqual"):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            # The server orders numbers only, so `date >= "2024-01-01"` matches nothing.
            raise ValueError(
                f"{op_type} on {key!r} compares against {value!r}; ranges compare numbers "
                "only (store dates as a number, such as a Unix timestamp)"
            )
        bad = []
    else:
        bad = [] if isinstance(value, _SCALAR) else [value]
    if bad:
        raise ValueError(
            f"{op_type} on {key!r} compares against {bad[0]!r}; a filter compares a field "
            "against strings, numbers and booleans only (an array field matches when any "
            "element does, so test the element instead)"
        )
    return {"key": key, "operation": {"type": op_type, "value": value}}


class Filter:
    """A filter in the server's flat shape. Combine with `&`, `|` and `~`.

    Each combination either has an exact flat form or raises `ValueError`; see the module
    docstring. Pass one anywhere a filter is taken, or call `to_dict()`.
    """

    __slots__ = ("must", "should", "must_not")

    def __init__(self, must=(), should=(), must_not=()):
        self.must: List[Dict[str, Any]] = list(must)
        self.should: List[Dict[str, Any]] = list(should)
        self.must_not: List[Dict[str, Any]] = list(must_not)

    def _single(self) -> Optional[Dict[str, Any]]:
        """The condition this is, when it is exactly one positive condition."""
        if len(self.must) == 1 and not self.should and not self.must_not:
            return self.must[0]
        return None

    def _positives(self) -> Optional[List[Dict[str, Any]]]:
        """The conditions of a plain disjunction: one condition, or a `should` alone."""
        single = self._single()
        if single is not None:
            return [single]
        if self.should and not self.must and not self.must_not:
            return list(self.should)
        return None

    def __and__(self, other: "Filter") -> "Filter":
        if not isinstance(other, Filter):
            return NotImplemented
        if self.should and other.should:
            # (a or b) and (c or d): distributing it is exponential, and one `should` list
            # cannot hold two independent disjunctions.
            raise ValueError(
                "cannot express two OR groups joined by AND; the filter has one `should` list"
            )
        return Filter(
            self.must + other.must,
            self.should or other.should,
            self.must_not + other.must_not,
        )

    def __or__(self, other: "Filter") -> "Filter":
        if not isinstance(other, Filter):
            return NotImplemented
        left, right = self._positives(), other._positives()
        if left is None or right is None:
            raise ValueError(
                "OR can only join single conditions (or other ORs of them); "
                "(a and b) or c has no form the server takes"
            )
        return Filter(should=left + right)

    def __invert__(self) -> "Filter":
        positives = self._positives()
        if positives is None:
            # not (a and b) is (not a) or (not b), and there is no negated condition to put
            # in `should`.
            raise ValueError("NOT can only negate a single condition or an OR of them")
        return Filter(must_not=positives)

    def __bool__(self) -> bool:
        # `a and b`, `not a` and `2020 <= Field("y") <= 2023` all go through bool() and would
        # quietly keep one side. Refusing is the only way to make them loud.
        raise TypeError(
            "a filter has no truth value: combine with &, | and ~ (not and/or/not), and write "
            "a range as Field(k).between(lo, hi) rather than a chained comparison"
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Filter) and self.to_dict() == other.to_dict()

    __hash__ = None  # type: ignore[assignment]

    def to_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for name in ("must", "should", "must_not"):
            conditions = getattr(self, name)
            if conditions:
                out[name] = [
                    {"key": c["key"], "operation": dict(c["operation"])} for c in conditions
                ]
        return out

    def __repr__(self) -> str:
        return f"Filter({self.to_dict()!r})"


class Field:
    """A payload field (or `"id"`, the record id), to build conditions on.

    >>> (Field("year") >= 2020) & (Field("lang") == "en")

    A dot reads into a nested payload -- `Field("meta.lang")` is the `lang` of the object under
    `meta`, at any depth. A payload that really holds a key with a dot in it keeps working:
    the server tries the whole name as a key before reading it as a path.
    """

    __slots__ = ("key",)

    def __init__(self, key: str):
        if not isinstance(key, str) or not key:
            raise ValueError(f"a field name must be a non-empty string, not {key!r}")
        self.key = key

    def _op(self, op_type: str, value: Any) -> Filter:
        return Filter(must=[_condition(self.key, op_type, value)])

    def __eq__(self, value: Any) -> Filter:  # type: ignore[override]
        return self._op("Equal", value)

    def __ne__(self, value: Any) -> Filter:  # type: ignore[override]
        return self._op("NotEqual", value)

    def __gt__(self, value: Any) -> Filter:
        return self._op("GreaterThan", value)

    def __ge__(self, value: Any) -> Filter:
        return self._op("GreaterThanOrEqual", value)

    def __lt__(self, value: Any) -> Filter:
        return self._op("LessThan", value)

    def __le__(self, value: Any) -> Filter:
        return self._op("LessThanOrEqual", value)

    __hash__ = None  # type: ignore[assignment]

    def is_in(self, values: Iterable[Any]) -> Filter:
        return self._op("In", list(values))

    def not_in(self, values: Iterable[Any]) -> Filter:
        """Present and not one of `values`; a record without the field does not match."""
        return self._op("NotIn", list(values))

    def between(self, low: Any, high: Any) -> Filter:
        """`low <= field <= high`. On an array field, each bound may be met by a different element."""
        return (self >= low) & (self <= high)

    def contains(self, text: str) -> Filter:
        return self._op("Contains", text)

    def starts_with(self, text: str) -> Filter:
        return self._op("StartsWith", text)

    def ends_with(self, text: str) -> Filter:
        return self._op("EndsWith", text)

    def __repr__(self) -> str:
        return f"Field({self.key!r})"


def _field_filter(key: str, value: Any) -> Filter:
    """`{key: value}` from a dict filter: equality, or `{"$op": v, ...}` all ANDed."""
    if key.startswith("$"):
        raise ValueError(f"unknown operator {key!r} at the top of a filter")
    if isinstance(value, dict):
        if not value:
            raise ValueError(f"empty condition for {key!r}")
        plain = [op for op in value if not op.startswith("$")]
        if plain:
            raise ValueError(
                f"{key!r}: {value!r} mixes operators with plain keys; comparing a field "
                "against an object is not supported"
            )
        result = Filter()
        for op, operand in value.items():
            result = result & Filter(must=[_condition(key, _map_operator(op), operand)])
        return result
    return Filter(must=[_condition(key, "Equal", value)])


def _dict_filter(filter_dict: Dict[str, Any]) -> Filter:
    if not isinstance(filter_dict, dict):
        raise ValueError(f"a filter is a dict, not {filter_dict!r}")
    result = Filter()
    # Every key is a term of an implicit AND -- including a key beside `$and` or `$or`.
    for key, value in filter_dict.items():
        if key in ("$and", "$or", "$nor", "$not"):
            if not isinstance(value, list) or not value:
                raise ValueError(f"{key} takes a non-empty list of filters, not {value!r}")
            parts = [_dict_filter(part) for part in value]
            if key == "$and":
                term = parts[0]
                for part in parts[1:]:
                    term = term & part
            else:
                term = parts[0]
                for part in parts[1:]:
                    term = term | part
                if key != "$or":
                    # $nor, and $not given a list as this SDK has always taken it: none of them.
                    term = ~term
        else:
            term = _field_filter(key, value)
        result = result & term
    return result


def to_filter(value: Any) -> Optional[Dict[str, Any]]:
    """Any filter this SDK takes, in the server's shape; None for no filter.

    Takes a `Filter` (from `Field`), a MongoDB-style dict, or the server's own shape, which is
    checked for stray keys and passed through.
    """
    if value is None:
        return None
    if isinstance(value, Filter):
        return value.to_dict() or None
    if not isinstance(value, dict):
        raise ValueError(f"a filter is a dict or a Filter, not {type(value).__name__}")
    if not value:
        return None
    native = {"must", "should", "must_not"}
    if native & value.keys():
        stray = value.keys() - native
        if stray:
            raise ValueError(
                f"a filter with must/should/must_not cannot also have {sorted(stray)}; "
                "they would be ignored"
            )
        return value
    return _dict_filter(value).to_dict() or None


def convert_simple_filter_to_dxmt(
    filter_dict: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Convert a filter to the server's shape. Kept under its old name; see `to_filter`.

    Accepts simple equality (`{"category": "AI"}`), operators
    (`{"year": {"$gte": 2020, "$lte": 2023}}`), `$and`, `$or`, `$nor` (and `$not` with a list,
    meaning the same), nested wherever the result has an exact form, the server's own shape, and
    a `Filter`. Raises `ValueError` for anything it cannot translate exactly.
    """
    return to_filter(filter_dict)
