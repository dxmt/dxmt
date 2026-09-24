"""Every example in the OpenAPI document is a request this server accepts.

Swagger UI fills "Try it out" from the operation's example, and when there is none it invents a
body from the schema. For this API that invented body is `{"name": "string", "dimension": 0}`,
which the server refuses -- so the page was something you could read and not something you could
run. Each operation with a body now carries an example, and they are written to be run in order
on a four-dimensional table called `demo`: create it, fill it, ask it things, change it, take it
apart.

This pins them. An example that drifts from its handler is worse than no example, because it is
a documented request that fails -- the same rot that once left nine annotated handlers out of
the spec entirely.

Skipped unless the server under test was built with `api-swagger` and started with `--swagger`;
there is no document to check otherwise.
"""

import json
import os
import re
import urllib.error
import urllib.request

import pytest

from .helpers.client import BASE, call, delete, get, post

#: The order the examples tell their story in. `{t}` and `{r}` are the table and record the
#: examples name themselves, so a path here matches the spec key after substitution.
WALKTHROUGH = [
    ("POST", "/tables"),
    ("POST", "/tables/{t}/records"),
    ("POST", "/tables/{t}/records/bulk"),
    ("GET", "/tables/{t}"),
    ("GET", "/tables/{t}/stats"),
    ("POST", "/tables/{t}/search"),
    ("POST", "/tables/{t}/search/groups"),
    ("POST", "/tables/{t}/records/list"),
    ("POST", "/tables/{t}/records/count"),
    ("GET", "/tables/{t}/records/{r}"),
    ("PUT", "/tables/{t}/records/{r}"),
    ("POST", "/tables/{t}/search/hybrid"),
    ("POST", "/tables/{t}/hybrid_search"),
    ("POST", "/tables/{t}/search/text"),
    ("POST", "/query"),
    ("PATCH", "/tables/{t}"),
    ("POST", "/tables/{t}/branch"),
    ("POST", "/tables/{t}/index/rebuild"),
    ("DELETE", "/tables/{t}/index"),
    ("POST", "/snapshot"),
    ("DELETE", "/tables/{t}/records"),
    ("GET", "/health/metrics"),
]

TABLE = "demo"
RECORD = "doc-1"

#: `POST /backup` and `POST /tables/{t}/search/text` are documented and excluded from the run,
#: for opposite reasons. Backup writes a directory tree wherever its `path` says, and running a
#: documentation example must not leave a copy of the database on the machine under test.
#: Text search needs `text-bm25`, which is off by default; its example is checked for existence
#: and correctness of shape, and answering `FEATURE_DISABLED` is the right answer for a build
#: without it.
NOT_RUN = {
    ("POST", "/backup"),
    # Export and import name a path on the server's filesystem, inside DXMT_BACKUP_ROOT. Running
    # the documented examples would leave a copy of a table on the machine under test and would
    # need the example's path to be one this run configured -- the same reason `/backup` is here.
    # `tests/table_export_test.rs` runs both ends against a root it owns.
    ("POST", "/tables/{t}/export"),
    ("POST", "/tables/import"),
}
MAY_BE_DISABLED = {("POST", "/tables/{t}/search/text")}


def spec_path(template):
    return "/api/v1" + template.replace("{t}", "{table_name}").replace("{r}", "{record_id}")


def declared(feature):
    """Whether the run said the server was built with this feature. `scripts/api_tests.sh`
    exports `DXMT_FEATURES`; when it names a feature, its absence is a failure rather than a
    skip, because a skipped test and an absent one look identical in a CI log."""
    return feature in {f for f in re.split(r"[,\s]+", os.environ.get("DXMT_FEATURES", "")) if f}


@pytest.fixture(scope="module")
def spec():
    """The served OpenAPI document, or a skip when this build does not serve one.

    Fetched with urllib rather than the suite's client because the document sits outside
    `/api/v1`, which that client prefixes onto everything.
    """
    root = BASE[: -len("/api/v1")]
    try:
        with urllib.request.urlopen(f"{root}/api/docs/openapi.json", timeout=10) as response:
            return json.load(response)
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as exc:
        message = (f"no OpenAPI document at {root}/api/docs/openapi.json ({exc}): build with "
                   f"--features api-swagger and start the server with --swagger")
        if declared("api-swagger"):
            pytest.fail(f"DXMT_FEATURES says the server has api-swagger, but there is {message}")
        pytest.skip(message)


def example_for(spec, method, template):
    """The request body example for one operation, or None where the operation takes no body."""
    path = spec_path(template)
    operation = spec["paths"].get(path, {}).get(method.lower())
    assert operation is not None, f"{method} {path} is not in the OpenAPI document"
    body = operation.get("requestBody")
    if body is None:
        return None
    content = body["content"]["application/json"]
    assert "example" in content, (
        f"{method} {path} takes a body and documents no example, so Swagger UI will invent one "
        f"from the schema and the request will fail"
    )
    return content["example"]


def disabled_feature(response):
    """A 501 that names a feature flag rather than a defect in the example.

    Not an excuse when the run declared the feature: then the example failing is the finding.
    """
    if response.status_code != 501:
        return False
    error = (response.json() or {}).get("error") or {}
    if error.get("code") != "FEATURE_DISABLED":
        return False
    assert not declared("text-bm25"), (
        f"DXMT_FEATURES says the server has text-bm25, but it answered FEATURE_DISABLED: "
        f"{response.text[:200]}"
    )
    return True


@pytest.fixture
def clean_slate():
    """The walkthrough creates `demo` itself; a leftover would fail the first step for the
    wrong reason, and `demo_experiment` is what the branch example makes."""
    for table in (TABLE, f"{TABLE}_experiment"):
        delete(f"/tables/{table}", expect_ok=False)
    yield
    for table in (TABLE, f"{TABLE}_experiment"):
        delete(f"/tables/{table}", expect_ok=False)


def test_the_documented_walkthrough_runs(spec, clean_slate):
    """Every example, in order, against a real server -- which is what "try it out" does."""
    failures = []
    for method, template in WALKTHROUGH:
        body = example_for(spec, method, template)
        path = template.replace("{t}", TABLE).replace("{r}", RECORD)
        response = call(method, path, body, expect_ok=False)
        if (method, template) in MAY_BE_DISABLED and disabled_feature(response):
            continue
        if response.status_code >= 400:
            failures.append(f"{method} {path} -> {response.status_code} {response.text[:200]}")

    assert not failures, "documented examples the server refuses:\n  " + "\n  ".join(failures)


def test_the_search_example_finds_the_records_the_insert_example_writes(spec, clean_slate):
    """The one assertion a status code cannot make: that the examples are about each other.

    The search example filters on `topic = rust`, which only means something if the records the
    insert examples write carry that payload. Both halves could return 200 forever while having
    nothing to do with one another.
    """
    post("/tables", example_for(spec, "POST", "/tables"))
    post(f"/tables/{TABLE}/records", example_for(spec, "POST", "/tables/{t}/records"))

    hits = post(f"/tables/{TABLE}/search", example_for(spec, "POST", "/tables/{t}/search")).json()
    returned = hits["data"]["hits"]
    assert returned, (
        "the search example returns nothing: its filter and the payloads the insert example "
        "writes have drifted apart"
    )
    assert all(hit["payload"]["topic"] == "rust" for hit in returned), (
        f"the search example's filter is not being applied to the example records: {returned}"
    )

    stored = get(f"/tables/{TABLE}/records/{RECORD}").json()["data"]
    assert stored["id"] == RECORD, (
        "the path-parameter example names a record the insert example does not create"
    )


def test_no_operation_with_a_body_is_left_without_an_example(spec):
    """The condition that made the page unusable, asserted directly: a body with no example is a
    body Swagger UI fills from the schema, and no schema-derived body for this API validates."""
    missing = []
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            body = operation.get("requestBody")
            if body is None:
                continue
            content = body.get("content", {}).get("application/json", {})
            if "example" not in content and "examples" not in content:
                missing.append(f"{method.upper()} {path}")
    assert not missing, "request bodies with no example:\n  " + "\n  ".join(missing)


def test_every_documented_example_is_either_run_or_deliberately_not(spec):
    """So that an endpoint added with an example does not quietly go unexercised."""
    covered = {(method, spec_path(template)) for method, template in WALKTHROUGH}
    covered |= {(method, spec_path(template)) for method, template in NOT_RUN}
    unexercised = []
    for path, operations in spec["paths"].items():
        for method, operation in operations.items():
            if operation.get("requestBody") is None:
                continue
            if (method.upper(), path) not in covered:
                unexercised.append(f"{method.upper()} {path}")
    assert not unexercised, (
        "documented examples that nothing runs -- add them to WALKTHROUGH, or to NOT_RUN with "
        "the reason:\n  " + "\n  ".join(unexercised)
    )


def test_every_route_the_walkthrough_covers_is_documented(spec):
    """A route dropped from the spec is a route the page stops being a console for."""
    documented = {
        (method.upper(), path)
        for path, operations in spec["paths"].items()
        for method in operations
    }
    undocumented = [
        f"{method} {spec_path(template)}"
        for method, template in WALKTHROUGH
        if (method, spec_path(template)) not in documented
    ]
    assert not undocumented, "in the walkthrough, absent from the spec:\n  " + "\n  ".join(
        undocumented
    )
