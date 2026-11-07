"""The dashboard as the browser meets it.

`tests/dashboard_test.rs` covers the headers and the caching; this covers the thing only a
client can check — that **every file `index.html` asks for is actually served**. A partial
embed (someone adds an `#[include]` filter, or the bundle is rebuilt while the binary is not)
produces a page that returns 200 and then fails to start, with the real cause visible nowhere
but the browser console.

It also replays `dashboard/dashboard-routes.json` against the live server. The Rust contract
test asserts those routes are *listed*; this asserts they *answer*. The manifest holds two
writes -- correcting and forgetting a memory, the console's one exception to read-only -- and
those are never replayed blindly: the replay remembers a memory of its own under a fresh id,
corrects it, and forgets exactly the ids it made, so a live server's memories are not touched.

Skips unless the page is served, and **fails instead of skipping** when `DXMT_FEATURES` names
`api-dashboard` — a skipped test and an absent one look identical in a CI log.
"""

import contextlib
import json
import os
import pathlib
import re
import urllib.error
import urllib.request
import uuid

import pytest

from .helpers.client import BASE, call, succeeded

ROOT = pathlib.Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "dashboard" / "dashboard-routes.json"


def declared(feature):
    return feature in {f for f in re.split(r"[,\s]+", os.environ.get("DXMT_FEATURES", "")) if f}


def origin():
    return BASE[: -len("/api/v1")]


def raw(path):
    """Fetch something outside /api/v1, which the suite's client prefixes onto everything.

    Headers come back lowercased rather than as urllib's own mapping: that mapping is
    case-insensitive, `dict()` of it is not, and the difference is a `KeyError` on a header
    that is present.
    """
    with urllib.request.urlopen(f"{origin()}{path}", timeout=15) as response:
        headers = {name.lower(): value for name, value in response.headers.items()}
        return response.status, response.read(), headers


@pytest.fixture(scope="module")
def page():
    try:
        status, body, headers = raw("/dashboard/")
    except (urllib.error.URLError, OSError) as exc:
        message = (
            f"no dashboard at {origin()}/dashboard/ ({exc}): build the bundle with "
            f"`make dashboard-build`, compile with --features api-dashboard, and start the "
            f"server with --dashboard"
        )
        if declared("api-dashboard"):
            pytest.fail(f"DXMT_FEATURES says this server has api-dashboard, but there is {message}")
        pytest.skip(message)
    assert status == 200
    return body.decode(), headers


def test_the_page_is_the_app_shell(page):
    html, headers = page
    assert '<div id="root">' in html, "this is not the app's shell"
    assert headers["content-type"].startswith("text/html")


ASSET = r'/dashboard/[A-Za-z0-9_./-]+\.(?:js|css|svg|png|woff2?|txt)'


def test_every_asset_the_page_references_is_served(page):
    """The check that catches a partial or stale embed, which nothing server-side can see.

    Two levels deep, not one, and through the stylesheet as well as the script. `index.html`
    names the entry bundle and the stylesheet; the PCA worker is named *inside* the bundle and
    the webfont *inside* the stylesheet. A one-level scan goes green while either 404s — and
    both fail silently: a missing worker is a projection that never appears, a missing font is
    a page that quietly falls back to the system stack, and neither puts anything in a log.
    """
    html, _ = page
    referenced = set(re.findall(ASSET, html))
    assert referenced, "index.html references no assets at all, so nothing was bundled"

    for entry in [a for a in referenced if a.endswith(('.js', '.css'))]:
        _, body, _ = raw(entry)
        referenced |= set(re.findall(ASSET, body.decode('utf-8', 'replace')))

    referenced = sorted(referenced)
    assert any('worker' in a for a in referenced), (
        "no worker chunk is referenced; either the build stopped code-splitting it or this "
        "test stopped following the entry bundle, and both would hide a broken projection"
    )
    assert any(a.endswith('.woff2') for a in referenced), (
        "no webfont is referenced; the stylesheet is not being followed, or the font stopped "
        "being self-hosted — and a CDN font is a page that is blank on an air-gapped host"
    )

    missing = []
    for asset in referenced:
        try:
            status, body, headers = raw(asset)
        except urllib.error.HTTPError as exc:
            missing.append(f"{asset} -> {exc.code}")
            continue
        if status != 200 or len(body) == 0:
            missing.append(f"{asset} -> {status}, {len(body)} bytes")
        elif headers.get("content-type", "").startswith("application/octet-stream"):
            # A browser refuses to execute a script served as octet-stream, and the error it
            # reports names the file rather than the cause.
            missing.append(f"{asset} -> served as application/octet-stream")

    assert not missing, "assets the page needs and the server does not have:\n  " + "\n  ".join(
        missing
    )


def test_the_page_declares_an_icon_so_the_browser_stops_asking_for_one(page):
    """A browser requests `/favicon.ico` on every page whether anyone wants one or not.

    Without a declared icon that is a permanent 404 in the console of a tool whose whole job is
    showing people what is wrong — reported as one, too. Declaring it inline as a data URI costs
    no request and no embedded file, and this asserts it stays that way rather than drifting
    back to a fetched file that would have to be embedded and could go missing.
    """
    html, _ = page
    assert 'rel="icon"' in html, "no icon is declared, so /favicon.ico will 404 on every load"
    assert 'href="data:image/svg+xml' in html, (
        "the icon is fetched rather than inline; that is a request and a file that can go "
        "missing, for something the browser asks for unprompted"
    )


def test_a_client_side_route_survives_a_reload(page):
    status, body, headers = raw("/dashboard/tables")
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert '<div id="root">' in body.decode(), "a deep link should return the app, not a 404 page"


#: The dashboard's only writes, the same two `the_dashboard_declares_no_route_that_writes`
#: allows. The replay exercises them on memories it made itself, and refuses to send any other
#: write it finds in the manifest.
MEMORY_WRITES = {("POST", "/api/v1/memories"), ("DELETE", "/api/v1/memories/{r}")}


@contextlib.contextmanager
def a_memory_table():
    """The server's memory table, made for the replay only if the server has none.

    The memory table is made on first use with the server's embedding model, which the server
    under test does not have -- so without one, recall answers 400. One that exists (a live
    server people use) is used as it is and never dropped; one that does not is made with the
    loopback provider `test_request_fields.py` runs, and dropped afterwards.

    Looked for in the table list rather than with `GET /tables/memory`: on a server without a
    model that answers 400 ("made on first use with the server's embedding model"), not 404.
    """
    tables = call("GET", "/tables?limit=1000").json()["data"]["tables"]
    if any(t["name"] == "memory" for t in tables):
        yield
        return
    from .test_request_fields import memory_table

    with memory_table():
        yield


def test_every_route_the_dashboard_declares_answers(page, table):
    """The manifest, replayed. The Rust test asserts these are listed; this asserts they work."""
    manifest = json.loads(MANIFEST.read_text())
    bodies = {
        "/records/list": {"limit": 1},
        "/graph": {"sample": 4, "neighbours": 2},
        "/search": {"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1},
        "/search/text": {"query": "x", "limit": 1},
        # `limit`, not `top_k`: this endpoint has never had a `top_k`, and the body here said
        # otherwise for as long as unknown fields were ignored. The dashboard itself sends
        # `limit` -- it was this replay that was wrong, and it only surfaced when the file
        # stopped skipping.
        "/search/hybrid": {"vector": [1.0, 0.0, 0.0, 0.0], "limit": 1},
        "/memories/recall": {"query": "x", "limit": 1},
    }

    with a_memory_table():
        failures, writes = replay(manifest, table, bodies)
        failures += correct_and_forget()

    assert writes == MEMORY_WRITES, (
        f"the manifest's writes changed: {sorted(writes)}; the replay only knows how to "
        f"exercise {sorted(MEMORY_WRITES)} safely"
    )
    assert not failures, (
        "routes the dashboard declares and the server will not answer:\n  " + "\n  ".join(failures)
    )


def replay(manifest, table, bodies):
    """Every reading route, sent. Writes are collected and not sent."""
    failures, writes = [], set()
    for route in manifest["routes"]:
        key = (route["method"], route["path"])
        if key in MEMORY_WRITES:
            writes.add(key)
            continue
        if route["method"] not in {"GET", "POST"}:
            # Not sent: a write nobody taught this replay about is a failure, never a request.
            writes.add(key)
            failures.append(f"{route['method']} {route['path']} is a write the replay will not send")
            continue
        path = route["path"].replace("{t}", table).replace("{r}", "r1")
        body = next((b for suffix, b in bodies.items() if path.endswith(suffix)), None)

        if path.startswith("/api/v1"):
            response = call(route["method"], path[len("/api/v1") :], body, expect_ok=False)
        else:
            # Root-level probes: /status and /health/metrics.
            try:
                status, _, _ = raw(path)
            except urllib.error.HTTPError as exc:
                failures.append(f"{route['method']} {path} -> {exc.code} ({route['why']})")
                continue
            if status != 200:
                failures.append(f"{route['method']} {path} -> {status} ({route['why']})")
            continue

        if succeeded(response):
            continue
        # A 501 here is a capability answer, not a broken route, and there are two of them:
        # `FEATURE_DISABLED` when the build has no `text-bm25`, and `FULL_TEXT_NOT_ENABLED`
        # when it does but this table was created without full text. The dashboard renders
        # both as "not available here" rather than as an error, so both are tolerated.
        if response.status_code == 501:
            code = (response.json().get("error") or {}).get("code")
            if code in {"FEATURE_DISABLED", "FULL_TEXT_NOT_ENABLED", "NOT_IMPLEMENTED"}:
                continue
        failures.append(
            f"{route['method']} {path} -> {response.status_code} ({route['why']}): "
            f"{response.text[:160]}"
        )
    return failures, writes


def correct_and_forget():
    """The two writes, as the Memories page makes them, on memories this test made.

    A fresh id is given so the server's duplicate check -- which could answer with someone
    else's memory -- is skipped, and only the ids made here are forgotten.
    """
    failures, made = [], []
    original = f"dashboard-replay-{uuid.uuid4().hex}"
    try:
        first = call("POST", "/memories", {"text": "the dashboard replay remembered this",
                                           "id": original, "tags": ["dashboard-replay"]},
                     expect_ok=False)
        if not succeeded(first):
            return [f"POST /api/v1/memories -> {first.status_code}: {first.text[:160]}"]
        made.append(original)
        if first.json()["data"]["id"] != original:
            failures.append(f"remembering with an id answered another: {first.text[:160]}")

        # The page's default correction: a new memory that supersedes the old one.
        fixed = call("POST", "/memories", {"text": "the dashboard replay corrected this",
                                           "tags": ["dashboard-replay"],
                                           "supersedes": [original]}, expect_ok=False)
        if not succeeded(fixed):
            failures.append(f"POST /api/v1/memories (supersedes) -> {fixed.status_code}: "
                            f"{fixed.text[:160]}")
        else:
            data = fixed.json()["data"]
            made.append(data["id"])
            if data["superseded"] != [original]:
                failures.append(f"the correction superseded {data['superseded']}, not {original}")
    finally:
        for mid in made:
            gone = call("DELETE", f"/memories/{mid}", expect_ok=False)
            if not succeeded(gone) or gone.json()["data"]["forgotten"] != 1:
                failures.append(f"DELETE /api/v1/memories/{mid} -> {gone.status_code}: "
                                f"{gone.text[:160]}")
    return failures
