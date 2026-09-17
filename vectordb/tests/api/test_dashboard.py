"""The dashboard as the browser meets it.

`tests/dashboard_test.rs` covers the headers and the caching; this covers the thing only a
client can check — that **every file `index.html` asks for is actually served**. A partial
embed (someone adds an `#[include]` filter, or the bundle is rebuilt while the binary is not)
produces a page that returns 200 and then fails to start, with the real cause visible nowhere
but the browser console.

It also replays `dashboard/dashboard-routes.json` against the live server. The Rust contract
test asserts those routes are *listed*; this asserts they *answer*.

Skips unless the page is served, and **fails instead of skipping** when `DXMT_FEATURES` names
`api-dashboard` — a skipped test and an absent one look identical in a CI log.
"""

import json
import os
import pathlib
import re
import urllib.error
import urllib.request

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


ASSET = r'/dashboard/[A-Za-z0-9_./-]+\.(?:js|css|svg|png|woff2?)'


def test_every_asset_the_page_references_is_served(page):
    """The check that catches a partial or stale embed, which nothing server-side can see.

    Two levels deep, not one. The entry bundle is named in `index.html`, but a code-split chunk
    is named *inside* the entry bundle — the PCA web worker is loaded that way. A one-level scan
    goes green while the worker 404s, and the symptom is a projection that silently never
    appears, with nothing in any log.
    """
    html, _ = page
    referenced = set(re.findall(ASSET, html))
    assert referenced, "index.html references no assets at all, so nothing was bundled"

    for entry in [a for a in referenced if a.endswith('.js')]:
        _, body, _ = raw(entry)
        referenced |= set(re.findall(ASSET, body.decode('utf-8', 'replace')))

    referenced = sorted(referenced)
    assert any('worker' in a for a in referenced), (
        "no worker chunk is referenced; either the build stopped code-splitting it or this "
        "test stopped following the entry bundle, and both would hide a broken projection"
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


def test_a_client_side_route_survives_a_reload(page):
    status, body, headers = raw("/dashboard/tables")
    assert status == 200
    assert headers["content-type"].startswith("text/html")
    assert '<div id="root">' in body.decode(), "a deep link should return the app, not a 404 page"


def test_every_route_the_dashboard_declares_answers(page, table):
    """The manifest, replayed. The Rust test asserts these are listed; this asserts they work."""
    manifest = json.loads(MANIFEST.read_text())
    bodies = {
        "/records/list": {"limit": 1},
        "/search": {"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1},
        "/search/text": {"query": "x", "limit": 1},
        "/search/hybrid": {"vector": [1.0, 0.0, 0.0, 0.0], "top_k": 1},
    }

    failures = []
    for route in manifest["routes"]:
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

    assert not failures, (
        "routes the dashboard declares and the server will not answer:\n  " + "\n  ".join(failures)
    )
