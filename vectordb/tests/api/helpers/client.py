"""The smallest HTTP client that makes these tests readable.

Deliberately not the Python SDK, for three reasons in ascending order of importance:

1. The SDK is itself under test elsewhere, and a bug in it would make this suite lie about the
   server.
2. **The SDK normalises the wire format.** It renames `payload` to `metadata` and unwraps the
   `{"success": ..., "data": ...}` envelope. `test_payload_comes_back_and_matches_what_went_in`
   asserted through the SDK would be asserting that the SDK's rename round-trips, which is
   precisely the bug class this suite exists to catch.
3. **A validating client cannot send the requests these tests need.** `test_a_wrong_dimension_is
   _refused` puts a 2-dimensional vector at a 4-dimensional table; a client that checks locally
   either rejects it before the wire or cannot express it.

`httpx` rather than `requests` because httpx is the SDK's one hard dependency and is therefore
already present wherever these tests can run; requests is not, and CI installs neither by
default.
"""

import os
import time

import httpx

BASE = os.environ.get("DXMT_URL", "http://localhost:9999").rstrip("/") + "/api/v1"
TIMEOUT = float(os.environ.get("DXMT_TIMEOUT", "30"))

#: One connection pool for the whole session. The functional tests would not notice, but the
#: property suite pages thousands of records and would otherwise pay TCP setup for each.
_HTTP = httpx.Client(timeout=TIMEOUT)


class ApiError(AssertionError):
    """A request the test did not expect to fail. Carries the body, which is where the reason is."""

    def __init__(self, method, path, response):
        body = response.text[:2000]
        super().__init__(f"{method} {path} -> {response.status_code}\n{body}")
        self.response = response


def succeeded(response):
    """2xx. Deliberately not httpx's `is_success` alias nor requests' `.ok`: `.ok` is
    `status_code < 400`, which counts a redirect as success, and a test that asserts a malformed
    request was *refused* must not be satisfied by one that was *moved*."""
    return 200 <= response.status_code < 300


def call(method, path, body=None, expect_ok=True):
    response = _HTTP.request(method, BASE + path, json=body)
    # Every exchange is checked against the OpenAPI document. See `helpers/openapi.py` for
    # what is a failure and what is only reported; a build without `api-swagger` serves no
    # document and this is a no-op.
    from . import openapi

    openapi.check(method, path, body, response, expect_ok=expect_ok)
    if expect_ok and not succeeded(response):
        raise ApiError(method, path, response)
    return response


def get(path, **kw):
    return call("GET", path, **kw)


def post(path, body=None, **kw):
    return call("POST", path, body, **kw)


def put(path, body=None, **kw):
    return call("PUT", path, body, **kw)


def delete(path, **kw):
    return call("DELETE", path, **kw)


def wait_for_server(seconds=60):
    """Poll `/health` until the server answers, so a suite started too early fails once with a
    clear message rather than reporting every test as broken."""
    deadline = time.time() + seconds
    last = None
    while time.time() < deadline:
        try:
            if succeeded(_HTTP.get(BASE + "/health", timeout=5)):
                return
        except httpx.RequestError as exc:  # noqa: PERF203 - the loop is the point
            last = exc
        time.sleep(1)
    raise RuntimeError(f"no server at {BASE} after {seconds}s (last error: {last})")
