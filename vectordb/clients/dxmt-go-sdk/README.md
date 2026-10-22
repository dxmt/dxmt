# dxmt-go-sdk

Go client for DXMT VectorDB: insert, read, update, delete and search over the server's HTTP API.

Zero dependencies beyond the standard library, and its own module — a client's dependencies have
no business reaching the server binary.

## Use

```bash
go get github.com/dxmt/dxmt/vectordb/clients/dxmt-go-sdk@main
```

No version is tagged yet, so name a branch or commit. The module lives in a subdirectory of
the repository and needs no registry; from a local checkout, point at it instead:

```bash
go mod edit -replace github.com/dxmt/dxmt/vectordb/clients/dxmt-go-sdk=/path/to/dxmt/vectordb/clients/dxmt-go-sdk
```

```go
import dxmt "github.com/dxmt/dxmt/vectordb/clients/dxmt-go-sdk"

db, err := dxmt.New("http://127.0.0.1:9999")
_, err = db.CreateTable(ctx, dxmt.CreateTable{Name: "docs", Dimension: 768, DistanceMetric: dxmt.Cosine})

_, err = db.Insert(ctx, "docs", []dxmt.Record{
    {ID: "a", Vector: embedding, Payload: map[string]any{"topic": "go"}},
})

record, err := db.Get(ctx, "docs", "a")          // record.Payload() parses payload_json
err = db.Update(ctx, "docs", "a", dxmt.Update{Payload: map[string]any{"topic": "systems"}, Merge: true})

query := dxmt.NewSearch(queryVector, 10)
query.Filter = dxmt.All(dxmt.Eq("topic", "systems"))
hits, err := db.Search(ctx, "docs", query)

removed, err := db.Delete(ctx, "docs", []string{"a"})   // removed.Deleted, not the ask
```

Errors that came from the server are `*APIError` and carry its own code and message:

```go
var apiErr *dxmt.APIError
if errors.As(err, &apiErr) && apiErr.IsNotFound() { ... }
```

### Authentication

A server started with `DXMT_API_KEY` answers 401 to everything but its health probes until the key is sent. It goes out as the `api-key` header; send it over TLS, or it is readable at every hop.

```go
db, err := dxmt.New("https://db.example.com", dxmt.WithAPIKey("your-key"))
```

Printing the client with `%v`, `%+v` or `%#v` shows `<redacted>` in place of the key.

## What the API does that the types make explicit

* **A filter is always `must` / `must_not` / `should`.** The server's `Filter` has those three
  fields and ignores unknown ones, so a bare condition is not rejected — it decodes to a filter
  with no conditions and matches everything. `All`, `Any` and `None` are the only constructors.
* **`TopK` and `Limit` are both sent.** The server reads both, and a `Limit` left at its default
  of 10 truncates a larger `TopK` without saying so. `NewSearch` sets them together.
* **Delete reports what it removed**, not what was asked for: `Deleted` is 1 when you ask for two
  ids and only one was there.
* **An update with a nil vector keeps the stored one**, along with the sparse and named vectors.
  That is what makes a payload-only update safe on a table used for hybrid search.
* **Record ids are percent-encoded into the path.** `doc.md#3` and `s3://bucket/key.pdf` are
  ordinary ids in a RAG corpus and both break an unencoded URL — the first at the fragment
  delimiter, the second by turning one path segment into three.
* **Errors keep the server's words.** A refusal arrives as `*APIError` with the server's own
  `Code` and `Message`; a body that is not JSON (a proxy's 502 page, an empty 404 for a path
  that does not exist) arrives as a plain error naming the HTTP status; a server that never
  answered wraps the `net` error. One route on the server, `POST /query`, still answers with a
  bare `{"error": "..."}`; this client does not call it, and would read it as code
  `UNSTRUCTURED` if it did.
* **Every `DistanceMetric` constant creates a table.** `DotProduct` was `"dotproduct"` in
  1.0.0, a spelling the server rejects; it is `"dot"` now, and `Manhattan` exists. The live
  test creates one table per constant.

## Run it

```bash
cd vectordb && cargo run -- serve          # standard mode, port 9999, NOT --simple
cd vectordb/clients/dxmt-go-sdk
go run ./examples/basic_crud               # the five operations, end to end
go test ./...                              # live tests; they skip when no server answers
DXMT_URL=http://host:9999 go test ./...    # point either elsewhere
```

The tests skip rather than fail when nothing is listening, so `go test` is safe in a checkout
with no server. A skip proves nothing, though: watch them run before believing a change here.
