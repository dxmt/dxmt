// Package dxmt is a Go client for DXMT VectorDB: insert, read, update, delete and search over
// the server's HTTP API.
//
//	db, err := dxmt.New("http://127.0.0.1:9999")
//	err = db.CreateTable(ctx, dxmt.CreateTable{Name: "docs", Dimension: 768, DistanceMetric: dxmt.Cosine})
//	_, err = db.Insert(ctx, "docs", []dxmt.Record{{ID: "a", Vector: embedding}})
//	hits, err := db.Search(ctx, "docs", dxmt.NewSearch(query, 10))
//
// Every call takes a context. Nothing here holds state beyond the base URL and an http.Client,
// so one Client is safe to share across goroutines and better than one per goroutine -- the
// transport pools connections.
package dxmt

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"strings"
	"time"
)

// Client is a handle to one server.
type Client struct {
	base   string
	http   *http.Client
	apiKey string
}

// String keeps the key out of logs: %v and %+v would otherwise print every field, unexported
// ones included.
func (c *Client) String() string {
	key := ""
	if c.apiKey != "" {
		key = " api-key=<redacted>"
	}
	return "dxmt.Client{" + c.base + key + "}"
}

// GoString does the same for %#v.
func (c *Client) GoString() string { return c.String() }

// Option configures a Client.
type Option func(*Client)

// WithHTTPClient supplies your own http.Client -- for timeouts, proxies, or a tuned transport --
// so this package does not have to grow an option for each.
func WithHTTPClient(c *http.Client) Option {
	return func(client *Client) { client.http = c }
}

// WithAPIKey sends key as the api-key header on every request, for a server started with
// DXMT_API_KEY. Without it such a server answers 401 to everything but its health probes.
func WithAPIKey(key string) Option {
	return func(client *Client) { client.apiKey = key }
}

// New returns a client for the server at base, e.g. "http://127.0.0.1:9999". The /api/v1 prefix
// is added here, so callers pass the same URL they would put in a browser.
func New(base string, opts ...Option) (*Client, error) {
	base = strings.TrimRight(base, "/")
	if !strings.HasPrefix(base, "http://") && !strings.HasPrefix(base, "https://") {
		return nil, fmt.Errorf("dxmt: base url must start with http:// or https://, got %q", base)
	}
	c := &Client{base: base, http: &http.Client{Timeout: 60 * time.Second}}
	for _, opt := range opts {
		opt(c)
	}
	return c, nil
}

// Ready reports whether the server is up and its local durability is healthy. /ready is the
// deployment probe: it answers 503 once a WAL write has failed, where /health is a diagnostic
// document that still answers 200.
func (c *Client) Ready(ctx context.Context) bool {
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.base+"/ready", nil)
	if err != nil {
		return false
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	io.Copy(io.Discard, resp.Body)
	return resp.StatusCode/100 == 2
}

// ── tables ───────────────────────────────────────────────────────────────────────────────

// CreateTable makes a table. Dimension is fixed for its life.
func (c *Client) CreateTable(ctx context.Context, table CreateTable) (TableInfo, error) {
	var out TableInfo
	err := c.do(ctx, http.MethodPost, "/tables", table, &out)
	return out, err
}

// ListTables returns every table on the server.
func (c *Client) ListTables(ctx context.Context) ([]TableInfo, error) {
	var listing struct {
		Tables []TableInfo `json:"tables"`
	}
	err := c.do(ctx, http.MethodGet, "/tables", nil, &listing)
	return listing.Tables, err
}

// GetTable describes one table.
//
// POST /tables answers with the table flat and GET /tables/{t} answers with it nested under
// "table", alongside a record_count that is the live one -- the copy inside "table" is whatever
// was stamped when the table was described. Both unwrap to the same TableInfo here.
func (c *Client) GetTable(ctx context.Context, name string) (TableInfo, error) {
	var described struct {
		RecordCount uint64    `json:"record_count"`
		Table       TableInfo `json:"table"`
	}
	if err := c.do(ctx, http.MethodGet, "/tables/"+seg(name), nil, &described); err != nil {
		return TableInfo{}, err
	}
	described.Table.RecordCount = described.RecordCount
	return described.Table, nil
}

// DeleteTable drops a table and everything in it.
func (c *Client) DeleteTable(ctx context.Context, name string) error {
	return c.do(ctx, http.MethodDelete, "/tables/"+seg(name), nil, nil)
}

// ── records ──────────────────────────────────────────────────────────────────────────────

// Insert writes records, replacing any that are already there -- which is what makes this an
// upsert rather than an insert. The server caps one request at 10,000 records.
func (c *Client) Insert(ctx context.Context, table string, records []Record) (InsertOutcome, error) {
	body := struct {
		Records []Record `json:"records"`
	}{records}
	var out InsertOutcome
	err := c.do(ctx, http.MethodPost, "/tables/"+seg(table)+"/records", body, &out)
	return out, err
}

// Get reads one record. The error satisfies errors.As(&*APIError) with IsNotFound when there is
// no such record.
func (c *Client) Get(ctx context.Context, table, id string) (StoredRecord, error) {
	var out StoredRecord
	err := c.do(ctx, http.MethodGet, "/tables/"+seg(table)+"/records/"+seg(id), nil, &out)
	return out, err
}

// Update changes a record's vector, its payload, or both. See Update for what a nil vector
// means.
func (c *Client) Update(ctx context.Context, table, id string, update Update) error {
	return c.do(ctx, http.MethodPut, "/tables/"+seg(table)+"/records/"+seg(id), update, nil)
}

// Delete removes records by id and reports how many were actually removed -- which is not the
// number asked for when some of the ids were not there.
func (c *Client) Delete(ctx context.Context, table string, ids []string) (DeleteOutcome, error) {
	body := struct {
		IDs []string `json:"ids"`
	}{ids}
	var out DeleteOutcome
	err := c.do(ctx, http.MethodDelete, "/tables/"+seg(table)+"/records", body, &out)
	return out, err
}

// ── search ───────────────────────────────────────────────────────────────────────────────

// Search returns the nearest records to the query vector.
func (c *Client) Search(ctx context.Context, table string, query SearchQuery) (SearchResult, error) {
	var out SearchResult
	err := c.do(ctx, http.MethodPost, "/tables/"+seg(table)+"/search", query, &out)
	return out, err
}

// ── plumbing ─────────────────────────────────────────────────────────────────────────────

// seg percent-encodes one path segment.
//
// A record id goes into the URL path, and ids in a RAG corpus routinely carry characters the
// path reserves: "doc.md#3" stops at the '#', which is a fragment delimiter, so the request
// reaches /records/doc.md and the server answers about a record nobody asked for.
// "s3://bucket/key" turns one segment into three. url.PathEscape leaves '/' alone, which is
// exactly the character that must not survive, so this escapes the whole segment.
func seg(value string) string {
	return strings.ReplaceAll(url.QueryEscape(value), "+", "%20")
}

// do sends a request and unwraps the server's envelope into out. A body that says
// success: false, or an error status, becomes *APIError carrying the server's own code and
// message -- this package never invents one.
func (c *Client) do(ctx context.Context, method, path string, body, out any) error {
	var reader io.Reader
	if body != nil {
		encoded, err := json.Marshal(body)
		if err != nil {
			return fmt.Errorf("dxmt: encoding request for %s: %w", path, err)
		}
		reader = bytes.NewReader(encoded)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.base+"/api/v1"+path, reader)
	if err != nil {
		return fmt.Errorf("dxmt: building request for %s: %w", path, err)
	}
	if body != nil {
		req.Header.Set("Content-Type", "application/json")
	}
	if c.apiKey != "" {
		req.Header.Set("api-key", c.apiKey)
	}
	resp, err := c.http.Do(req)
	if err != nil {
		return fmt.Errorf("dxmt: %s %s: %w", method, path, err)
	}
	defer resp.Body.Close()
	raw, err := io.ReadAll(resp.Body)
	if err != nil {
		return fmt.Errorf("dxmt: reading response from %s: %w", path, err)
	}

	var env envelope
	if err := json.Unmarshal(raw, &env); err != nil {
		return fmt.Errorf("dxmt: unexpected response from %s: %w", path, err)
	}
	if !env.Success || present(env.Error) {
		return apiError(resp.StatusCode, env)
	}
	if out == nil {
		return nil
	}
	if !present(env.Data) {
		return &APIError{Status: resp.StatusCode, Code: "MALFORMED_RESPONSE",
			Message: "success with no data"}
	}
	if err := json.Unmarshal(env.Data, out); err != nil {
		return fmt.Errorf("dxmt: unexpected data from %s: %w", path, err)
	}
	return nil
}

// present reports whether a raw field carries something.
//
// A JSON null decodes into a json.RawMessage holding the four bytes "null", not an empty one,
// so `len(raw) > 0` is true for every `"error": null` the server sends -- which is every
// successful response. Checking the length alone turned every success into a failure.
func present(raw json.RawMessage) bool {
	return len(raw) > 0 && string(raw) != "null"
}

// apiError reads both error shapes the server produces.
func apiError(status int, env envelope) error {
	e := &APIError{Status: status, Code: "MALFORMED_RESPONSE",
		Message: "the server refused without saying why"}
	if !present(env.Error) {
		return e
	}
	// The enveloped shape: {"code": ..., "message": ..., "details": ...}.
	var structured APIError
	if json.Unmarshal(env.Error, &structured) == nil && structured.Code != "" {
		structured.Status = status
		return &structured
	}
	// The bare shape: {"error": "Validation failed", "details": "..."}. No code to pass
	// through, so say where the absence comes from rather than inventing one a caller might
	// match on.
	var message string
	if json.Unmarshal(env.Error, &message) == nil {
		return &APIError{Status: status, Code: "UNSTRUCTURED", Message: message,
			Details: env.Details}
	}
	return e
}
