package dxmt

import (
	"encoding/json"
	"fmt"
)

// envelope wraps every response the server sends.
//
// One route still answers outside it, with a bare {"error": "..."} and no "success" field:
// POST /query, the DXQL handler, which this client does not call. Batch insert's validation
// and overload paths used to as well -- the wrong-dimension refusal arrived bare -- and moved
// onto the envelope on 2026-09-27. The bare shape still decodes, because a client that rejected
// it would turn a future server regression into an unreadable error instead of a readable one.
type envelope struct {
	Success bool            `json:"success"`
	Data    json.RawMessage `json:"data"`
	Error   json.RawMessage `json:"error"`
	Details json.RawMessage `json:"details"`
}

// APIError is the server's refusal: a stable code, a message, and optional details.
type APIError struct {
	Status  int             `json:"-"`
	Code    string          `json:"code"`
	Message string          `json:"message"`
	Details json.RawMessage `json:"details,omitempty"`
}

func (e *APIError) Error() string {
	return fmt.Sprintf("%d (%s): %s", e.Status, e.Code, e.Message)
}

// IsNotFound reports whether this is the server saying the thing does not exist. Use it with
// errors.As:
//
//	var apiErr *dxmt.APIError
//	if errors.As(err, &apiErr) && apiErr.IsNotFound() { ... }
func (e *APIError) IsNotFound() bool { return e.Status == 404 }

// DistanceMetric is how a table measures distance, spelled the way the server parses it:
// cosine, euclidean, dot, manhattan.
//
// DotProduct was "dotproduct" in 1.0.0, a spelling the server has never accepted, so a table
// created with it was a 400 INVALID_DISTANCE_METRIC on every call. The live test used Cosine,
// so nothing noticed. Manhattan was missing.
type DistanceMetric string

const (
	Cosine     DistanceMetric = "cosine"
	Euclidean  DistanceMetric = "euclidean"
	DotProduct DistanceMetric = "dot"
	Manhattan  DistanceMetric = "manhattan"
)

// CreateTable describes a table to make. Dimension is required; everything else has a server
// default.
type CreateTable struct {
	Name           string         `json:"name"`
	Dimension      int            `json:"dimension"`
	DistanceMetric DistanceMetric `json:"distance_metric,omitempty"`
	Description    string         `json:"description,omitempty"`
	IndexType      string         `json:"index_type,omitempty"`
}

// Record is a record on its way in. Payload is free-form JSON.
type Record struct {
	ID      string         `json:"id"`
	Vector  []float32      `json:"vector"`
	Payload map[string]any `json:"payload,omitempty"`
}

// StoredRecord is a record on its way out. The server names the payload field payload_json and
// sends it as a string; Payload parses it so callers do not have to know that.
type StoredRecord struct {
	ID          string    `json:"id"`
	Vector      []float32 `json:"vector"`
	PayloadJSON string    `json:"payload_json"`
	Version     uint64    `json:"version"`
}

// Payload is the record's payload as a map, or nil when it has none or it does not parse.
func (r StoredRecord) Payload() map[string]any {
	if r.PayloadJSON == "" {
		return nil
	}
	var out map[string]any
	if json.Unmarshal([]byte(r.PayloadJSON), &out) != nil {
		return nil
	}
	return out
}

// Update is what an update changes. A nil Vector keeps the stored dense, sparse and named
// vectors, which is what makes a payload-only update safe on a table used for hybrid search.
type Update struct {
	Vector []float32 `json:"vector,omitempty"`
	// Payload replaces the stored payload, or merges into it when Merge is set.
	//
	// A nil map leaves the stored payload alone; an empty non-nil map (map[string]any{})
	// clears it. `omitempty` cannot tell those apart -- it drops an empty map as readily as a
	// nil one -- so clearing a payload was not expressible from this client at all: the field
	// never reached the wire and the server, which keeps what an update does not mention,
	// left the old payload in place.
	Payload map[string]any `json:"payload,omitempty"`
	// Merge sends merge_mode="merge": the keys given are merged into the stored payload
	// instead of replacing it.
	Merge bool `json:"-"`
	// Upsert creates the record when it is not there. Off by default, like the server.
	Upsert bool `json:"upsert,omitempty"`
}

// MarshalJSON renders Merge as the server's merge_mode field.
func (u Update) MarshalJSON() ([]byte, error) {
	type wire struct {
		Vector    []float32       `json:"vector,omitempty"`
		Payload   *map[string]any `json:"payload,omitempty"`
		MergeMode string          `json:"merge_mode,omitempty"`
		Upsert    bool            `json:"upsert,omitempty"`
	}
	// A pointer, so that a nil map omits the field and an empty one sends `"payload": {}` --
	// which is how the server is told to clear it.
	w := wire{Vector: u.Vector, Upsert: u.Upsert}
	if u.Payload != nil {
		w.Payload = &u.Payload
	}
	if u.Merge {
		w.MergeMode = "merge"
	}
	return json.Marshal(w)
}

// Operation is a payload comparison, tagged the way the server expects:
// {"type": "Equal", "value": ...}.
//
// omitempty on an `any` field omits only a nil interface, so false, 0 and "" are sent and
// filter as values (TestAFilterOnAFalsyValueFilters pins that); the operations that take no
// value -- IsEmpty, IsNull -- are the ones that omit it, which the server expects.
type Operation struct {
	Type  string `json:"type"`
	Value any    `json:"value,omitempty"`
}

// Condition is one clause of a filter: a payload key and what must be true of it.
type Condition struct {
	Key       string    `json:"key"`
	Operation Operation `json:"operation"`
}

// Eq is the common case: Eq("colour", "red").
func Eq(key string, value any) Condition {
	return Condition{Key: key, Operation: Operation{Type: "Equal", Value: value}}
}

// Ne, Gt, Gte, Lt, Lte cover the rest of the comparisons.
func Ne(key string, value any) Condition {
	return Condition{Key: key, Operation: Operation{Type: "NotEqual", Value: value}}
}
func Gt(key string, value any) Condition {
	return Condition{Key: key, Operation: Operation{Type: "GreaterThan", Value: value}}
}
func Gte(key string, value any) Condition {
	return Condition{Key: key, Operation: Operation{Type: "GreaterThanOrEqual", Value: value}}
}
func Lt(key string, value any) Condition {
	return Condition{Key: key, Operation: Operation{Type: "LessThan", Value: value}}
}
func Lte(key string, value any) Condition {
	return Condition{Key: key, Operation: Operation{Type: "LessThanOrEqual", Value: value}}
}
func In(key string, values ...any) Condition {
	return Condition{Key: key, Operation: Operation{Type: "In", Value: values}}
}

// Filter is always must / must_not / should, never a bare condition: the server's Filter has
// those three fields and encoding/json's counterpart on the Rust side ignores anything else, so
// an unwrapped condition does not fail -- it decodes to a filter with no conditions and matches
// everything. All, Any and None are the only ways to build one.
type Filter struct {
	Must    []Condition `json:"must,omitempty"`
	MustNot []Condition `json:"must_not,omitempty"`
	Should  []Condition `json:"should,omitempty"`
}

// All requires every condition to hold.
func All(conditions ...Condition) *Filter { return &Filter{Must: conditions} }

// Any requires at least one condition to hold.
func Any(conditions ...Condition) *Filter { return &Filter{Should: conditions} }

// None requires that no condition holds.
func None(conditions ...Condition) *Filter { return &Filter{MustNot: conditions} }

// SearchQuery is a nearest-neighbour search.
type SearchQuery struct {
	Vector []float32 `json:"vector"`
	// TopK and Limit are both sent. The server reads both, and a Limit left at its default of
	// 10 truncates a larger TopK without saying so; NewSearch sets them together.
	TopK        int     `json:"top_k"`
	Limit       int     `json:"limit"`
	Filter      *Filter `json:"filter,omitempty"`
	WithPayload bool    `json:"with_payload"`
	WithVector  bool    `json:"with_vector"`
}

// NewSearch builds a query that asks for k hits and their payloads.
func NewSearch(vector []float32, k int) SearchQuery {
	return SearchQuery{Vector: vector, TopK: k, Limit: k, WithPayload: true}
}

// SearchHit is one result.
type SearchHit struct {
	ID      string         `json:"id"`
	Score   float32        `json:"score"`
	Payload map[string]any `json:"payload"`
	Vector  []float32      `json:"vector"`
}

// SearchResult is what a search returns.
type SearchResult struct {
	Hits   []SearchHit `json:"hits"`
	Total  int         `json:"total"`
	TookMs uint64      `json:"tookms"`
}

// TableInfo describes a table.
type TableInfo struct {
	Name        string `json:"name"`
	Dimension   int    `json:"dimension"`
	RecordCount uint64 `json:"record_count"`
	IndexType   string `json:"index_type"`
}

// InsertOutcome is what an insert reports.
type InsertOutcome struct {
	Count       int      `json:"count"`
	InsertedIDs []string `json:"inserted_ids"`
}

// DeleteOutcome is what a delete reports. Deleted is how many records the server actually
// removed, which is not always how many were asked for: deleting ids that are not there removes
// none of them.
type DeleteOutcome struct {
	Deleted   int `json:"deleted_count"`
	Requested int `json:"requested_count"`
}
