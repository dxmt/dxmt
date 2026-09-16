package dxmt

import (
	"encoding/json"
	"fmt"
)

// envelope wraps every response the server sends.
//
// Four routes answer outside it, with a bare {"error": "...", "details": "..."} and no
// "success" field -- batch insert's validation and overload paths, and two in the DXQL handler.
// That is a server inconsistency rather than a second contract, but a client cannot wish it
// away: the most common first mistake anyone makes is a vector of the wrong dimension, and that
// is one of the four. Both shapes decode here.
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

// DistanceMetric is how a table measures distance. The server takes these lowercase.
type DistanceMetric string

const (
	Cosine     DistanceMetric = "cosine"
	Euclidean  DistanceMetric = "euclidean"
	DotProduct DistanceMetric = "dotproduct"
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
		Vector    []float32      `json:"vector,omitempty"`
		Payload   map[string]any `json:"payload,omitempty"`
		MergeMode string         `json:"merge_mode,omitempty"`
		Upsert    bool           `json:"upsert,omitempty"`
	}
	w := wire{Vector: u.Vector, Payload: u.Payload, Upsert: u.Upsert}
	if u.Merge {
		w.MergeMode = "merge"
	}
	return json.Marshal(w)
}

// Operation is a payload comparison, tagged the way the server expects:
// {"type": "Equal", "value": ...}.
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
