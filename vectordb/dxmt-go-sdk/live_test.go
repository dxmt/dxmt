package dxmt_test

// The client against a real server. Skips itself when there is none.
//
//	cd vectordb && cargo run -- serve &
//	cd vectordb/dxmt-go-sdk && go test ./...
//
// DXMT_URL points it elsewhere. Every test makes its own table and drops it, so they can run
// against a server that holds other data, in any order.

import (
	"context"
	"errors"
	"os"
	"testing"

	dxmt "github.com/dxmt/dxmt/vectordb/dxmt-go-sdk"
)

const dim = 8

func url() string {
	if u := os.Getenv("DXMT_URL"); u != "" {
		return u
	}
	return "http://127.0.0.1:9999"
}

// connect returns nil after skipping when nothing is listening: the unit of this module is a
// client, and a client with no server to talk to has nothing to assert.
func connect(t *testing.T) (*dxmt.Client, context.Context) {
	t.Helper()
	var opts []dxmt.Option
	if key := os.Getenv("DXMT_API_KEY"); key != "" {
		// A server started with DXMT_API_KEY needs it on every request.
		opts = append(opts, dxmt.WithAPIKey(key))
	}
	db, err := dxmt.New(url(), opts...)
	if err != nil {
		t.Fatalf("bad url: %v", err)
	}
	ctx := context.Background()
	if !db.Ready(ctx) {
		t.Skipf("no DXMT server at %s", url())
	}
	return db, ctx
}

func vector(i int) []float32 {
	v := make([]float32, dim)
	for d := range v {
		v[d] = 0.05
	}
	v[i%dim] = 1.0
	return v
}

func freshTable(t *testing.T, db *dxmt.Client, ctx context.Context, name string) {
	t.Helper()
	_ = db.DeleteTable(ctx, name)
	if _, err := db.CreateTable(ctx, dxmt.CreateTable{
		Name: name, Dimension: dim, DistanceMetric: dxmt.Cosine,
	}); err != nil {
		t.Fatalf("create table: %v", err)
	}
	t.Cleanup(func() { _ = db.DeleteTable(ctx, name) })
}

func TestRecordSurvivesInsertReadUpdateAndDelete(t *testing.T) {
	db, ctx := connect(t)
	table := "go_sdk_crud"
	freshTable(t, db, ctx, table)

	inserted, err := db.Insert(ctx, table, []dxmt.Record{
		{ID: "r1", Vector: vector(1), Payload: map[string]any{"colour": "red", "n": 1}},
	})
	if err != nil {
		t.Fatalf("insert: %v", err)
	}
	if inserted.Count != 1 {
		t.Fatalf("inserted %d, want 1", inserted.Count)
	}

	read, err := db.Get(ctx, table, "r1")
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	if read.ID != "r1" || read.Payload()["colour"] != "red" {
		t.Fatalf("read back %+v", read)
	}
	if len(read.Vector) != dim || read.Vector[1] != 1.0 {
		t.Fatalf("the vector came back changed: %v", read.Vector)
	}

	// Merge keeps the keys it does not mention; replace would drop "n".
	if err := db.Update(ctx, table, "r1", dxmt.Update{
		Payload: map[string]any{"colour": "blue"}, Merge: true,
	}); err != nil {
		t.Fatalf("update: %v", err)
	}
	read, _ = db.Get(ctx, table, "r1")
	if read.Payload()["colour"] != "blue" {
		t.Fatalf("update did not take: %v", read.Payload())
	}
	if read.Payload()["n"] == nil {
		t.Fatal("merge must keep the untouched key")
	}

	deleted, err := db.Delete(ctx, table, []string{"r1", "never-existed"})
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	if deleted.Deleted != 1 {
		t.Fatalf("deleted %d, want 1: only the id that was there counts", deleted.Deleted)
	}

	_, err = db.Get(ctx, table, "r1")
	var apiErr *dxmt.APIError
	if !errors.As(err, &apiErr) || !apiErr.IsNotFound() {
		t.Fatalf("expected a 404 after delete, got %v", err)
	}
}

func TestSearchRanksOwnVectorFirstAndRespectsAFilter(t *testing.T) {
	db, ctx := connect(t)
	table := "go_sdk_search"
	freshTable(t, db, ctx, table)

	records := make([]dxmt.Record, 6)
	for i := range records {
		parity := "odd"
		if i%2 == 0 {
			parity = "even"
		}
		records[i] = dxmt.Record{
			ID: string(rune('a' + i)), Vector: vector(i),
			Payload: map[string]any{"parity": parity},
		}
	}
	if _, err := db.Insert(ctx, table, records); err != nil {
		t.Fatalf("insert: %v", err)
	}

	got, err := db.Search(ctx, table, dxmt.NewSearch(vector(3), 3))
	if err != nil {
		t.Fatalf("search: %v", err)
	}
	if len(got.Hits) != 3 {
		t.Fatalf("got %d hits, want 3", len(got.Hits))
	}
	if got.Hits[0].ID != "d" {
		t.Fatalf("a record is its own nearest neighbour; got %q", got.Hits[0].ID)
	}
	if got.Hits[0].Payload == nil {
		t.Fatal("payload was asked for and not returned")
	}

	query := dxmt.NewSearch(vector(3), 3)
	query.Filter = dxmt.All(dxmt.Eq("parity", "even"))
	filtered, err := db.Search(ctx, table, query)
	if err != nil {
		t.Fatalf("filtered search: %v", err)
	}
	if len(filtered.Hits) == 0 {
		t.Fatal("the filter matched nothing at all")
	}
	for _, hit := range filtered.Hits {
		if hit.ID == "d" {
			t.Fatal("'d' is odd and must not survive the filter")
		}
	}

	query = dxmt.NewSearch(vector(0), 1)
	query.WithVector = true
	withVector, _ := db.Search(ctx, table, query)
	if len(withVector.Hits[0].Vector) != dim {
		t.Fatal("WithVector should return the stored vector")
	}
}

func TestIDWithReservedCharactersAddressesTheRightRecord(t *testing.T) {
	db, ctx := connect(t)
	table := "go_sdk_ids"
	freshTable(t, db, ctx, table)

	// The shapes a RAG corpus actually produces. Unencoded, '#' truncates the path at the
	// fragment and '/' splits one segment into two, so the request lands on a different record
	// -- which is worse than an error, because it succeeds.
	awkward := []string{"README.md#3", "s3://bucket/key.pdf", "page 12", "a+b&c=d", "100%"}
	records := make([]dxmt.Record, len(awkward))
	for i, id := range awkward {
		records[i] = dxmt.Record{ID: id, Vector: vector(i), Payload: map[string]any{"n": i}}
	}
	if _, err := db.Insert(ctx, table, records); err != nil {
		t.Fatalf("insert: %v", err)
	}

	for i, id := range awkward {
		read, err := db.Get(ctx, table, id)
		if err != nil {
			t.Fatalf("get %q: %v", id, err)
		}
		if read.ID != id {
			t.Fatalf("asked for %q, got %q", id, read.ID)
		}
		if n, _ := read.Payload()["n"].(float64); int(n) != i {
			t.Fatalf("%q addressed the wrong record: payload %v", id, read.Payload())
		}
		if err := db.Update(ctx, table, id, dxmt.Update{
			Payload: map[string]any{"seen": true}, Merge: true,
		}); err != nil {
			t.Fatalf("update %q: %v", id, err)
		}
		after, _ := db.Get(ctx, table, id)
		if after.Payload()["seen"] != true {
			t.Fatalf("%q updated the wrong record", id)
		}
	}

	deleted, err := db.Delete(ctx, table, awkward)
	if err != nil {
		t.Fatalf("delete: %v", err)
	}
	if deleted.Deleted != len(awkward) {
		t.Fatalf("deleted %d of %d awkward ids", deleted.Deleted, len(awkward))
	}
}

func TestServerErrorReachesTheCallerIntact(t *testing.T) {
	db, ctx := connect(t)
	table := "go_sdk_errors"
	freshTable(t, db, ctx, table)

	// Wrong dimension: the server refuses through the un-enveloped path, and the client must
	// pass its reason through rather than flattening it into "request failed".
	_, err := db.Insert(ctx, table, []dxmt.Record{{ID: "bad", Vector: []float32{1, 2}}})
	var apiErr *dxmt.APIError
	if !errors.As(err, &apiErr) {
		t.Fatalf("a 2-dimensional vector in an 8-dimensional table must be refused, got %v", err)
	}
	if apiErr.Message == "" {
		t.Fatalf("the server's reason should survive: %+v", apiErr)
	}

	_, err = db.Search(ctx, "no_such_table", dxmt.NewSearch(vector(0), 1))
	if !errors.As(err, &apiErr) || !apiErr.IsNotFound() {
		t.Fatalf("searching a missing table should 404, got %v", err)
	}
}
