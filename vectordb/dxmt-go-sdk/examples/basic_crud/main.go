// Insert, read, update, delete and search, against a running server.
//
//	cd vectordb && cargo run -- serve                          # in one terminal
//	cd vectordb/dxmt-go-sdk && go run ./examples/basic_crud
//
// DXMT_URL overrides the address. The table is created at the start and dropped at the end, so
// this can be run repeatedly. The Python and Rust equivalents do the same things in the same
// order, on purpose, so the three clients can be read side by side.
package main

import (
	"context"
	"errors"
	"fmt"
	"os"

	dxmt "github.com/dxmt/dxmt/vectordb/dxmt-go-sdk"
)

const (
	table = "go_sdk_demo"
	dim   = 8
)

// axis leans on one dimension, so the nearest neighbour of axis(i) is the record built from it.
func axis(i int) []float32 {
	v := make([]float32, dim)
	for d := range v {
		v[d] = 0.05
	}
	v[i%dim] = 1.0
	return v
}

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
}

func run() error {
	url := os.Getenv("DXMT_URL")
	if url == "" {
		url = "http://127.0.0.1:9999"
	}
	db, err := dxmt.New(url)
	if err != nil {
		return err
	}
	ctx := context.Background()
	if !db.Ready(ctx) {
		return fmt.Errorf("no server at %s (start one with: cd vectordb && cargo run -- serve)", url)
	}
	fmt.Println("connected to", url)

	// Start from nothing, even if a previous run left the table behind.
	_ = db.DeleteTable(ctx, table)
	info, err := db.CreateTable(ctx, dxmt.CreateTable{
		Name: table, Dimension: dim, DistanceMetric: dxmt.Cosine,
	})
	if err != nil {
		return err
	}
	fmt.Printf("created table %q (%d dimensions)\n", info.Name, info.Dimension)

	// ── insert ────────────────────────────────────────────────────────────────────────────
	records := make([]dxmt.Record, 6)
	for i := range records {
		topic := "odd"
		if i%2 == 0 {
			topic = "even"
		}
		records[i] = dxmt.Record{
			ID:     fmt.Sprintf("doc-%d", i),
			Vector: axis(i),
			Payload: map[string]any{
				"title": fmt.Sprintf("document %d", i), "topic": topic, "rank": i,
			},
		}
	}
	inserted, err := db.Insert(ctx, table, records)
	if err != nil {
		return err
	}
	fmt.Printf("inserted %d records\n", inserted.Count)

	// ── read one back ─────────────────────────────────────────────────────────────────────
	doc, err := db.Get(ctx, table, "doc-3")
	if err != nil {
		return err
	}
	fmt.Printf("read doc-3: payload %v vector[0:3] %v\n", doc.Payload(), doc.Vector[:3])

	// ── search ────────────────────────────────────────────────────────────────────────────
	hits, err := db.Search(ctx, table, dxmt.NewSearch(axis(3), 3))
	if err != nil {
		return err
	}
	fmt.Printf("nearest to axis(3), %d hits in %d ms:\n", hits.Total, hits.TookMs)
	for _, hit := range hits.Hits {
		fmt.Printf("    %-8s %.4f\n", hit.ID, hit.Score)
	}

	// ── search with a filter ──────────────────────────────────────────────────────────────
	query := dxmt.NewSearch(axis(3), 3)
	query.Filter = dxmt.All(dxmt.Eq("topic", "even"))
	filtered, err := db.Search(ctx, table, query)
	if err != nil {
		return err
	}
	fmt.Println("same query, topic=even only:")
	for _, hit := range filtered.Hits {
		fmt.Printf("    %-8s %.4f\n", hit.ID, hit.Score)
	}

	// ── update ────────────────────────────────────────────────────────────────────────────
	// Payload only: the vector is left alone, so the record keeps its place in the graph.
	if err := db.Update(ctx, table, "doc-3", dxmt.Update{
		Payload: map[string]any{"title": "renamed"}, Merge: true,
	}); err != nil {
		return err
	}
	doc, _ = db.Get(ctx, table, "doc-3")
	fmt.Println("after a merge update, doc-3 payload:", doc.Payload())

	// Vector as well: now it answers to a different query.
	if err := db.Update(ctx, table, "doc-3", dxmt.Update{Vector: axis(5)}); err != nil {
		return err
	}
	moved, _ := db.Search(ctx, table, dxmt.NewSearch(axis(5), 2))
	fmt.Print("after moving doc-3 onto axis(5), nearest to axis(5): ")
	for i, hit := range moved.Hits {
		if i > 0 {
			fmt.Print(", ")
		}
		fmt.Print(hit.ID)
	}
	fmt.Println()

	// ── delete ────────────────────────────────────────────────────────────────────────────
	gone, err := db.Delete(ctx, table, []string{"doc-0", "doc-1", "not-a-real-id"})
	if err != nil {
		return err
	}
	fmt.Printf("asked to delete 3 ids, actually deleted %d\n", gone.Deleted)

	_, err = db.Get(ctx, table, "doc-0")
	var apiErr *dxmt.APIError
	if errors.As(err, &apiErr) && apiErr.IsNotFound() {
		fmt.Println("doc-0 is gone, as expected")
	} else {
		return fmt.Errorf("doc-0 should have been deleted, got %v", err)
	}

	left, _ := db.GetTable(ctx, table)
	fmt.Printf("table now holds %d records\n", left.RecordCount)

	if err := db.DeleteTable(ctx, table); err != nil {
		return err
	}
	fmt.Printf("dropped %q\n", table)
	return nil
}
