// How much does the Go client cost per query?
//
//	cd vectordb && cargo run --release -- serve &
//	cd vectordb/dxmt-go-sdk && go run ./examples/bench -table bench -dim 768 -k 100 -threads 16
//
// This is not a benchmark of the database. It is a benchmark of the *client*, and the number
// that matters is the last line: CPU per query. The Rust `loadgen` in the server crate exists
// because a client that serialises 768 floats per request can cost more than the server does,
// and a server-side improvement measured through an expensive client arrives damped. The same
// question applies to every SDK this repository ships, and asking it is cheap.
package main

import (
	"context"
	"flag"
	"fmt"
	"math/rand"
	"os"
	"sync"
	"sync/atomic"
	"syscall"
	"time"

	dxmt "github.com/dxmt/dxmt/vectordb/dxmt-go-sdk"
)

func cpuSeconds() float64 {
	var usage syscall.Rusage
	if syscall.Getrusage(syscall.RUSAGE_SELF, &usage) != nil {
		return 0
	}
	sec := func(t syscall.Timeval) float64 { return float64(t.Sec) + float64(t.Usec)/1e6 }
	return sec(usage.Utime) + sec(usage.Stime)
}

func main() {
	url := flag.String("url", "http://127.0.0.1:9999", "server")
	table := flag.String("table", "bench", "table")
	dim := flag.Int("dim", 768, "dimensions")
	k := flag.Int("k", 100, "hits per query")
	threads := flag.Int("threads", 16, "concurrent callers")
	duration := flag.Int("duration", 15, "measured seconds")
	warmup := flag.Int("warmup", 5, "warmup seconds")
	queries := flag.Int("queries", 1000, "distinct query vectors")
	flag.Parse()

	db, err := dxmt.New(*url)
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(2)
	}
	ctx := context.Background()
	if !db.Ready(ctx) {
		fmt.Fprintf(os.Stderr, "no server at %s\n", *url)
		os.Exit(2)
	}

	// Built once, like the Rust generators: the query set is fixed, and re-deriving it every
	// call would measure the random number generator.
	rng := rand.New(rand.NewSource(20260916))
	vectors := make([][]float32, *queries)
	for i := range vectors {
		v := make([]float32, *dim)
		var norm float64
		for d := range v {
			v[d] = float32(rng.NormFloat64())
			norm += float64(v[d]) * float64(v[d])
		}
		for d := range v {
			v[d] /= float32(norm)
		}
		vectors[i] = v
	}

	var served, failed int64
	var latencies sync.Map // worker -> []time.Duration, merged at the end
	stop := make(chan struct{})
	var wg sync.WaitGroup

	for worker := 0; worker < *threads; worker++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			mine := make([]time.Duration, 0, 1<<16)
			for i := worker; ; i++ {
				select {
				case <-stop:
					latencies.Store(worker, mine)
					return
				default:
				}
				query := dxmt.NewSearch(vectors[i%len(vectors)], *k)
				query.WithPayload = false
				start := time.Now()
				if _, err := db.Search(ctx, *table, query); err != nil {
					atomic.AddInt64(&failed, 1)
					continue
				}
				mine = append(mine, time.Since(start))
				atomic.AddInt64(&served, 1)
			}
		}(worker)
	}

	time.Sleep(time.Duration(*warmup) * time.Second)
	atomic.StoreInt64(&served, 0)
	atomic.StoreInt64(&failed, 0)
	cpuBefore := cpuSeconds()
	measured := time.Now()

	time.Sleep(time.Duration(*duration) * time.Second)
	elapsed := time.Since(measured).Seconds()
	cpu := cpuSeconds() - cpuBefore
	close(stop)
	wg.Wait()

	// Latencies from before the warmup cutoff are in there too; the QPS and CPU numbers are the
	// ones this tool is for, and they are measured over the window.
	var all []time.Duration
	latencies.Range(func(_, v any) bool {
		all = append(all, v.([]time.Duration)...)
		return true
	})
	ok := atomic.LoadInt64(&served)

	fmt.Println("client        Go SDK (JSON over HTTP)")
	fmt.Printf("threads       %d\n", *threads)
	fmt.Printf("served        %d in %.1fs   failed %d\n", ok, elapsed, atomic.LoadInt64(&failed))
	fmt.Printf("QPS           %.0f\n", float64(ok)/elapsed)
	if ok > 0 {
		fmt.Printf("client CPU    %.2f cores (%.0f us/query)\n",
			cpu/elapsed, cpu*1e6/float64(ok))
	}
}
