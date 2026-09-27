package dxmt

import (
	"context"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

// A server that records the api-key header of each request and answers with an empty list.
func recording(t *testing.T) (*httptest.Server, *[]string) {
	t.Helper()
	var seen []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seen = append(seen, r.Header.Get("api-key"))
		w.Header().Set("Content-Type", "application/json")
		fmt.Fprint(w, `{"success":true,"data":{"tables":[]},"error":null}`)
	}))
	t.Cleanup(srv.Close)
	return srv, &seen
}

func TestTheKeyIsSentAsTheAPIKeyHeader(t *testing.T) {
	srv, seen := recording(t)
	c, err := New(srv.URL, WithAPIKey("s3cret"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := c.ListTables(context.Background()); err != nil {
		t.Fatal(err)
	}
	if len(*seen) != 1 || (*seen)[0] != "s3cret" {
		t.Fatalf("api-key header was %q", *seen)
	}
}

func TestWithoutAKeyNoHeaderIsSent(t *testing.T) {
	srv, seen := recording(t)
	c, _ := New(srv.URL)
	if _, err := c.ListTables(context.Background()); err != nil {
		t.Fatal(err)
	}
	if (*seen)[0] != "" {
		t.Fatalf("sent api-key %q with no key configured", (*seen)[0])
	}
}

// %v, %+v and %#v print unexported fields by default; the key must not be among them.
func TestPrintingTheClientDoesNotPrintTheKey(t *testing.T) {
	c, _ := New("http://127.0.0.1:9", WithAPIKey("s3cret"))
	for _, verb := range []string{"%v", "%+v", "%#v", "%s"} {
		if out := fmt.Sprintf(verb, c); strings.Contains(out, "s3cret") {
			t.Fatalf("%s printed the key: %s", verb, out)
		}
	}
}
