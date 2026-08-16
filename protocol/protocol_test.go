package protocol_test

import (
	"reflect"
	"strings"
	"testing"

	"github.com/yuntao-wei/SparseE/csr"
	"github.com/yuntao-wei/SparseE/fhe"
	"github.com/yuntao-wei/SparseE/protocol"
)

func TestPaperExampleEndToEnd(t *testing.T) {
	a, err := csr.FromDense([][]int64{
		{2, 3, 0},
		{0, 0, 4},
		{0, 5, 6},
	})
	if err != nil {
		t.Fatal(err)
	}
	b := [][]int64{{1, 10}, {2, 20}, {3, 30}}

	fheClient, fheServer, err := fhe.NewRoles()
	if err != nil {
		t.Fatal(err)
	}
	client, err := protocol.NewClient(fheClient)
	if err != nil {
		t.Fatal(err)
	}
	server, err := protocol.NewServer(fheServer)
	if err != nil {
		t.Fatal(err)
	}

	request, err := client.Prepare(a, b)
	if err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	if request.Dimensions.Edges != 5 || request.Dimensions.NetworkSize != 8 {
		t.Fatalf("unexpected protocol dimensions: %+v", request.Dimensions)
	}
	if request.GatherPlan.SelectorCount() != 20 {
		t.Fatalf("selector count=%d, want 20", request.GatherPlan.SelectorCount())
	}

	scattered, err := server.Scatter(request)
	if err != nil {
		t.Fatalf("Scatter: %v", err)
	}
	metadata, err := csr.Preprocess(a)
	if err != nil {
		t.Fatal(err)
	}
	scatterIndex := 0
	for sourceRow, degree := range metadata.CD {
		for copyIndex := 0; copyIndex < degree; copyIndex++ {
			assertDecryptsTo(t, fheClient, scattered[scatterIndex], b[sourceRow])
			scatterIndex++
		}
	}

	networkInput := append(append([]*fhe.Ciphertext(nil), scattered...), request.PaddingZeros...)
	gathered, err := server.Gather(networkInput, request.GatherPlan)
	if err != nil {
		t.Fatalf("Gather: %v", err)
	}
	for index, column := range metadata.CO {
		assertDecryptsTo(t, fheClient, gathered[index], b[column])
	}

	response, err := server.Execute(request)
	if err != nil {
		t.Fatalf("Execute: %v", err)
	}
	actual, err := client.Decrypt(response)
	if err != nil {
		t.Fatalf("Decrypt: %v", err)
	}
	expected, err := csr.ReferenceSpMM(a, b)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(actual, expected) {
		t.Fatalf("SparseE output=%v, want reference=%v", actual, expected)
	}
}

func TestEmptyRowsAndEmptyMatrix(t *testing.T) {
	tests := []struct {
		name string
		a    [][]int64
		b    [][]int64
	}{
		{
			name: "empty rows",
			a: [][]int64{
				{0, 0, 0},
				{7, 0, 0},
				{0, 0, 0},
			},
			b: [][]int64{{2, 3}, {5, 7}, {11, 13}},
		},
		{
			name: "zero edges",
			a: [][]int64{
				{0, 0},
				{0, 0},
			},
			b: [][]int64{{2, -1}, {4, 3}},
		},
	}

	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			a, err := csr.FromDense(test.a)
			if err != nil {
				t.Fatal(err)
			}
			client, server, err := protocol.NewRoles()
			if err != nil {
				t.Fatal(err)
			}
			request, err := client.Prepare(a, test.b)
			if err != nil {
				t.Fatalf("Prepare: %v", err)
			}
			response, err := server.Execute(request)
			if err != nil {
				t.Fatalf("Execute: %v", err)
			}
			actual, err := client.Decrypt(response)
			if err != nil {
				t.Fatalf("Decrypt: %v", err)
			}
			expected, err := csr.ReferenceSpMM(a, test.b)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(actual, expected) {
				t.Fatalf("SparseE output=%v, want reference=%v", actual, expected)
			}
		})
	}
}

func TestServerRequestContainsNoCOOrPlaintextSelectors(t *testing.T) {
	assertNoSensitiveRoutingType(t, reflect.TypeOf(protocol.Request{}), map[reflect.Type]bool{})
}

func TestCenteredPlaintextRangeIsEnforced(t *testing.T) {
	client, server, err := protocol.NewRoles()
	if err != nil {
		t.Fatal(err)
	}

	t.Run("boundary values remain ordinary integers", func(t *testing.T) {
		a, err := csr.FromDense([][]int64{{32767}, {-32769}})
		if err != nil {
			t.Fatal(err)
		}
		request, err := client.Prepare(a, [][]int64{{1}})
		if err != nil {
			t.Fatalf("Prepare boundary request: %v", err)
		}
		response, err := server.Execute(request)
		if err != nil {
			t.Fatalf("Execute boundary request: %v", err)
		}
		actual, err := client.Decrypt(response)
		if err != nil {
			t.Fatalf("Decrypt boundary request: %v", err)
		}
		want := [][]int64{{32767}, {-32769}}
		if !reflect.DeepEqual(actual, want) {
			t.Fatalf("boundary result=%v, want %v", actual, want)
		}
	})

	t.Run("out of range result is rejected before encryption", func(t *testing.T) {
		a, err := csr.FromDense([][]int64{{32769}})
		if err != nil {
			t.Fatal(err)
		}
		_, err = client.Prepare(a, [][]int64{{1}})
		if err == nil || !strings.Contains(err.Error(), "wrap modulo 65537") {
			t.Fatalf("Prepare error=%v, want explicit modulo-wrap rejection", err)
		}
	})
}

func assertNoSensitiveRoutingType(t *testing.T, typ reflect.Type, seen map[reflect.Type]bool) {
	t.Helper()
	for typ.Kind() == reflect.Pointer || typ.Kind() == reflect.Slice || typ.Kind() == reflect.Array {
		if typ.Kind() == reflect.Slice && typ.Elem().Kind() == reflect.Bool {
			t.Fatalf("server request exposes plaintext selector slice in %s", typ)
		}
		typ = typ.Elem()
	}
	if typ.Kind() != reflect.Struct || seen[typ] {
		return
	}
	if typ.PkgPath() != reflect.TypeOf(protocol.Request{}).PkgPath() {
		return
	}
	seen[typ] = true
	for i := 0; i < typ.NumField(); i++ {
		field := typ.Field(i)
		if strings.EqualFold(field.Name, "CO") || strings.Contains(strings.ToLower(field.Name), "desttosrc") {
			t.Fatalf("server request exposes sensitive routing field %s.%s", typ, field.Name)
		}
		assertNoSensitiveRoutingType(t, field.Type, seen)
	}
}

func assertDecryptsTo(t *testing.T, client *fhe.Client, ciphertext *fhe.Ciphertext, want []int64) {
	t.Helper()
	have, err := client.DecryptRow(ciphertext, len(want))
	if err != nil {
		t.Fatalf("DecryptRow: %v", err)
	}
	if !reflect.DeepEqual(have, want) {
		t.Fatalf("decrypted row=%v, want %v", have, want)
	}
}
