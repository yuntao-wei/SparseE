// Package benchmarks measures SparseE protocol workloads with Lattigo.
package benchmarks

import (
	"reflect"
	"testing"

	"github.com/yuntao-wei/SparseE/csr"
	"github.com/yuntao-wei/SparseE/protocol"
	"github.com/yuntao-wei/SparseE/workloads"
)

func TestRepresentativeWorkloadsEndToEnd(t *testing.T) {
	for _, workload := range workloads.RepresentativeWorkloads() {
		t.Run(workload.Name, func(t *testing.T) {
			a, err := csr.FromDense(workload.SparseA)
			if err != nil {
				t.Fatal(err)
			}
			client, server, err := protocol.NewRoles()
			if err != nil {
				t.Fatal(err)
			}

			request, err := client.Prepare(a, workload.DenseB)
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
			expected, err := csr.ReferenceSpMM(a, workload.DenseB)
			if err != nil {
				t.Fatal(err)
			}
			if !reflect.DeepEqual(actual, expected) {
				t.Fatalf("SparseE output=%v, want reference=%v", actual, expected)
			}
		})
	}
}

func BenchmarkSparseEProtocol(b *testing.B) {
	for _, workload := range workloads.RepresentativeWorkloads() {
		b.Run(workload.Name, func(b *testing.B) {
			a, err := csr.FromDense(workload.SparseA)
			if err != nil {
				b.Fatal(err)
			}
			client, server, err := protocol.NewRoles()
			if err != nil {
				b.Fatal(err)
			}
			request, err := client.Prepare(a, workload.DenseB)
			if err != nil {
				b.Fatal(err)
			}

			b.ResetTimer()
			for i := 0; i < b.N; i++ {
				if _, err := server.Execute(request); err != nil {
					b.Fatal(err)
				}
			}
			b.StopTimer()
			b.ReportMetric(float64(request.Dimensions.Edges), "nnz/op")
			b.ReportMetric(float64(request.GatherPlan.SelectorCount()), "selectors/op")
			b.ReportMetric(float64(request.GatherPlan.SelectorCount()), "rgsw-ep/op")
		})
	}
}
