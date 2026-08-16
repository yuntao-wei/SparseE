package workloads

import (
	"slices"
	"testing"

	"github.com/yuntao-wei/SparseE/csr"
)

func TestRepresentativeWorkloadMetadata(t *testing.T) {
	for _, workload := range RepresentativeWorkloads() {
		t.Run(workload.Name, func(t *testing.T) {
			matrix, err := csr.FromDense(workload.SparseA)
			if err != nil {
				t.Fatal(err)
			}
			metadata, err := csr.Preprocess(matrix)
			if err != nil {
				t.Fatal(err)
			}
			if !slices.Equal(metadata.CD, workload.ExpectedCD) {
				t.Fatalf("CD = %v; want %v", metadata.CD, workload.ExpectedCD)
			}
			if !slices.Equal(metadata.RP, workload.ExpectedRP) {
				t.Fatalf("RP = %v; want %v", metadata.RP, workload.ExpectedRP)
			}
		})
	}
}
