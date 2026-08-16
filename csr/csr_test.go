package csr

import (
	"reflect"
	"testing"
)

func TestFromDenseAndPreprocessPaperExample(t *testing.T) {
	matrix, err := FromDense([][]int64{
		{2, 3, 0},
		{0, 0, 4},
		{0, 5, 6},
	})
	if err != nil {
		t.Fatalf("FromDense() error = %v", err)
	}

	wantMatrix := Matrix{
		Rows:   3,
		Cols:   3,
		RowPtr: []int{0, 2, 3, 5},
		ColInd: []int{0, 1, 2, 1, 2},
		Values: []int64{2, 3, 4, 5, 6},
	}
	if !reflect.DeepEqual(matrix, wantMatrix) {
		t.Fatalf("FromDense() = %#v, want %#v", matrix, wantMatrix)
	}

	metadata, err := Preprocess(matrix)
	if err != nil {
		t.Fatalf("Preprocess() error = %v", err)
	}
	wantMetadata := Metadata{
		VA: []int64{2, 3, 4, 5, 6},
		CO: []int{0, 1, 2, 1, 2},
		RP: []int{0, 2, 3, 5},
		CD: []int{1, 2, 2},
	}
	if !reflect.DeepEqual(metadata, wantMetadata) {
		t.Fatalf("Preprocess() = %#v, want %#v", metadata, wantMetadata)
	}

	matrix.Values[0] = 99
	if metadata.VA[0] != 2 {
		t.Fatal("Preprocess() metadata aliases the input matrix")
	}
}

func TestFromDensePreservesEmptyRows(t *testing.T) {
	matrix, err := FromDense([][]int64{
		{0, 0, 0},
		{7, 0, 0},
		{0, 0, 0},
	})
	if err != nil {
		t.Fatalf("FromDense() error = %v", err)
	}
	if got, want := matrix.RowPtr, []int{0, 0, 1, 1}; !reflect.DeepEqual(got, want) {
		t.Fatalf("RowPtr = %v, want %v", got, want)
	}
	metadata, err := Preprocess(matrix)
	if err != nil {
		t.Fatalf("Preprocess() error = %v", err)
	}
	if got, want := metadata.CD, []int{1, 0, 0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("CD = %v, want %v", got, want)
	}
}

func TestNewMatrixRepresentsZeroByKShape(t *testing.T) {
	matrix, err := NewMatrix(0, 4, []int{0}, nil, nil)
	if err != nil {
		t.Fatalf("NewMatrix() error = %v", err)
	}
	metadata, err := Preprocess(matrix)
	if err != nil {
		t.Fatalf("Preprocess() error = %v", err)
	}
	if got, want := metadata.CD, []int{0, 0, 0, 0}; !reflect.DeepEqual(got, want) {
		t.Fatalf("CD = %v, want %v", got, want)
	}
}

func TestReferenceSpMM(t *testing.T) {
	matrix, err := FromDense([][]int64{
		{2, 3, 0},
		{0, 0, 4},
		{0, 5, 6},
	})
	if err != nil {
		t.Fatal(err)
	}
	got, err := ReferenceSpMM(matrix, [][]int64{
		{1, 10},
		{2, 20},
		{3, 30},
	})
	if err != nil {
		t.Fatalf("ReferenceSpMM() error = %v", err)
	}
	want := [][]int64{{8, 80}, {12, 120}, {28, 280}}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("ReferenceSpMM() = %v, want %v", got, want)
	}
}

func TestCSRRejectsMalformedInputs(t *testing.T) {
	tests := []struct {
		name   string
		matrix Matrix
	}{
		{"negative dimensions", Matrix{Rows: -1}},
		{"row pointer length", Matrix{Rows: 1, Cols: 1, RowPtr: []int{0}}},
		{"row pointer start", Matrix{Rows: 1, Cols: 1, RowPtr: []int{1, 1}, ColInd: []int{0}, Values: []int64{1}}},
		{"row pointer order", Matrix{Rows: 2, Cols: 1, RowPtr: []int{0, 1, 0}, ColInd: []int{0}, Values: []int64{1}}},
		{"row pointer range", Matrix{Rows: 1, Cols: 1, RowPtr: []int{0, 2}, ColInd: []int{0}, Values: []int64{1}}},
		{"length mismatch", Matrix{Rows: 1, Cols: 1, RowPtr: []int{0, 1}, ColInd: nil, Values: []int64{1}}},
		{"column range", Matrix{Rows: 1, Cols: 1, RowPtr: []int{0, 1}, ColInd: []int{1}, Values: []int64{1}}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if err := test.matrix.Validate(); err == nil {
				t.Fatal("Validate() error = nil")
			}
		})
	}

	if _, err := FromDense([][]int64{{1, 2}, {3}}); err == nil {
		t.Fatal("FromDense(ragged) error = nil")
	}
}

func TestValidateMetadataRejectsInconsistency(t *testing.T) {
	valid := Metadata{
		VA: []int64{2, 3, 4},
		CO: []int{0, 1, 1},
		RP: []int{0, 2, 3},
		CD: []int{1, 2},
	}
	if err := ValidateMetadata(valid, 2, 2); err != nil {
		t.Fatalf("ValidateMetadata(valid) error = %v", err)
	}

	invalid := []Metadata{
		{VA: []int64{2}, CO: nil, RP: []int{0, 1}, CD: []int{0}},
		{VA: []int64{2}, CO: []int{1}, RP: []int{0, 1}, CD: []int{1}},
		{VA: []int64{2}, CO: []int{0}, RP: []int{0, 1}, CD: []int{0}},
		{VA: []int64{2}, CO: []int{0}, RP: []int{0, 1}, CD: []int{-1}},
		{VA: []int64{2}, CO: []int{0}, RP: []int{0, 2}, CD: []int{1}},
	}
	for i, metadata := range invalid {
		if err := ValidateMetadata(metadata, 1, 1); err == nil {
			t.Fatalf("ValidateMetadata(invalid[%d]) error = nil", i)
		}
	}
}

func TestReferenceSpMMRejectsShapeMismatch(t *testing.T) {
	matrix, err := FromDense([][]int64{{1, 0}, {0, 1}})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ReferenceSpMM(matrix, [][]int64{{1}}); err == nil {
		t.Fatal("ReferenceSpMM(row mismatch) error = nil")
	}
	if _, err := ReferenceSpMM(matrix, [][]int64{{1}, {2, 3}}); err == nil {
		t.Fatal("ReferenceSpMM(ragged) error = nil")
	}
}

func TestReferenceSpMMUsesExactIntermediates(t *testing.T) {
	matrix, err := NewMatrix(
		1,
		2,
		[]int{0, 2},
		[]int{0, 1},
		[]int64{1<<62 - 1, 1<<62 - 1},
	)
	if err != nil {
		t.Fatal(err)
	}
	result, err := ReferenceSpMM(matrix, [][]int64{{2}, {-2}})
	if err != nil {
		t.Fatalf("ReferenceSpMM cancellation: %v", err)
	}
	if !reflect.DeepEqual(result, [][]int64{{0}}) {
		t.Fatalf("ReferenceSpMM cancellation=%v, want [[0]]", result)
	}

	overflow, err := NewMatrix(1, 1, []int{0, 1}, []int{0}, []int64{1 << 62})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ReferenceSpMM(overflow, [][]int64{{4}}); err == nil {
		t.Fatal("ReferenceSpMM overflow error = nil")
	}
}
