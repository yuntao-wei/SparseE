// Package csr implements the sparse-matrix representation used by SparseE
// client preprocessing.
package csr

import "fmt"

// Matrix stores a matrix in conventional compressed sparse row order.
// RowPtr has length Rows+1 and Values and ColInd both have length NNZ().
type Matrix struct {
	Rows   int
	Cols   int
	RowPtr []int
	ColInd []int
	Values []int64
}

// NewMatrix validates and copies an existing CSR representation.
func NewMatrix(rows, cols int, rowPtr, colInd []int, values []int64) (Matrix, error) {
	matrix := Matrix{
		Rows:   rows,
		Cols:   cols,
		RowPtr: append([]int(nil), rowPtr...),
		ColInd: append([]int(nil), colInd...),
		Values: append([]int64(nil), values...),
	}
	if err := matrix.Validate(); err != nil {
		return Matrix{}, err
	}
	return matrix, nil
}

// FromDense converts a rectangular dense matrix into conventional CSR order.
func FromDense(dense [][]int64) (Matrix, error) {
	rows := len(dense)
	cols := 0
	if rows != 0 {
		cols = len(dense[0])
	}

	matrix := Matrix{
		Rows:   rows,
		Cols:   cols,
		RowPtr: make([]int, 1, rows+1),
	}
	for rowIndex, row := range dense {
		if len(row) != cols {
			return Matrix{}, fmt.Errorf("dense row %d has width %d; want %d", rowIndex, len(row), cols)
		}
		for col, value := range row {
			if value != 0 {
				matrix.ColInd = append(matrix.ColInd, col)
				matrix.Values = append(matrix.Values, value)
			}
		}
		matrix.RowPtr = append(matrix.RowPtr, len(matrix.Values))
	}
	return matrix, nil
}

// NNZ returns the number of explicitly stored entries.
func (m Matrix) NNZ() int {
	return len(m.Values)
}

// Validate checks the structural CSR invariants needed by preprocessing and
// reference SpMM.
func (m Matrix) Validate() error {
	if m.Rows < 0 || m.Cols < 0 {
		return fmt.Errorf("matrix dimensions must be nonnegative")
	}
	if len(m.RowPtr) != m.Rows+1 {
		return fmt.Errorf("CSR RowPtr length is %d; want Rows+1 = %d", len(m.RowPtr), m.Rows+1)
	}
	if len(m.ColInd) != len(m.Values) {
		return fmt.Errorf("CSR ColInd length %d differs from Values length %d", len(m.ColInd), len(m.Values))
	}
	if len(m.RowPtr) == 0 || m.RowPtr[0] != 0 {
		return fmt.Errorf("CSR RowPtr must start at zero")
	}
	for i := 1; i < len(m.RowPtr); i++ {
		if m.RowPtr[i] < m.RowPtr[i-1] {
			return fmt.Errorf("CSR RowPtr must be nondecreasing at index %d", i)
		}
		if m.RowPtr[i] > len(m.Values) {
			return fmt.Errorf("CSR RowPtr[%d] = %d exceeds nnz %d", i, m.RowPtr[i], len(m.Values))
		}
	}
	if m.RowPtr[len(m.RowPtr)-1] != len(m.Values) {
		return fmt.Errorf("CSR RowPtr terminator %d differs from nnz %d", m.RowPtr[len(m.RowPtr)-1], len(m.Values))
	}
	for i, col := range m.ColInd {
		if col < 0 || col >= m.Cols {
			return fmt.Errorf("CSR ColInd[%d] = %d is outside [0,%d)", i, col, m.Cols)
		}
	}
	return nil
}
