package csr

import (
	"fmt"
	"math/big"
)

// ReferenceSpMM computes plaintext A*B from CSR indices.
func ReferenceSpMM(a Matrix, b [][]int64) ([][]int64, error) {
	if err := a.Validate(); err != nil {
		return nil, err
	}
	if len(b) != a.Cols {
		return nil, fmt.Errorf("dense B row count is %d; want A column count %d", len(b), a.Cols)
	}
	width := 0
	if len(b) != 0 {
		width = len(b[0])
	}
	for row, values := range b {
		if len(values) != width {
			return nil, fmt.Errorf("dense B row %d has width %d; want %d", row, len(values), width)
		}
	}

	result := make([][]int64, a.Rows)
	for row := 0; row < a.Rows; row++ {
		result[row] = make([]int64, width)
		for feature := 0; feature < width; feature++ {
			accumulator := new(big.Int)
			for index := a.RowPtr[row]; index < a.RowPtr[row+1]; index++ {
				left := big.NewInt(a.Values[index])
				right := big.NewInt(b[a.ColInd[index]][feature])
				accumulator.Add(accumulator, new(big.Int).Mul(left, right))
			}
			if !accumulator.IsInt64() {
				return nil, fmt.Errorf("reference result C[%d,%d]=%s exceeds int64", row, feature, accumulator.String())
			}
			result[row][feature] = accumulator.Int64()
		}
	}
	return result, nil
}
