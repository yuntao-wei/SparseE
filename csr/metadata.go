package csr

import "fmt"

// Metadata contains the CSR arrays and the SparseE column-degree vector.
// VA and CO remain in CSR row order, RP is a conventional row-pointer array,
// and CD[col] counts the occurrences of col in CO.
type Metadata struct {
	VA []int64
	CO []int
	RP []int
	CD []int
}

// Preprocess derives SparseE's VA, CO, RP, and CD client metadata from CSR.
func Preprocess(matrix Matrix) (Metadata, error) {
	if err := matrix.Validate(); err != nil {
		return Metadata{}, err
	}

	metadata := Metadata{
		VA: append([]int64(nil), matrix.Values...),
		CO: append([]int(nil), matrix.ColInd...),
		RP: append([]int(nil), matrix.RowPtr...),
		CD: make([]int, matrix.Cols),
	}
	for _, col := range metadata.CO {
		metadata.CD[col]++
	}
	if err := ValidateMetadata(metadata, matrix.Rows, matrix.Cols); err != nil {
		return Metadata{}, err
	}
	return metadata, nil
}

// ValidateMetadata checks the relationships required by Scatter, Gather, and Apply.
func ValidateMetadata(metadata Metadata, rows, cols int) error {
	if rows < 0 || cols < 0 {
		return fmt.Errorf("metadata dimensions must be nonnegative")
	}
	if len(metadata.VA) != len(metadata.CO) {
		return fmt.Errorf("VA length %d differs from CO length %d", len(metadata.VA), len(metadata.CO))
	}
	if len(metadata.RP) != rows+1 {
		return fmt.Errorf("RP length is %d; want rows+1 = %d", len(metadata.RP), rows+1)
	}
	if len(metadata.CD) != cols {
		return fmt.Errorf("CD length is %d; want column count %d", len(metadata.CD), cols)
	}
	if len(metadata.RP) == 0 || metadata.RP[0] != 0 {
		return fmt.Errorf("RP must start at zero")
	}
	for i := 1; i < len(metadata.RP); i++ {
		if metadata.RP[i] < metadata.RP[i-1] {
			return fmt.Errorf("RP must be nondecreasing at index %d", i)
		}
		if metadata.RP[i] > len(metadata.VA) {
			return fmt.Errorf("RP[%d] = %d exceeds nnz %d", i, metadata.RP[i], len(metadata.VA))
		}
	}
	if metadata.RP[len(metadata.RP)-1] != len(metadata.VA) {
		return fmt.Errorf("RP terminator %d differs from nnz %d", metadata.RP[len(metadata.RP)-1], len(metadata.VA))
	}

	counts := make([]int, cols)
	for i, col := range metadata.CO {
		if col < 0 || col >= cols {
			return fmt.Errorf("CO[%d] = %d is outside [0,%d)", i, col, cols)
		}
		counts[col]++
	}
	for col, degree := range metadata.CD {
		if degree < 0 {
			return fmt.Errorf("CD[%d] is negative", col)
		}
		if counts[col] != degree {
			return fmt.Errorf("CD[%d] = %d; occurrence count in CO is %d", col, degree, counts[col])
		}
	}
	return nil
}
