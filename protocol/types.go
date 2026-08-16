// Package protocol implements the encrypted SparseE Scatter-Gather-Apply workflow.
package protocol

import (
	"fmt"

	"github.com/yuntao-wei/SparseE/fhe"
	"github.com/yuntao-wei/SparseE/permutation"
)

// Dimensions contains only public matrix and network sizes.
type Dimensions struct {
	Rows        int
	Inner       int
	Width       int
	Edges       int
	NetworkSize int
}

// EncryptedBenesPlan mirrors the public recursive Benes topology while storing
// every control bit exclusively as an RGSW selector ciphertext.
type EncryptedBenesPlan struct {
	LogicalSize int
	NetworkSize int
	First       []*fhe.Selector
	Last        []*fhe.Selector
	Children    []EncryptedBenesPlan
}

// SelectorCount reports the number of encrypted HSwitch controls in the plan.
func (plan EncryptedBenesPlan) SelectorCount() int {
	count := len(plan.First) + len(plan.Last)
	for _, child := range plan.Children {
		count += child.SelectorCount()
	}
	return count
}

// Request contains encrypted server inputs and public CD, RP, and dimensions.
type Request struct {
	Dimensions   Dimensions
	CD           []int
	RP           []int
	EncryptedB   []*fhe.Ciphertext
	EncryptedVA  []*fhe.Ciphertext
	ScatterZeros []*fhe.Ciphertext
	PaddingZeros []*fhe.Ciphertext
	OutputZeros  []*fhe.Ciphertext
	GatherPlan   EncryptedBenesPlan
}

// Response contains encrypted output rows and their public shape.
type Response struct {
	RowCount      int
	Width         int
	EncryptedRows []*fhe.Ciphertext
}

func (request *Request) validate() error {
	if request == nil {
		return fmt.Errorf("nil protocol request")
	}
	dimensions := request.Dimensions
	if dimensions.Rows < 0 ||
		dimensions.Inner < 0 ||
		dimensions.Width < 0 ||
		dimensions.Edges < 0 ||
		dimensions.NetworkSize < 0 {
		return fmt.Errorf("protocol dimensions must be nonnegative")
	}
	expectedNetworkSize, err := permutation.NextPowerOfTwo(dimensions.Edges)
	if err != nil {
		return err
	}
	if dimensions.NetworkSize != expectedNetworkSize {
		return fmt.Errorf(
			"network size is %d; want %d for E=%d",
			dimensions.NetworkSize,
			expectedNetworkSize,
			dimensions.Edges,
		)
	}
	if err := validatePublicMetadata(request.CD, request.RP, dimensions); err != nil {
		return err
	}

	ciphertextSets := []struct {
		name       string
		values     []*fhe.Ciphertext
		wantLength int
	}{
		{"encrypted B rows", request.EncryptedB, dimensions.Inner},
		{"encrypted VA values", request.EncryptedVA, dimensions.Edges},
		{"scatter zero ciphertexts", request.ScatterZeros, dimensions.Edges},
		{"padding zero ciphertexts", request.PaddingZeros, dimensions.NetworkSize - dimensions.Edges},
		{"output zero ciphertexts", request.OutputZeros, dimensions.Rows},
	}
	for _, set := range ciphertextSets {
		if err := validateCiphertextSet(set.name, set.values, set.wantLength); err != nil {
			return err
		}
	}

	if request.GatherPlan.LogicalSize != dimensions.Edges ||
		request.GatherPlan.NetworkSize != dimensions.NetworkSize {
		return fmt.Errorf(
			"Gather plan sizes are (%d,%d); want (%d,%d)",
			request.GatherPlan.LogicalSize,
			request.GatherPlan.NetworkSize,
			dimensions.Edges,
			dimensions.NetworkSize,
		)
	}
	return validateEncryptedPlan(request.GatherPlan, true)
}

func validatePublicMetadata(cd, rp []int, dimensions Dimensions) error {
	if len(cd) != dimensions.Inner {
		return fmt.Errorf("CD length is %d; want inner dimension %d", len(cd), dimensions.Inner)
	}
	total := 0
	maxInt := int(^uint(0) >> 1)
	for column, degree := range cd {
		if degree < 0 {
			return fmt.Errorf("CD[%d] is negative", column)
		}
		if degree > maxInt-total {
			return fmt.Errorf("sum(CD) overflows int")
		}
		total += degree
	}
	if total != dimensions.Edges {
		return fmt.Errorf("sum(CD) is %d; want E=%d", total, dimensions.Edges)
	}

	if len(rp) != dimensions.Rows+1 {
		return fmt.Errorf("RP length is %d; want rows+1=%d", len(rp), dimensions.Rows+1)
	}
	if len(rp) == 0 || rp[0] != 0 {
		return fmt.Errorf("RP must start at zero")
	}
	for i := 1; i < len(rp); i++ {
		if rp[i] < rp[i-1] {
			return fmt.Errorf("RP must be nondecreasing at index %d", i)
		}
		if rp[i] > dimensions.Edges {
			return fmt.Errorf("RP[%d]=%d exceeds E=%d", i, rp[i], dimensions.Edges)
		}
	}
	if rp[len(rp)-1] != dimensions.Edges {
		return fmt.Errorf("RP terminator is %d; want E=%d", rp[len(rp)-1], dimensions.Edges)
	}
	return nil
}

func validateCiphertextSet(name string, values []*fhe.Ciphertext, wantLength int) error {
	if len(values) != wantLength {
		return fmt.Errorf("%s length is %d; want %d", name, len(values), wantLength)
	}
	for i, ciphertext := range values {
		if ciphertext == nil || ciphertext.Degree() < 0 {
			return fmt.Errorf("%s[%d] is not a valid RLWE ciphertext", name, i)
		}
	}
	return nil
}

func validateEncryptedPlan(plan EncryptedBenesPlan, root bool) error {
	n := plan.NetworkSize
	if n == 0 {
		if plan.LogicalSize != 0 || len(plan.First) != 0 || len(plan.Last) != 0 || len(plan.Children) != 0 {
			return fmt.Errorf("empty encrypted Benes plan contains routing data")
		}
		return nil
	}
	if n < 0 || n&(n-1) != 0 {
		return fmt.Errorf("encrypted Benes network size %d is not a positive power of two", n)
	}
	if plan.LogicalSize < 0 || plan.LogicalSize > n {
		return fmt.Errorf("encrypted Benes logical size %d is outside [0,%d]", plan.LogicalSize, n)
	}
	if !root && plan.LogicalSize != n {
		return fmt.Errorf("encrypted Benes child logical size %d differs from network size %d", plan.LogicalSize, n)
	}
	if n == 1 {
		if len(plan.First) != 0 || len(plan.Last) != 0 || len(plan.Children) != 0 {
			return fmt.Errorf("one-wire encrypted Benes plan contains switches")
		}
		return nil
	}
	if n == 2 {
		if len(plan.First) != 1 || len(plan.Last) != 0 || len(plan.Children) != 0 {
			return fmt.Errorf("two-wire encrypted Benes plan has invalid shape")
		}
		return validateSelectors(plan.First, "first")
	}
	if len(plan.First) != n/2 || len(plan.Last) != n/2 || len(plan.Children) != 2 {
		return fmt.Errorf("%d-wire encrypted Benes plan has invalid recursive shape", n)
	}
	if err := validateSelectors(plan.First, "first"); err != nil {
		return err
	}
	if err := validateSelectors(plan.Last, "last"); err != nil {
		return err
	}
	for i, child := range plan.Children {
		if child.NetworkSize != n/2 {
			return fmt.Errorf(
				"encrypted Benes child %d network size is %d; want %d",
				i,
				child.NetworkSize,
				n/2,
			)
		}
		if err := validateEncryptedPlan(child, false); err != nil {
			return fmt.Errorf("encrypted Benes child %d: %w", i, err)
		}
	}
	return nil
}

func validateSelectors(selectors []*fhe.Selector, stage string) error {
	for i, selector := range selectors {
		if selector == nil {
			return fmt.Errorf("encrypted Benes %s selector %d is nil", stage, i)
		}
	}
	return nil
}

func validateResponse(response *Response) error {
	if response == nil {
		return fmt.Errorf("nil protocol response")
	}
	if response.RowCount < 0 || response.Width < 0 {
		return fmt.Errorf("response dimensions must be nonnegative")
	}
	return validateCiphertextSet("encrypted result rows", response.EncryptedRows, response.RowCount)
}
