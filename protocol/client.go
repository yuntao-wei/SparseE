package protocol

import (
	"fmt"
	"math/big"

	"github.com/yuntao-wei/SparseE/csr"
	"github.com/yuntao-wei/SparseE/fhe"
	"github.com/yuntao-wei/SparseE/permutation"
)

type clientBackend interface {
	EncryptBRow([]int64) (*fhe.Ciphertext, error)
	EncryptVA(int64) (*fhe.Ciphertext, error)
	EncryptSelector(bool) (*fhe.Selector, error)
	DecryptRow(*fhe.Ciphertext, int) ([]int64, error)
	CenteredPlaintextBounds() (int64, int64, error)
}

// Client owns preprocessing, encryption, and decryption state.
type Client struct {
	backend clientBackend
}

// NewClient binds the protocol Client to a Lattigo FHE client role.
func NewClient(backend *fhe.Client) (*Client, error) {
	if backend == nil {
		return nil, fmt.Errorf("nil FHE client")
	}
	return &Client{backend: backend}, nil
}

// NewRoles creates matching protocol roles over one Lattigo key set.
func NewRoles() (*Client, *Server, error) {
	fheClient, fheServer, err := fhe.NewRoles()
	if err != nil {
		return nil, nil, err
	}
	client, err := NewClient(fheClient)
	if err != nil {
		return nil, nil, err
	}
	server, err := NewServer(fheServer)
	if err != nil {
		return nil, nil, err
	}
	return client, server, nil
}

// Prepare derives metadata, compiles and encrypts the Benes plan, and encrypts
// the protocol inputs. Zero ciphertexts are encrypted independently.
func (client *Client) Prepare(a csr.Matrix, b [][]int64) (*Request, error) {
	if client == nil || client.backend == nil {
		return nil, fmt.Errorf("nil protocol client")
	}
	if err := a.Validate(); err != nil {
		return nil, fmt.Errorf("validate sparse A: %w", err)
	}
	width, err := validateDenseB(b, a.Cols)
	if err != nil {
		return nil, err
	}
	lower, upper, err := client.backend.CenteredPlaintextBounds()
	if err != nil {
		return nil, fmt.Errorf("read FHE plaintext range: %w", err)
	}
	if err := validateIntegerSpMMRange(a, b, lower, upper); err != nil {
		return nil, err
	}

	metadata, err := csr.Preprocess(a)
	if err != nil {
		return nil, fmt.Errorf("preprocess CSR A: %w", err)
	}
	destToSrc, err := permutation.BuildGatherMap(metadata.CO, metadata.CD)
	if err != nil {
		return nil, fmt.Errorf("build Gather map: %w", err)
	}
	plainPlan, err := permutation.CompileBenes(destToSrc)
	if err != nil {
		return nil, fmt.Errorf("compile Benes plan: %w", err)
	}
	encryptedPlan, err := client.encryptBenesPlan(plainPlan)
	if err != nil {
		return nil, err
	}

	encryptedB := make([]*fhe.Ciphertext, len(b))
	for row := range b {
		encryptedB[row], err = client.backend.EncryptBRow(b[row])
		if err != nil {
			return nil, fmt.Errorf("encrypt B row %d: %w", row, err)
		}
	}
	encryptedVA := make([]*fhe.Ciphertext, len(metadata.VA))
	for index, value := range metadata.VA {
		encryptedVA[index], err = client.backend.EncryptVA(value)
		if err != nil {
			return nil, fmt.Errorf("encrypt VA[%d]: %w", index, err)
		}
	}

	zeroRow := make([]int64, width)
	scatterZeros, err := client.encryptIndependentZeros(len(metadata.VA), zeroRow, "scatter")
	if err != nil {
		return nil, err
	}
	paddingZeros, err := client.encryptIndependentZeros(plainPlan.NetworkSize-len(metadata.VA), zeroRow, "padding")
	if err != nil {
		return nil, err
	}
	outputZeros, err := client.encryptIndependentZeros(a.Rows, zeroRow, "output")
	if err != nil {
		return nil, err
	}

	request := &Request{
		Dimensions: Dimensions{
			Rows:        a.Rows,
			Inner:       a.Cols,
			Width:       width,
			Edges:       len(metadata.VA),
			NetworkSize: plainPlan.NetworkSize,
		},
		CD:           append([]int(nil), metadata.CD...),
		RP:           append([]int(nil), metadata.RP...),
		EncryptedB:   encryptedB,
		EncryptedVA:  encryptedVA,
		ScatterZeros: scatterZeros,
		PaddingZeros: paddingZeros,
		OutputZeros:  outputZeros,
		GatherPlan:   encryptedPlan,
	}
	if err := request.validate(); err != nil {
		return nil, fmt.Errorf("validate prepared request: %w", err)
	}
	return request, nil
}

// Decrypt decodes an encrypted protocol response.
func (client *Client) Decrypt(response *Response) ([][]int64, error) {
	if client == nil || client.backend == nil {
		return nil, fmt.Errorf("nil protocol client")
	}
	if err := validateResponse(response); err != nil {
		return nil, err
	}
	result := make([][]int64, response.RowCount)
	for row, ciphertext := range response.EncryptedRows {
		values, err := client.backend.DecryptRow(ciphertext, response.Width)
		if err != nil {
			return nil, fmt.Errorf("decrypt result row %d: %w", row, err)
		}
		result[row] = values
	}
	return result, nil
}

func validateDenseB(b [][]int64, expectedRows int) (int, error) {
	if len(b) != expectedRows {
		return 0, fmt.Errorf("dense B row count is %d; want A column count %d", len(b), expectedRows)
	}
	width := 0
	if len(b) != 0 {
		width = len(b[0])
	}
	for row, values := range b {
		if len(values) != width {
			return 0, fmt.Errorf("dense B row %d has width %d; want %d", row, len(values), width)
		}
	}
	return width, nil
}

// validateIntegerSpMMRange checks exact outputs against the centered plaintext range.
func validateIntegerSpMMRange(a csr.Matrix, b [][]int64, lower, upper int64) error {
	lowerBound := big.NewInt(lower)
	upperBound := big.NewInt(upper)
	accumulator := new(big.Int)
	product := new(big.Int)
	left := new(big.Int)
	right := new(big.Int)

	width := 0
	if len(b) != 0 {
		width = len(b[0])
	}
	for row := 0; row < a.Rows; row++ {
		for feature := 0; feature < width; feature++ {
			accumulator.SetInt64(0)
			for index := a.RowPtr[row]; index < a.RowPtr[row+1]; index++ {
				left.SetInt64(a.Values[index])
				right.SetInt64(b[a.ColInd[index]][feature])
				product.Mul(left, right)
				accumulator.Add(accumulator, product)
			}
			if accumulator.Cmp(lowerBound) < 0 || accumulator.Cmp(upperBound) > 0 {
				modulus := new(big.Int).Sub(upperBound, lowerBound)
				modulus.Add(modulus, big.NewInt(1))
				return fmt.Errorf(
					"integer SpMM result C[%d,%d]=%s is outside centered BGV plaintext range [%d,%d]; evaluation would wrap modulo %s",
					row,
					feature,
					accumulator.String(),
					lower,
					upper,
					modulus.String(),
				)
			}
		}
	}
	return nil
}

func (client *Client) encryptIndependentZeros(count int, zeroRow []int64, purpose string) ([]*fhe.Ciphertext, error) {
	result := make([]*fhe.Ciphertext, count)
	for i := range result {
		ciphertext, err := client.backend.EncryptBRow(zeroRow)
		if err != nil {
			return nil, fmt.Errorf("encrypt independent %s zero %d: %w", purpose, i, err)
		}
		result[i] = ciphertext
	}
	return result, nil
}

func (client *Client) encryptBenesPlan(plan permutation.BenesPlan) (EncryptedBenesPlan, error) {
	encrypted := EncryptedBenesPlan{
		LogicalSize: plan.LogicalSize,
		NetworkSize: plan.NetworkSize,
		First:       make([]*fhe.Selector, len(plan.First)),
		Last:        make([]*fhe.Selector, len(plan.Last)),
		Children:    make([]EncryptedBenesPlan, len(plan.Children)),
	}
	for i, bit := range plan.First {
		selector, err := client.backend.EncryptSelector(bit)
		if err != nil {
			return EncryptedBenesPlan{}, fmt.Errorf(
				"encrypt first-stage selector %d for %d-wire Benes plan: %w",
				i,
				plan.NetworkSize,
				err,
			)
		}
		encrypted.First[i] = selector
	}
	for i, bit := range plan.Last {
		selector, err := client.backend.EncryptSelector(bit)
		if err != nil {
			return EncryptedBenesPlan{}, fmt.Errorf(
				"encrypt last-stage selector %d for %d-wire Benes plan: %w",
				i,
				plan.NetworkSize,
				err,
			)
		}
		encrypted.Last[i] = selector
	}
	for i, child := range plan.Children {
		encryptedChild, err := client.encryptBenesPlan(child)
		if err != nil {
			return EncryptedBenesPlan{}, fmt.Errorf("encrypt Benes child %d: %w", i, err)
		}
		encrypted.Children[i] = encryptedChild
	}
	return encrypted, nil
}
