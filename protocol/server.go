package protocol

import (
	"fmt"

	"github.com/yuntao-wei/SparseE/fhe"
)

type serverBackend interface {
	Add(*fhe.Ciphertext, *fhe.Ciphertext) (*fhe.Ciphertext, error)
	HSwitch(*fhe.Ciphertext, *fhe.Ciphertext, *fhe.Selector) (*fhe.Ciphertext, *fhe.Ciphertext, error)
	Apply(*fhe.Ciphertext, *fhe.Ciphertext) (*fhe.Ciphertext, error)
}

// Server owns the FHE evaluation state.
type Server struct {
	backend serverBackend
}

// NewServer binds the protocol Server to a Lattigo FHE evaluator role.
func NewServer(backend *fhe.Server) (*Server, error) {
	if backend == nil {
		return nil, fmt.Errorf("nil FHE server")
	}
	return &Server{backend: backend}, nil
}

// Execute runs the encrypted Scatter-Gather-Apply pipeline. The only plaintext
// routing information available here is CD, RP, and public dimensions.
func (server *Server) Execute(request *Request) (*Response, error) {
	if server == nil || server.backend == nil {
		return nil, fmt.Errorf("nil protocol server")
	}
	if err := request.validate(); err != nil {
		return nil, err
	}

	scattered, err := server.scatter(request)
	if err != nil {
		return nil, fmt.Errorf("scatter: %w", err)
	}
	networkInput := make([]*fhe.Ciphertext, 0, request.Dimensions.NetworkSize)
	networkInput = append(networkInput, scattered...)
	networkInput = append(networkInput, request.PaddingZeros...)
	gathered, err := server.gather(networkInput, request.GatherPlan)
	if err != nil {
		return nil, fmt.Errorf("gather: %w", err)
	}
	outputs, err := server.apply(
		gathered[:request.Dimensions.Edges],
		request.EncryptedVA,
		request.RP,
		request.OutputZeros,
	)
	if err != nil {
		return nil, fmt.Errorf("apply: %w", err)
	}

	return &Response{
		RowCount:      request.Dimensions.Rows,
		Width:         request.Dimensions.Width,
		EncryptedRows: outputs,
	}, nil
}

// Scatter produces CD[column] independently rerandomized copies of each
// encrypted B row by evaluating Add(Enc(B[column]), fresh Enc(0)).
func (server *Server) Scatter(request *Request) ([]*fhe.Ciphertext, error) {
	if server == nil || server.backend == nil {
		return nil, fmt.Errorf("nil protocol server")
	}
	if err := request.validate(); err != nil {
		return nil, err
	}
	return server.scatter(request)
}

func (server *Server) scatter(request *Request) ([]*fhe.Ciphertext, error) {
	result := make([]*fhe.Ciphertext, request.Dimensions.Edges)
	outputIndex := 0
	for column, degree := range request.CD {
		for copyIndex := 0; copyIndex < degree; copyIndex++ {
			ciphertext, err := server.backend.Add(request.EncryptedB[column], request.ScatterZeros[outputIndex])
			if err != nil {
				return nil, fmt.Errorf("replicate B row %d copy %d: %w", column, copyIndex, err)
			}
			result[outputIndex] = ciphertext
			outputIndex++
		}
	}
	return result, nil
}

// Gather evaluates encrypted Benes controls with HSwitch.
func (server *Server) Gather(input []*fhe.Ciphertext, plan EncryptedBenesPlan) ([]*fhe.Ciphertext, error) {
	if server == nil || server.backend == nil {
		return nil, fmt.Errorf("nil protocol server")
	}
	if err := validateEncryptedPlan(plan, true); err != nil {
		return nil, err
	}
	if err := validateCiphertextSet("Gather input", input, plan.NetworkSize); err != nil {
		return nil, err
	}
	return server.gather(input, plan)
}

func (server *Server) gather(input []*fhe.Ciphertext, plan EncryptedBenesPlan) ([]*fhe.Ciphertext, error) {
	n := plan.NetworkSize
	if n <= 1 {
		return append([]*fhe.Ciphertext(nil), input...), nil
	}
	if n == 2 {
		left, right, err := server.backend.HSwitch(input[0], input[1], plan.First[0])
		if err != nil {
			return nil, fmt.Errorf("2-wire HSwitch: %w", err)
		}
		return []*fhe.Ciphertext{left, right}, nil
	}

	upper := make([]*fhe.Ciphertext, n/2)
	lower := make([]*fhe.Ciphertext, n/2)
	for pair := 0; pair < n/2; pair++ {
		left, right, err := server.backend.HSwitch(input[2*pair], input[2*pair+1], plan.First[pair])
		if err != nil {
			return nil, fmt.Errorf("first-stage HSwitch %d in %d-wire network: %w", pair, n, err)
		}
		upper[pair], lower[pair] = left, right
	}
	upperOutput, err := server.gather(upper, plan.Children[0])
	if err != nil {
		return nil, fmt.Errorf("upper %d-wire child: %w", n/2, err)
	}
	lowerOutput, err := server.gather(lower, plan.Children[1])
	if err != nil {
		return nil, fmt.Errorf("lower %d-wire child: %w", n/2, err)
	}

	output := make([]*fhe.Ciphertext, n)
	for pair := 0; pair < n/2; pair++ {
		left, right, err := server.backend.HSwitch(upperOutput[pair], lowerOutput[pair], plan.Last[pair])
		if err != nil {
			return nil, fmt.Errorf("last-stage HSwitch %d in %d-wire network: %w", pair, n, err)
		}
		output[2*pair], output[2*pair+1] = left, right
	}
	return output, nil
}

// Apply multiplies each gathered RLWE row by its encrypted VA ciphertext using
// the FHE Server Apply/MulRelin primitive, then groups additions by plaintext RP.
func (server *Server) Apply(
	gathered []*fhe.Ciphertext,
	encryptedVA []*fhe.Ciphertext,
	rp []int,
	outputZeros []*fhe.Ciphertext,
) ([]*fhe.Ciphertext, error) {
	if server == nil || server.backend == nil {
		return nil, fmt.Errorf("nil protocol server")
	}
	if len(encryptedVA) != len(gathered) {
		return nil, fmt.Errorf("encrypted VA length is %d; want gathered length %d", len(encryptedVA), len(gathered))
	}
	if err := validateCiphertextSet("Apply gathered rows", gathered, len(gathered)); err != nil {
		return nil, err
	}
	if err := validateCiphertextSet("Apply encrypted VA", encryptedVA, len(gathered)); err != nil {
		return nil, err
	}
	if err := validateApplyRP(rp, len(outputZeros), len(gathered)); err != nil {
		return nil, err
	}
	if err := validateCiphertextSet("Apply output zeros", outputZeros, len(outputZeros)); err != nil {
		return nil, err
	}
	return server.apply(gathered, encryptedVA, rp, outputZeros)
}

func (server *Server) apply(
	gathered []*fhe.Ciphertext,
	encryptedVA []*fhe.Ciphertext,
	rp []int,
	outputZeros []*fhe.Ciphertext,
) ([]*fhe.Ciphertext, error) {
	outputs := append([]*fhe.Ciphertext(nil), outputZeros...)
	for row := 0; row < len(outputs); row++ {
		for index := rp[row]; index < rp[row+1]; index++ {
			product, err := server.backend.Apply(gathered[index], encryptedVA[index])
			if err != nil {
				return nil, fmt.Errorf("MulRelin for row %d VA index %d: %w", row, index, err)
			}
			sum, err := server.backend.Add(outputs[row], product)
			if err != nil {
				return nil, fmt.Errorf("accumulate row %d VA index %d: %w", row, index, err)
			}
			outputs[row] = sum
		}
	}
	return outputs, nil
}

func validateApplyRP(rp []int, rows, edges int) error {
	if len(rp) != rows+1 {
		return fmt.Errorf("RP length is %d; want rows+1=%d", len(rp), rows+1)
	}
	if len(rp) == 0 || rp[0] != 0 {
		return fmt.Errorf("RP must start at zero")
	}
	for i := 1; i < len(rp); i++ {
		if rp[i] < rp[i-1] {
			return fmt.Errorf("RP must be nondecreasing at index %d", i)
		}
		if rp[i] > edges {
			return fmt.Errorf("RP[%d]=%d exceeds E=%d", i, rp[i], edges)
		}
	}
	if rp[len(rp)-1] != edges {
		return fmt.Errorf("RP terminator is %d; want E=%d", rp[len(rp)-1], edges)
	}
	return nil
}
