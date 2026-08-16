// Package fhe implements SparseE cryptographic primitives with Lattigo v6.2.0.
package fhe

import (
	"errors"
	"fmt"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
)

// Ciphertext wraps a BGV/RLWE ciphertext.
type Ciphertext struct {
	value *rlwe.Ciphertext
}

// Degree reports the RLWE ciphertext degree.
func (ct *Ciphertext) Degree() int {
	if ct == nil || ct.value == nil {
		return -1
	}
	return ct.value.Degree()
}

// Level reports the active Q level.
func (ct *Ciphertext) Level() int {
	if ct == nil || ct.value == nil {
		return -1
	}
	return ct.value.Level()
}

// MarshalBinary serializes a ciphertext.
func (ct *Ciphertext) MarshalBinary() ([]byte, error) {
	if err := validateCiphertext(ct); err != nil {
		return nil, err
	}
	return ct.value.MarshalBinary()
}

// Selector is a standard randomized RGSW encryption of a selector bit.
type Selector struct {
	value *rgsw.Ciphertext
}

type externalProductEvaluator interface {
	ExternalProduct(*rlwe.Ciphertext, *rgsw.Ciphertext, *rlwe.Ciphertext)
}

// MarshalBinary serializes a selector ciphertext.
func (selector *Selector) MarshalBinary() ([]byte, error) {
	if err := validateSelector(selector); err != nil {
		return nil, err
	}
	return selector.value.MarshalBinary()
}

// Client is the only role that owns the secret key and decryption capability.
type Client struct {
	params        bgv.Parameters
	secretKey     *rlwe.SecretKey
	encoder       *bgv.Encoder
	encryptor     *rlwe.Encryptor
	rgswEncryptor *rgsw.Encryptor
	decryptor     *rlwe.Decryptor
	report        ParameterReport
}

// Server owns evaluation state and a relinearization key.
type Server struct {
	params        bgv.Parameters
	bgvEvaluator  *bgv.Evaluator
	rgswEvaluator externalProductEvaluator
	report        ParameterReport
}

// NewRoles creates client and server roles over one BGV parameter set.
func NewRoles() (*Client, *Server, error) {
	params, err := newParameters()
	if err != nil {
		return nil, nil, fmt.Errorf("create Lattigo BGV parameters: %w", err)
	}

	keyGenerator := rlwe.NewKeyGenerator(params)
	secretKey, publicKey := keyGenerator.GenKeyPairNew()
	relinearizationKey := keyGenerator.GenRelinearizationKeyNew(secretKey)
	evaluationKeys := rlwe.NewMemEvaluationKeySet(relinearizationKey)
	report := newParameterReport(params)

	client := &Client{
		params:        params,
		secretKey:     secretKey,
		encoder:       bgv.NewEncoder(params),
		encryptor:     rlwe.NewEncryptor(params, publicKey),
		rgswEncryptor: rgsw.NewEncryptor(params, secretKey),
		decryptor:     rlwe.NewDecryptor(params, secretKey),
		report:        report,
	}

	server := &Server{
		params:        params,
		bgvEvaluator:  bgv.NewEvaluator(params, evaluationKeys),
		rgswEvaluator: rgsw.NewEvaluator(params, evaluationKeys),
		report:        report,
	}

	return client, server, nil
}

// ParameterReport returns the paper target and concrete executable parameters.
func (client *Client) ParameterReport() ParameterReport {
	if client == nil {
		return ParameterReport{}
	}
	return cloneReport(client.report)
}

// CenteredPlaintextBounds returns the integer interval that BGV decoding maps
// to without modular wraparound.
func (client *Client) CenteredPlaintextBounds() (lower, upper int64, err error) {
	if client == nil {
		return 0, 0, errors.New("nil client")
	}
	modulus := client.params.PlaintextModulus()
	if modulus > uint64(^uint64(0)>>1) {
		return 0, 0, fmt.Errorf("plaintext modulus %d exceeds int64", modulus)
	}
	threshold := int64(modulus / 2)
	return -int64(modulus) + threshold, threshold - 1, nil
}

// ParameterReport returns the paper target and concrete executable parameters.
func (server *Server) ParameterReport() ParameterReport {
	if server == nil {
		return ParameterReport{}
	}
	return cloneReport(server.report)
}

// EncryptBRow SIMD-packs one dense B row and applies standard randomized,
// noisy public-key RLWE encryption.
func (client *Client) EncryptBRow(row []int64) (*Ciphertext, error) {
	if client == nil {
		return nil, errors.New("nil client")
	}
	if len(row) > client.params.MaxSlots() {
		return nil, fmt.Errorf("B row width %d exceeds %d BGV slots", len(row), client.params.MaxSlots())
	}
	return client.encryptSlots(row)
}

// EncryptVA broadcasts one sparse value to every SIMD slot and encrypts it as
// a randomized BGV/RLWE ciphertext.
func (client *Client) EncryptVA(value int64) (*Ciphertext, error) {
	if client == nil {
		return nil, errors.New("nil client")
	}
	values := make([]int64, client.params.MaxSlots())
	for i := range values {
		values[i] = value
	}
	return client.encryptSlots(values)
}

// EncryptSelector encrypts a selector bit as a randomized RGSW ciphertext.
func (client *Client) EncryptSelector(bit bool) (*Selector, error) {
	if client == nil {
		return nil, errors.New("nil client")
	}

	levelQ := client.params.MaxLevelQ()
	levelP := client.params.MaxLevelP()
	plaintext := rlwe.NewPlaintext(client.params, levelQ)
	plaintext.IsNTT = false
	if bit {
		for i := 0; i <= levelQ; i++ {
			plaintext.Value.Coeffs[i][0] = 1
		}
	}
	client.params.RingQ().AtLevel(levelQ).NTT(plaintext.Value, plaintext.Value)
	plaintext.IsNTT = true

	ciphertext := rgsw.NewCiphertext(
		client.params.Parameters,
		levelQ,
		levelP,
		actualRGSWBaseTwoDecompositionBits,
	)
	if err := client.rgswEncryptor.Encrypt(plaintext, ciphertext); err != nil {
		return nil, fmt.Errorf("encrypt RGSW selector: %w", err)
	}
	return &Selector{value: ciphertext}, nil
}

// DecryptRow decrypts and decodes the requested number of SIMD slots.
func (client *Client) DecryptRow(ciphertext *Ciphertext, width int) ([]int64, error) {
	if client == nil {
		return nil, errors.New("nil client")
	}
	if err := validateCiphertext(ciphertext); err != nil {
		return nil, err
	}
	if width < 0 || width > client.params.MaxSlots() {
		return nil, fmt.Errorf("decode width %d is outside [0, %d]", width, client.params.MaxSlots())
	}

	plaintext := client.decryptor.DecryptNew(ciphertext.value)
	values := make([]int64, width)
	if err := client.encoder.Decode(plaintext, values); err != nil {
		return nil, fmt.Errorf("decode BGV plaintext: %w", err)
	}
	return values, nil
}

// Add homomorphically adds two RLWE ciphertexts on the Server.
func (server *Server) Add(left, right *Ciphertext) (*Ciphertext, error) {
	if err := server.validateBinaryOperands(left, right); err != nil {
		return nil, err
	}
	result, err := server.bgvEvaluator.AddNew(left.value, right.value)
	if err != nil {
		return nil, fmt.Errorf("BGV add: %w", err)
	}
	return &Ciphertext{value: result}, nil
}

// Sub homomorphically subtracts two RLWE ciphertexts on the Server.
func (server *Server) Sub(left, right *Ciphertext) (*Ciphertext, error) {
	if err := server.validateBinaryOperands(left, right); err != nil {
		return nil, err
	}
	result, err := server.bgvEvaluator.SubNew(left.value, right.value)
	if err != nil {
		return nil, fmt.Errorf("BGV subtract: %w", err)
	}
	return &Ciphertext{value: result}, nil
}

// ExternalProduct computes the RLWE x RGSW product used by Gather.
func (server *Server) ExternalProduct(ciphertext *Ciphertext, selector *Selector) (*Ciphertext, error) {
	if server == nil {
		return nil, errors.New("nil server")
	}
	if err := validateCiphertext(ciphertext); err != nil {
		return nil, err
	}
	if err := validateSelector(selector); err != nil {
		return nil, err
	}
	if ciphertext.value.Level() != selector.value.LevelQ() {
		return nil, fmt.Errorf(
			"external-product Q-level mismatch: RLWE=%d RGSW=%d",
			ciphertext.value.Level(),
			selector.value.LevelQ(),
		)
	}

	result := bgv.NewCiphertext(server.params, 1, selector.value.LevelQ())
	result.MetaData = ciphertext.value.MetaData.CopyNew()
	server.rgswEvaluator.ExternalProduct(ciphertext.value, selector.value, result)
	return &Ciphertext{value: result}, nil
}

// HSwitch implements the SparseE Gather switch:
//
//	correction = (a-b) x selector
//	outA       = a - correction
//	outB       = b + correction
//
// The correction is computed by exactly one Lattigo RGSW external product and
// shared by both outputs.
func (server *Server) HSwitch(a, b *Ciphertext, selector *Selector) (*Ciphertext, *Ciphertext, error) {
	delta, err := server.Sub(a, b)
	if err != nil {
		return nil, nil, err
	}
	correction, err := server.ExternalProduct(delta, selector)
	if err != nil {
		return nil, nil, err
	}
	outA, err := server.Sub(a, correction)
	if err != nil {
		return nil, nil, err
	}
	outB, err := server.Add(b, correction)
	if err != nil {
		return nil, nil, err
	}
	return outA, outB, nil
}

// Apply multiplies two RLWE ciphertexts and relinearizes the result.
func (server *Server) Apply(row, encryptedVA *Ciphertext) (*Ciphertext, error) {
	if err := server.validateBinaryOperands(row, encryptedVA); err != nil {
		return nil, err
	}
	result, err := server.bgvEvaluator.MulRelinNew(row.value, encryptedVA.value)
	if err != nil {
		return nil, fmt.Errorf("BGV MulRelin for Apply: %w", err)
	}
	return &Ciphertext{value: result}, nil
}

func (client *Client) encryptSlots(values []int64) (*Ciphertext, error) {
	plaintext := bgv.NewPlaintext(client.params, client.params.MaxLevel())
	if err := client.encoder.Encode(values, plaintext); err != nil {
		return nil, fmt.Errorf("encode BGV slots: %w", err)
	}

	ciphertext := bgv.NewCiphertext(client.params, 1, client.params.MaxLevel())
	if err := client.encryptor.Encrypt(plaintext, ciphertext); err != nil {
		return nil, fmt.Errorf("encrypt BGV plaintext: %w", err)
	}
	return &Ciphertext{value: ciphertext}, nil
}

func (server *Server) validateBinaryOperands(left, right *Ciphertext) error {
	if server == nil {
		return errors.New("nil server")
	}
	if err := validateCiphertext(left); err != nil {
		return fmt.Errorf("left operand: %w", err)
	}
	if err := validateCiphertext(right); err != nil {
		return fmt.Errorf("right operand: %w", err)
	}
	return nil
}

func validateCiphertext(ciphertext *Ciphertext) error {
	if ciphertext == nil || ciphertext.value == nil {
		return errors.New("nil RLWE ciphertext")
	}
	return nil
}

func validateSelector(selector *Selector) error {
	if selector == nil || selector.value == nil {
		return errors.New("nil RGSW selector")
	}
	return nil
}

func cloneReport(report ParameterReport) ParameterReport {
	cloned := report
	cloned.Actual.QModuli = append([]uint64(nil), report.Actual.QModuli...)
	cloned.Actual.PModuli = append([]uint64(nil), report.Actual.PModuli...)
	cloned.Actual.QPrimeBits = append([]int(nil), report.Actual.QPrimeBits...)
	cloned.Actual.PPrimeBits = append([]int(nil), report.Actual.PPrimeBits...)
	cloned.Actual.RGSWBaseTwoDecompositionCounts = append(
		[]int(nil),
		report.Actual.RGSWBaseTwoDecompositionCounts...,
	)
	cloned.Limitations = append([]string(nil), report.Limitations...)
	return cloned
}
