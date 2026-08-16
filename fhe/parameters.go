package fhe

import (
	"fmt"
	"math/big"
	"math/bits"

	"github.com/tuneinsight/lattigo/v6/schemes/bgv"
)

const (
	actualLogN                    = 12
	actualPlaintextModulus uint64 = 65537

	// A zero base-two setting selects the RNS decomposition. Two Q limbs and
	// one P limb produce two RGSW gadget components.
	actualRGSWBaseTwoDecompositionBits = 0
)

// The executable profile represents Q with two 32-bit NTT primes and P with
// one 61-bit NTT prime. Their aggregate Q and QP widths are 64 and 125 bits.
var (
	actualQModuli = []uint64{4294483969, 4294475777}
	actualPModuli = []uint64{2305843009213554689}
)

// PaperParameterTarget records the parameter structure reported by SparseE.
type PaperParameterTarget struct {
	RingDimension               int
	RLWECiphertextModulusBits   int
	RGSWCiphertextModulusBits   int
	RGSWSpecialModulusBits      int
	RGSWDecompositionLevelCount int
}

// ActualParameters records the concrete parameters accepted by Lattigo v6.2.0.
type ActualParameters struct {
	LogN                           int
	RingDimension                  int
	QModuli                        []uint64
	PModuli                        []uint64
	QPrimeBits                     []int
	PPrimeBits                     []int
	RLWECiphertextQBits            int
	AuxiliaryPBits                 int
	ExtendedQPBits                 int
	PlaintextModulus               uint64
	DefaultScale                   uint64
	RGSWLevelQ                     int
	RGSWLevelP                     int
	RGSWBaseTwoDecompositionBits   int
	RGSWBaseRNSDecompositionCount  int
	RGSWBaseTwoDecompositionCounts []int
	RGSWTotalGadgetDigits          int
	ErrorDistribution              string
	SecretDistribution             string
	RLWEAggregateWidthMatchesPaper bool
	AggregateBitWidthsMatchPaper   bool
	Native64BitModulusLayout       bool
	ExactPaperParameterMatch       bool
	PaperL2SemanticallyEquivalent  bool
}

// ParameterReport compares the paper target with the executable parameters.
type ParameterReport struct {
	Target        PaperParameterTarget
	Actual        ActualParameters
	Limitations   []string
	SecurityClaim string
}

// ValidateExactPaperMatch checks the executable q/Q/q'/l parameterization.
func (report ParameterReport) ValidateExactPaperMatch() error {
	if report.Actual.ExactPaperParameterMatch && report.Actual.PaperL2SemanticallyEquivalent {
		return nil
	}
	return fmt.Errorf(
		"SparseE parameter mismatch: target q/Q/q'/l=%d/%d/%d/%d; "+
			"actual Q/P/QP=%d/%d/%d bits; aggregate-width-match=%t; "+
			"native-64-bit-layout=%t",
		report.Target.RLWECiphertextModulusBits,
		report.Target.RGSWCiphertextModulusBits,
		report.Target.RGSWSpecialModulusBits,
		report.Target.RGSWDecompositionLevelCount,
		report.Actual.RLWECiphertextQBits,
		report.Actual.AuxiliaryPBits,
		report.Actual.ExtendedQPBits,
		report.Actual.AggregateBitWidthsMatchPaper,
		report.Actual.Native64BitModulusLayout,
	)
}

func newParameters() (bgv.Parameters, error) {
	return bgv.NewParametersFromLiteral(bgv.ParametersLiteral{
		LogN:             actualLogN,
		Q:                append([]uint64(nil), actualQModuli...),
		P:                append([]uint64(nil), actualPModuli...),
		PlaintextModulus: actualPlaintextModulus,
	})
}

func newParameterReport(params bgv.Parameters) ParameterReport {
	levelQ := params.MaxLevelQ()
	levelP := params.MaxLevelP()
	baseTwoCounts := params.BaseTwoDecompositionVectorSize(
		levelQ,
		levelP,
		actualRGSWBaseTwoDecompositionBits,
	)

	baseRNSCount := params.BaseRNSDecompositionVectorSize(levelQ, levelP)
	totalDigits := 0
	for i := 0; i < baseRNSCount && i < len(baseTwoCounts); i++ {
		totalDigits += baseTwoCounts[i]
	}

	q := append([]uint64(nil), params.Q()...)
	p := append([]uint64(nil), params.P()...)
	qp := make([]uint64, 0, len(q)+len(p))
	qp = append(qp, q...)
	qp = append(qp, p...)
	qBits := productBitLen(q)
	pBits := productBitLen(p)
	qpBits := productBitLen(qp)

	return ParameterReport{
		Target: PaperParameterTarget{
			RingDimension:               4096,
			RLWECiphertextModulusBits:   64,
			RGSWCiphertextModulusBits:   128,
			RGSWSpecialModulusBits:      64,
			RGSWDecompositionLevelCount: 2,
		},
		Actual: ActualParameters{
			LogN:                           params.LogN(),
			RingDimension:                  params.N(),
			QModuli:                        q,
			PModuli:                        p,
			QPrimeBits:                     modulusBits(q),
			PPrimeBits:                     modulusBits(p),
			RLWECiphertextQBits:            qBits,
			AuxiliaryPBits:                 pBits,
			ExtendedQPBits:                 qpBits,
			PlaintextModulus:               params.PlaintextModulus(),
			DefaultScale:                   params.DefaultScale().Uint64(),
			RGSWLevelQ:                     levelQ,
			RGSWLevelP:                     levelP,
			RGSWBaseTwoDecompositionBits:   actualRGSWBaseTwoDecompositionBits,
			RGSWBaseRNSDecompositionCount:  baseRNSCount,
			RGSWBaseTwoDecompositionCounts: append([]int(nil), baseTwoCounts...),
			RGSWTotalGadgetDigits:          totalDigits,
			ErrorDistribution:              fmt.Sprintf("%T %v", params.Xe(), params.Xe()),
			SecretDistribution:             fmt.Sprintf("%T %v", params.Xs(), params.Xs()),
			RLWEAggregateWidthMatchesPaper: qBits == 64,
			AggregateBitWidthsMatchPaper:   qBits == 64 && pBits == 64 && qpBits == 128,
			Native64BitModulusLayout:       false,
			ExactPaperParameterMatch:       false,
			PaperL2SemanticallyEquivalent:  false,
		},
		Limitations: []string{
			"Executable Q uses two 32-bit RNS primes; SparseE reports a 64-bit q modulus.",
			"Executable RGSW parameters use a 61-bit P modulus and a 125-bit QP product; " +
				"SparseE reports q'=64 bits and Q=128 bits.",
			"Two RNS gadget components are not assumed to be equivalent to SparseE l=2.",
			"The executable profile uses BGV t=65537 and the default Lattigo RLWE distributions.",
		},
		SecurityClaim: "Executable parameter security is not asserted to match the SparseE parameters.",
	}
}

func modulusBits(moduli []uint64) []int {
	result := make([]int, len(moduli))
	for i, modulus := range moduli {
		result[i] = bits.Len64(modulus)
	}
	return result
}

func productBitLen(moduli []uint64) int {
	product := big.NewInt(1)
	for _, modulus := range moduli {
		product.Mul(product, new(big.Int).SetUint64(modulus))
	}
	if len(moduli) == 0 {
		return 0
	}
	return product.BitLen()
}
