package fhe_test

import (
	"bytes"
	"reflect"
	"strings"
	"testing"

	"github.com/yuntao-wei/SparseE/fhe"
)

func TestLattigoSparseEBackend(t *testing.T) {
	client, server, err := fhe.NewRoles()
	if err != nil {
		t.Fatalf("NewRoles: %v", err)
	}

	t.Run("parameter report separates paper target from runtime", func(t *testing.T) {
		report := client.ParameterReport()
		if report.Target.RingDimension != 4096 || report.Actual.LogN != 12 {
			t.Fatalf("unexpected ring parameters: target=%+v actual=%+v", report.Target, report.Actual)
		}
		if report.Actual.ExactPaperParameterMatch {
			t.Fatal("unexpected exact paper parameter match")
		}
		if report.Actual.PaperL2SemanticallyEquivalent {
			t.Fatal("unexpected equivalence with paper l=2")
		}
		if report.Actual.RGSWTotalGadgetDigits != 2 {
			t.Fatalf("expected two actual gadget digits, got %d", report.Actual.RGSWTotalGadgetDigits)
		}
		if report.Actual.RGSWBaseTwoDecompositionBits != 0 || report.Actual.RGSWBaseRNSDecompositionCount != 2 {
			t.Fatalf("unexpected RGSW decomposition: %+v", report.Actual)
		}
		lower, upper, err := client.CenteredPlaintextBounds()
		if err != nil {
			t.Fatalf("CenteredPlaintextBounds: %v", err)
		}
		if lower != -32769 || upper != 32767 {
			t.Fatalf("centered plaintext bounds=[%d,%d], want [-32769,32767]", lower, upper)
		}
		if !report.Actual.RLWEAggregateWidthMatchesPaper {
			t.Fatalf("aggregate RLWE Q width does not match the paper: %+v", report.Actual)
		}
		if report.Actual.AggregateBitWidthsMatchPaper {
			t.Fatal("unexpected full aggregate-width match")
		}
		if report.Actual.RLWECiphertextQBits != 64 ||
			report.Actual.AuxiliaryPBits != 61 ||
			report.Actual.ExtendedQPBits != 125 {
			t.Fatalf("unexpected aggregate modulus widths: %+v", report.Actual)
		}
		if len(report.Actual.QModuli) != 2 || len(report.Actual.PModuli) != 1 || report.Actual.Native64BitModulusLayout {
			t.Fatalf("unexpected RNS modulus layout: %+v", report.Actual)
		}
		if len(report.Limitations) == 0 || report.SecurityClaim == "" {
			t.Fatal("parameter report is missing mismatch metadata")
		}
		if err := report.ValidateExactPaperMatch(); err == nil {
			t.Fatal("exact paper parameter validation unexpectedly succeeded")
		}
	})

	t.Run("RLWE and RGSW encryption are nondeterministic", func(t *testing.T) {
		first, err := client.EncryptBRow([]int64{1, -2, 3})
		if err != nil {
			t.Fatalf("first EncryptBRow: %v", err)
		}
		second, err := client.EncryptBRow([]int64{1, -2, 3})
		if err != nil {
			t.Fatalf("second EncryptBRow: %v", err)
		}
		firstBytes, err := first.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal first RLWE: %v", err)
		}
		secondBytes, err := second.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal second RLWE: %v", err)
		}
		if bytes.Equal(firstBytes, secondBytes) {
			t.Fatal("two encryptions of the same B row are byte-identical")
		}
		assertDecryptsTo(t, client, first, []int64{1, -2, 3})
		assertDecryptsTo(t, client, second, []int64{1, -2, 3})

		valueA, err := client.EncryptVA(5)
		if err != nil {
			t.Fatalf("first EncryptVA: %v", err)
		}
		valueB, err := client.EncryptVA(5)
		if err != nil {
			t.Fatalf("second EncryptVA: %v", err)
		}
		valueABytes, err := valueA.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal first encrypted VA: %v", err)
		}
		valueBBytes, err := valueB.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal second encrypted VA: %v", err)
		}
		if bytes.Equal(valueABytes, valueBBytes) {
			t.Fatal("two encryptions of the same VA value are byte-identical")
		}
		assertDecryptsTo(t, client, valueA, []int64{5, 5, 5})
		assertDecryptsTo(t, client, valueB, []int64{5, 5, 5})

		selectorA, err := client.EncryptSelector(true)
		if err != nil {
			t.Fatalf("first EncryptSelector: %v", err)
		}
		selectorB, err := client.EncryptSelector(true)
		if err != nil {
			t.Fatalf("second EncryptSelector: %v", err)
		}
		selectorABytes, err := selectorA.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal first RGSW: %v", err)
		}
		selectorBBytes, err := selectorB.MarshalBinary()
		if err != nil {
			t.Fatalf("marshal second RGSW: %v", err)
		}
		if bytes.Equal(selectorABytes, selectorBBytes) {
			t.Fatal("two encryptions of the same selector are byte-identical")
		}
	})

	selectorZero, err := client.EncryptSelector(false)
	if err != nil {
		t.Fatalf("EncryptSelector(false): %v", err)
	}
	selectorOne, err := client.EncryptSelector(true)
	if err != nil {
		t.Fatalf("EncryptSelector(true): %v", err)
	}

	t.Run("server Add and Sub preserve RLWE semantics", func(t *testing.T) {
		left, err := client.EncryptBRow([]int64{4, -1, 6})
		if err != nil {
			t.Fatalf("encrypt left: %v", err)
		}
		right, err := client.EncryptBRow([]int64{3, 5, -2})
		if err != nil {
			t.Fatalf("encrypt right: %v", err)
		}

		sum, err := server.Add(left, right)
		if err != nil {
			t.Fatalf("Add: %v", err)
		}
		difference, err := server.Sub(left, right)
		if err != nil {
			t.Fatalf("Sub: %v", err)
		}
		assertDecryptsTo(t, client, sum, []int64{7, 4, 4})
		assertDecryptsTo(t, client, difference, []int64{1, -6, 8})
	})

	t.Run("external product selects zero or identity", func(t *testing.T) {
		input, err := client.EncryptBRow([]int64{7, -3, 11})
		if err != nil {
			t.Fatalf("EncryptBRow: %v", err)
		}
		zeroProduct, err := server.ExternalProduct(input, selectorZero)
		if err != nil {
			t.Fatalf("ExternalProduct selector=0: %v", err)
		}
		oneProduct, err := server.ExternalProduct(input, selectorOne)
		if err != nil {
			t.Fatalf("ExternalProduct selector=1: %v", err)
		}
		assertDecryptsTo(t, client, zeroProduct, []int64{0, 0, 0})
		assertDecryptsTo(t, client, oneProduct, []int64{7, -3, 11})
	})

	t.Run("HSwitch selector zero passes and selector one swaps", func(t *testing.T) {
		a, err := client.EncryptBRow([]int64{1, 2, 3})
		if err != nil {
			t.Fatalf("encrypt a: %v", err)
		}
		b, err := client.EncryptBRow([]int64{9, 8, 7})
		if err != nil {
			t.Fatalf("encrypt b: %v", err)
		}

		outA, outB, err := server.HSwitch(a, b, selectorZero)
		if err != nil {
			t.Fatalf("HSwitch selector=0: %v", err)
		}
		assertDecryptsTo(t, client, outA, []int64{1, 2, 3})
		assertDecryptsTo(t, client, outB, []int64{9, 8, 7})

		outA, outB, err = server.HSwitch(a, b, selectorOne)
		if err != nil {
			t.Fatalf("HSwitch selector=1: %v", err)
		}
		assertDecryptsTo(t, client, outA, []int64{9, 8, 7})
		assertDecryptsTo(t, client, outB, []int64{1, 2, 3})
	})

	t.Run("41 sequential HSwitch operations remain decryptable before Apply", func(t *testing.T) {
		a, err := client.EncryptBRow([]int64{1, 2, 3})
		if err != nil {
			t.Fatalf("encrypt a: %v", err)
		}
		b, err := client.EncryptBRow([]int64{9, 8, 7})
		if err != nil {
			t.Fatalf("encrypt b: %v", err)
		}
		for step := 0; step < 41; step++ {
			selector, err := client.EncryptSelector(true)
			if err != nil {
				t.Fatalf("encrypt selector at step %d: %v", step, err)
			}
			a, b, err = server.HSwitch(a, b, selector)
			if err != nil {
				t.Fatalf("HSwitch at step %d: %v", step, err)
			}
		}
		assertDecryptsTo(t, client, a, []int64{9, 8, 7})
		assertDecryptsTo(t, client, b, []int64{1, 2, 3})

		value, err := client.EncryptVA(2)
		if err != nil {
			t.Fatalf("EncryptVA: %v", err)
		}
		product, err := server.Apply(a, value)
		if err != nil {
			t.Fatalf("Apply after HSwitch cascade: %v", err)
		}
		assertDecryptsTo(t, client, product, []int64{18, 16, 14})
	})

	t.Run("Apply is RLWE multiplication with relinearization", func(t *testing.T) {
		row, err := client.EncryptBRow([]int64{2, -4, 5})
		if err != nil {
			t.Fatalf("EncryptBRow: %v", err)
		}
		value, err := client.EncryptVA(-3)
		if err != nil {
			t.Fatalf("EncryptVA: %v", err)
		}
		product, err := server.Apply(row, value)
		if err != nil {
			t.Fatalf("Apply: %v", err)
		}
		if product.Degree() != 1 {
			t.Fatalf("Apply output degree=%d, want relinearized degree 1", product.Degree())
		}
		assertDecryptsTo(t, client, product, []int64{-6, 12, -15})
	})
}

func TestServerExposesNoDecryptionAPIOrSecretState(t *testing.T) {
	serverType := reflect.TypeOf(fhe.Server{})
	serverPointerType := reflect.TypeOf((*fhe.Server)(nil))

	for i := 0; i < serverPointerType.NumMethod(); i++ {
		method := serverPointerType.Method(i)
		if strings.Contains(strings.ToLower(method.Name), "decrypt") {
			t.Fatalf("Server exposes decryption method %q", method.Name)
		}
	}

	for i := 0; i < serverType.NumField(); i++ {
		field := serverType.Field(i)
		fieldType := strings.ToLower(field.Type.String())
		if strings.Contains(fieldType, "secretkey") || strings.Contains(fieldType, "decryptor") {
			t.Fatalf("Server field %q contains secret/decryption state of type %s", field.Name, field.Type)
		}
	}
}

func assertDecryptsTo(t *testing.T, client *fhe.Client, ciphertext *fhe.Ciphertext, want []int64) {
	t.Helper()
	have, err := client.DecryptRow(ciphertext, len(want))
	if err != nil {
		t.Fatalf("DecryptRow: %v", err)
	}
	if !reflect.DeepEqual(have, want) {
		t.Fatalf("decrypted row=%v, want %v", have, want)
	}
}
