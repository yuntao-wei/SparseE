package fhe

import (
	"reflect"
	"testing"

	"github.com/tuneinsight/lattigo/v6/core/rgsw"
	"github.com/tuneinsight/lattigo/v6/core/rlwe"
)

type countingExternalProductEvaluator struct {
	delegate externalProductEvaluator
	calls    int
}

func (e *countingExternalProductEvaluator) ExternalProduct(
	op0 *rlwe.Ciphertext,
	op1 *rgsw.Ciphertext,
	opOut *rlwe.Ciphertext,
) {
	e.calls++
	e.delegate.ExternalProduct(op0, op1, opOut)
}

func TestHSwitchUsesOneExternalProduct(t *testing.T) {
	client, server, err := NewRoles()
	if err != nil {
		t.Fatalf("NewRoles: %v", err)
	}

	counter := &countingExternalProductEvaluator{delegate: server.rgswEvaluator}
	server.rgswEvaluator = counter

	for _, test := range []struct {
		name  string
		bit   bool
		wantA []int64
		wantB []int64
	}{
		{name: "pass", bit: false, wantA: []int64{1, 2, 3}, wantB: []int64{9, 8, 7}},
		{name: "swap", bit: true, wantA: []int64{9, 8, 7}, wantB: []int64{1, 2, 3}},
	} {
		t.Run(test.name, func(t *testing.T) {
			a, err := client.EncryptBRow([]int64{1, 2, 3})
			if err != nil {
				t.Fatalf("EncryptBRow(a): %v", err)
			}
			b, err := client.EncryptBRow([]int64{9, 8, 7})
			if err != nil {
				t.Fatalf("EncryptBRow(b): %v", err)
			}
			selector, err := client.EncryptSelector(test.bit)
			if err != nil {
				t.Fatalf("EncryptSelector: %v", err)
			}

			callsBefore := counter.calls
			outA, outB, err := server.HSwitch(a, b, selector)
			if err != nil {
				t.Fatalf("HSwitch: %v", err)
			}
			if calls := counter.calls - callsBefore; calls != 1 {
				t.Fatalf("HSwitch called RGSW ExternalProduct %d times, want 1", calls)
			}

			gotA, err := client.DecryptRow(outA, 3)
			if err != nil {
				t.Fatalf("DecryptRow(outA): %v", err)
			}
			gotB, err := client.DecryptRow(outB, 3)
			if err != nil {
				t.Fatalf("DecryptRow(outB): %v", err)
			}
			if !reflect.DeepEqual(gotA, test.wantA) {
				t.Fatalf("outA=%v, want %v", gotA, test.wantA)
			}
			if !reflect.DeepEqual(gotB, test.wantB) {
				t.Fatalf("outB=%v, want %v", gotB, test.wantB)
			}
		})
	}
}
