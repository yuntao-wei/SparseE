package permutation

import (
	"math/rand"
	"reflect"
	"testing"
)

func TestBuildGatherMapPaperExample(t *testing.T) {
	got, err := BuildGatherMap([]int{0, 1, 2, 1, 2}, []int{1, 2, 2})
	if err != nil {
		t.Fatalf("BuildGatherMap() error = %v", err)
	}
	want := []int{0, 1, 3, 2, 4}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("BuildGatherMap() = %v, want %v", got, want)
	}
}

func TestBuildGatherMapUsesStableOccurrencesAndDirection(t *testing.T) {
	co := []int{2, 0, 2, 1}
	cd := []int{1, 1, 2}
	destToSrc, err := BuildGatherMap(co, cd)
	if err != nil {
		t.Fatalf("BuildGatherMap() error = %v", err)
	}
	want := []int{2, 0, 3, 1}
	if !reflect.DeepEqual(destToSrc, want) {
		t.Fatalf("destToSrc = %v, want %v", destToSrc, want)
	}
	sourceToDest, err := InvertDestToSrc(destToSrc)
	if err != nil {
		t.Fatalf("InvertDestToSrc() error = %v", err)
	}
	if reflect.DeepEqual(destToSrc, sourceToDest) {
		t.Fatalf("test permutation unexpectedly self-inverse: %v", destToSrc)
	}

	scatteredLabels := []string{"B0#0", "B1#0", "B2#0", "B2#1"}
	got, err := ReferencePermute(scatteredLabels, destToSrc)
	if err != nil {
		t.Fatalf("ReferencePermute() error = %v", err)
	}
	wantLabels := []string{"B2#0", "B0#0", "B2#1", "B1#0"}
	if !reflect.DeepEqual(got, wantLabels) {
		t.Fatalf("ReferencePermute() = %v, want %v", got, wantLabels)
	}
}

func TestBuildGatherMapRejectsInvalidMetadata(t *testing.T) {
	tests := []struct {
		name string
		co   []int
		cd   []int
	}{
		{"sum mismatch", []int{0}, []int{2}},
		{"negative degree", nil, []int{-1}},
		{"negative column", []int{-1}, []int{1}},
		{"column out of range", []int{1}, []int{1}},
		{"too many occurrences", []int{0, 0}, []int{1, 1}},
	}
	for _, test := range tests {
		t.Run(test.name, func(t *testing.T) {
			if _, err := BuildGatherMap(test.co, test.cd); err == nil {
				t.Fatal("BuildGatherMap() error = nil")
			}
		})
	}
}

func TestNextPowerOfTwo(t *testing.T) {
	tests := map[int]int{0: 0, 1: 1, 2: 2, 3: 4, 5: 8, 8: 8, 9: 16}
	for input, want := range tests {
		got, err := NextPowerOfTwo(input)
		if err != nil {
			t.Fatalf("NextPowerOfTwo(%d) error = %v", input, err)
		}
		if got != want {
			t.Fatalf("NextPowerOfTwo(%d) = %d, want %d", input, got, want)
		}
	}
	if _, err := NextPowerOfTwo(-1); err == nil {
		t.Fatal("NextPowerOfTwo(-1) error = nil")
	}
}

func TestCompileBenesPadsArbitraryE(t *testing.T) {
	destToSrc := []int{2, 0, 4, 1, 3}
	plan, err := CompileBenes(destToSrc)
	if err != nil {
		t.Fatalf("CompileBenes() error = %v", err)
	}
	if plan.LogicalSize != 5 || plan.NetworkSize != 8 {
		t.Fatalf("plan sizes = (%d,%d), want (5,8)", plan.LogicalSize, plan.NetworkSize)
	}
	if got, want := plan.SelectorCount(), 20; got != want {
		t.Fatalf("SelectorCount() = %d, want %d", got, want)
	}

	input := identity(plan.NetworkSize)
	output, err := ApplyPlanToIndices(plan, input)
	if err != nil {
		t.Fatalf("ApplyPlanToIndices() error = %v", err)
	}
	want := []int{2, 0, 4, 1, 3, 5, 6, 7}
	if !reflect.DeepEqual(output, want) {
		t.Fatalf("ApplyPlanToIndices() = %v, want %v", output, want)
	}
}

func TestBenesSpecialCases(t *testing.T) {
	empty, err := CompileBenes(nil)
	if err != nil {
		t.Fatalf("CompileBenes(nil) error = %v", err)
	}
	output, err := ApplyPlanToIndices(empty, nil)
	if err != nil || len(output) != 0 {
		t.Fatalf("empty ApplyPlanToIndices() = %v, %v", output, err)
	}

	single, err := CompileBenes([]int{0})
	if err != nil {
		t.Fatalf("CompileBenes(single) error = %v", err)
	}
	output, err = ApplyPlanToIndices(single, []int{42})
	if err != nil || !reflect.DeepEqual(output, []int{42}) {
		t.Fatalf("single ApplyPlanToIndices() = %v, %v", output, err)
	}
}

func TestBenesExhaustiveArbitrarySmallSizes(t *testing.T) {
	for size := 0; size <= 7; size++ {
		permutation := identity(size)
		checked := 0
		for {
			assertPlanRealizes(t, permutation)
			checked++
			if !nextPermutation(permutation) {
				break
			}
		}
		if checked == 0 {
			t.Fatalf("size %d checked no permutations", size)
		}
	}
}

func TestBenesRandomLargerPermutations(t *testing.T) {
	rng := rand.New(rand.NewSource(20260816))
	for _, size := range []int{8, 13, 16, 31, 32, 65} {
		for trial := 0; trial < 20; trial++ {
			assertPlanRealizes(t, rng.Perm(size))
		}
	}
}

func TestCompileBenesDeterministic(t *testing.T) {
	permutation := []int{5, 1, 6, 0, 3, 2, 4}
	first, err := CompileBenes(permutation)
	if err != nil {
		t.Fatal(err)
	}
	second, err := CompileBenes(permutation)
	if err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(first, second) {
		t.Fatal("CompileBenes() is not deterministic")
	}
}

func TestPermutationValidation(t *testing.T) {
	invalid := [][]int{{0, 0}, {0, 2}, {-1, 0}}
	for _, permutation := range invalid {
		if _, err := InvertDestToSrc(permutation); err == nil {
			t.Fatalf("InvertDestToSrc(%v) error = nil", permutation)
		}
		if _, err := CompileBenes(permutation); err == nil {
			t.Fatalf("CompileBenes(%v) error = nil", permutation)
		}
	}

	plan, err := CompileBenes([]int{1, 0})
	if err != nil {
		t.Fatal(err)
	}
	if _, err := ApplyPlanToIndices(plan, []int{0}); err == nil {
		t.Fatal("ApplyPlanToIndices(size mismatch) error = nil")
	}
	if _, err := ReferencePermute([]int{0}, []int{1, 0}); err == nil {
		t.Fatal("ReferencePermute(size mismatch) error = nil")
	}
}

func assertPlanRealizes(t *testing.T, destToSrc []int) {
	t.Helper()
	plan, err := CompileBenes(destToSrc)
	if err != nil {
		t.Fatalf("CompileBenes(%v) error = %v", destToSrc, err)
	}
	output, err := ApplyPlanToIndices(plan, identity(plan.NetworkSize))
	if err != nil {
		t.Fatalf("ApplyPlanToIndices(%v) error = %v", destToSrc, err)
	}
	want := append([]int(nil), destToSrc...)
	for i := len(destToSrc); i < plan.NetworkSize; i++ {
		want = append(want, i)
	}
	if !reflect.DeepEqual(output, want) {
		t.Fatalf("Benes output for %v = %v, want %v", destToSrc, output, want)
	}
}

func identity(size int) []int {
	values := make([]int, size)
	for i := range values {
		values[i] = i
	}
	return values
}

func nextPermutation(values []int) bool {
	i := len(values) - 2
	for i >= 0 && values[i] >= values[i+1] {
		i--
	}
	if i < 0 {
		return false
	}
	j := len(values) - 1
	for values[j] <= values[i] {
		j--
	}
	values[i], values[j] = values[j], values[i]
	for left, right := i+1, len(values)-1; left < right; left, right = left+1, right-1 {
		values[left], values[right] = values[right], values[left]
	}
	return true
}
