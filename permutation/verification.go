package permutation

import "fmt"

// ApplyPlanToIndices evaluates a Benes plan over plaintext indices.
func ApplyPlanToIndices(plan BenesPlan, input []int) ([]int, error) {
	if err := validatePlan(plan, true); err != nil {
		return nil, err
	}
	if len(input) != plan.NetworkSize {
		return nil, fmt.Errorf("Benes input length is %d; want network size %d", len(input), plan.NetworkSize)
	}
	return applyPlan(plan, input)
}

// ReferencePermute applies destination-to-source indexing directly.
func ReferencePermute[T any](input []T, destToSrc []int) ([]T, error) {
	if len(input) != len(destToSrc) {
		return nil, fmt.Errorf("reference input length is %d; want permutation length %d", len(input), len(destToSrc))
	}
	if _, err := InvertDestToSrc(destToSrc); err != nil {
		return nil, err
	}
	output := make([]T, len(input))
	for dst, src := range destToSrc {
		output[dst] = input[src]
	}
	return output, nil
}

func applyPlan(plan BenesPlan, input []int) ([]int, error) {
	n := plan.NetworkSize
	if n <= 1 {
		return append([]int(nil), input...), nil
	}
	if n == 2 {
		switched := switchPair(input[0], input[1], plan.First[0])
		return []int{switched[0], switched[1]}, nil
	}

	upper := make([]int, n/2)
	lower := make([]int, n/2)
	for pair := 0; pair < n/2; pair++ {
		switched := switchPair(input[2*pair], input[2*pair+1], plan.First[pair])
		upper[pair], lower[pair] = switched[0], switched[1]
	}
	upperOut, err := applyPlan(plan.Children[0], upper)
	if err != nil {
		return nil, err
	}
	lowerOut, err := applyPlan(plan.Children[1], lower)
	if err != nil {
		return nil, err
	}

	output := make([]int, n)
	for pair := 0; pair < n/2; pair++ {
		switched := switchPair(upperOut[pair], lowerOut[pair], plan.Last[pair])
		output[2*pair], output[2*pair+1] = switched[0], switched[1]
	}
	return output, nil
}

func switchPair(a, b int, swap bool) [2]int {
	if swap {
		return [2]int{b, a}
	}
	return [2]int{a, b}
}

func validatePlan(plan BenesPlan, root bool) error {
	n := plan.NetworkSize
	if n == 0 {
		if plan.LogicalSize != 0 || len(plan.First) != 0 || len(plan.Last) != 0 || len(plan.Children) != 0 {
			return fmt.Errorf("empty Benes plan contains routing data")
		}
		return nil
	}
	if n < 0 || n&(n-1) != 0 {
		return fmt.Errorf("Benes network size %d is not a positive power of two", n)
	}
	if plan.LogicalSize < 0 || plan.LogicalSize > n {
		return fmt.Errorf("logical size %d is outside [0,%d]", plan.LogicalSize, n)
	}
	if !root && plan.LogicalSize != n {
		return fmt.Errorf("child logical size %d differs from network size %d", plan.LogicalSize, n)
	}
	if n == 1 {
		if len(plan.First) != 0 || len(plan.Last) != 0 || len(plan.Children) != 0 {
			return fmt.Errorf("one-wire Benes plan contains switches")
		}
		return nil
	}
	if n == 2 {
		if len(plan.First) != 1 || len(plan.Last) != 0 || len(plan.Children) != 0 {
			return fmt.Errorf("two-wire Benes plan has invalid shape")
		}
		return nil
	}
	if len(plan.First) != n/2 || len(plan.Last) != n/2 || len(plan.Children) != 2 {
		return fmt.Errorf("%d-wire Benes plan has invalid recursive shape", n)
	}
	for i := range plan.Children {
		if plan.Children[i].NetworkSize != n/2 {
			return fmt.Errorf("child %d network size is %d; want %d", i, plan.Children[i].NetworkSize, n/2)
		}
		if err := validatePlan(plan.Children[i], false); err != nil {
			return err
		}
	}
	return nil
}
