package permutation

import "fmt"

const intMax = int(^uint(0) >> 1)

// BenesPlan is a deterministic recursive Radix-2 Benes routing plan.
// LogicalSize is the unpadded E at the root, NetworkSize is the public
// power-of-two wire count, and false/true controls mean pass/swap.
type BenesPlan struct {
	LogicalSize int
	NetworkSize int
	First       []bool
	Last        []bool
	Children    []BenesPlan
}

// NextPowerOfTwo returns the smallest power of two greater than or equal to
// value. Zero represents an empty network.
func NextPowerOfTwo(value int) (int, error) {
	if value < 0 {
		return 0, fmt.Errorf("network size must be nonnegative")
	}
	if value <= 1 {
		return value, nil
	}
	result := 1
	for result < value {
		if result > intMax/2 {
			return 0, fmt.Errorf("next power of two for %d overflows int", value)
		}
		result *= 2
	}
	return result, nil
}

// CompileBenes compiles a destination-to-source permutation into Benes switch
// controls and pads arbitrary sizes with identity wires.
func CompileBenes(destToSrc []int) (BenesPlan, error) {
	if _, err := InvertDestToSrc(destToSrc); err != nil {
		return BenesPlan{}, err
	}
	networkSize, err := NextPowerOfTwo(len(destToSrc))
	if err != nil {
		return BenesPlan{}, err
	}
	if networkSize == 0 {
		return BenesPlan{}, nil
	}

	paddedDestToSrc := make([]int, networkSize)
	copy(paddedDestToSrc, destToSrc)
	for i := len(destToSrc); i < networkSize; i++ {
		paddedDestToSrc[i] = i
	}
	sourceToDest, err := InvertDestToSrc(paddedDestToSrc)
	if err != nil {
		return BenesPlan{}, err
	}
	plan, err := compileSourceToDest(sourceToDest)
	if err != nil {
		return BenesPlan{}, err
	}
	plan.LogicalSize = len(destToSrc)
	return plan, nil
}

// SelectorCount returns the number of HSwitch controls in the recursive plan.
func (plan BenesPlan) SelectorCount() int {
	count := len(plan.First) + len(plan.Last)
	for _, child := range plan.Children {
		count += child.SelectorCount()
	}
	return count
}

func compileSourceToDest(sourceToDest []int) (BenesPlan, error) {
	n := len(sourceToDest)
	plan := BenesPlan{LogicalSize: n, NetworkSize: n}
	if n == 0 || n&(n-1) != 0 {
		return BenesPlan{}, fmt.Errorf("Benes network size %d is not a positive power of two", n)
	}
	if n == 1 {
		return plan, nil
	}
	if n == 2 {
		plan.First = []bool{sourceToDest[0] == 1}
		return plan, nil
	}

	outMembers, err := outputPairMembers(sourceToDest)
	if err != nil {
		return BenesPlan{}, err
	}
	color := make([]int8, n)
	for i := range color {
		color[i] = -1
	}
	queue := make([]int, 0, n)
	for start := 0; start < n; start++ {
		if color[start] != -1 {
			continue
		}
		color[start] = 0
		queue = append(queue[:0], start)
		for len(queue) != 0 {
			wire := queue[0]
			queue = queue[1:]

			inputMate := wire ^ 1
			if err := colorOpposite(color, wire, inputMate, &queue, "input"); err != nil {
				return BenesPlan{}, err
			}

			outPair := sourceToDest[wire] / 2
			members := outMembers[outPair]
			outputMate := members[0]
			if outputMate == wire {
				outputMate = members[1]
			}
			if err := colorOpposite(color, wire, outputMate, &queue, "output"); err != nil {
				return BenesPlan{}, err
			}
		}
	}

	plan.First = make([]bool, n/2)
	plan.Last = make([]bool, n/2)
	upper := make([]int, n/2)
	lower := make([]int, n/2)
	for pair := 0; pair < n/2; pair++ {
		plan.First[pair] = color[2*pair] != 0
	}
	for src, dst := range sourceToDest {
		subInput := src / 2
		subOutput := dst / 2
		if color[src] == 0 {
			upper[subInput] = subOutput
		} else {
			lower[subInput] = subOutput
		}
	}
	for outPair, members := range outMembers {
		topOutputSource := members[0]
		if sourceToDest[topOutputSource] != 2*outPair {
			topOutputSource = members[1]
		}
		plan.Last[outPair] = color[topOutputSource] != 0
	}

	upperPlan, err := compileSourceToDest(upper)
	if err != nil {
		return BenesPlan{}, err
	}
	lowerPlan, err := compileSourceToDest(lower)
	if err != nil {
		return BenesPlan{}, err
	}
	plan.Children = []BenesPlan{upperPlan, lowerPlan}
	return plan, nil
}

func outputPairMembers(sourceToDest []int) ([][2]int, error) {
	n := len(sourceToDest)
	if _, err := InvertDestToSrc(sourceToDest); err != nil {
		return nil, err
	}
	members := make([][2]int, n/2)
	counts := make([]int, n/2)
	for src, dst := range sourceToDest {
		pair := dst / 2
		if counts[pair] >= 2 {
			return nil, fmt.Errorf("invalid permutation for Benes output pair %d", pair)
		}
		members[pair][counts[pair]] = src
		counts[pair]++
	}
	for pair, count := range counts {
		if count != 2 {
			return nil, fmt.Errorf("Benes output pair %d has %d members; want 2", pair, count)
		}
	}
	return members, nil
}

func colorOpposite(color []int8, wire, mate int, queue *[]int, edge string) error {
	want := int8(1) - color[wire]
	if color[mate] == -1 {
		color[mate] = want
		*queue = append(*queue, mate)
		return nil
	}
	if color[mate] != want {
		return fmt.Errorf("Benes coloring conflict at %s pair", edge)
	}
	return nil
}
