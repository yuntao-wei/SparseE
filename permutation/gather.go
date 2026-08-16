// Package permutation implements SparseE's client-side Gather-map and Benes
// selector planning. It contains no FHE operations.
package permutation

import "fmt"

// BuildGatherMap returns the canonical destination-to-source permutation:
// gathered[dst] = scattered[destToSrc[dst]]. Scatter order is grouped by
// column, while destination order follows CO's stable occurrence order.
func BuildGatherMap(co, cd []int) ([]int, error) {
	prefix := make([]int, len(cd)+1)
	for col, degree := range cd {
		if degree < 0 {
			return nil, fmt.Errorf("CD[%d] is negative", col)
		}
		if degree > intMax-prefix[col] {
			return nil, fmt.Errorf("sum(CD) overflows int")
		}
		prefix[col+1] = prefix[col] + degree
	}
	if prefix[len(prefix)-1] != len(co) {
		return nil, fmt.Errorf("sum(CD) is %d; want len(CO) = %d", prefix[len(prefix)-1], len(co))
	}

	seen := make([]int, len(cd))
	destToSrc := make([]int, len(co))
	for dst, col := range co {
		if col < 0 || col >= len(cd) {
			return nil, fmt.Errorf("CO[%d] = %d is outside [0,%d)", dst, col, len(cd))
		}
		if seen[col] >= cd[col] {
			return nil, fmt.Errorf("CO occurrences for column %d exceed CD[%d] = %d", col, col, cd[col])
		}
		destToSrc[dst] = prefix[col] + seen[col]
		seen[col]++
	}
	for col, count := range seen {
		if count != cd[col] {
			return nil, fmt.Errorf("CO occurrences for column %d are %d; want CD[%d] = %d", col, count, col, cd[col])
		}
	}
	return destToSrc, nil
}

// InvertDestToSrc returns sourceToDest[src] for a valid destination-to-source
// permutation.
func InvertDestToSrc(destToSrc []int) ([]int, error) {
	inverse := make([]int, len(destToSrc))
	seen := make([]bool, len(destToSrc))
	for dst, src := range destToSrc {
		if src < 0 || src >= len(destToSrc) {
			return nil, fmt.Errorf("destToSrc[%d] = %d is outside [0,%d)", dst, src, len(destToSrc))
		}
		if seen[src] {
			return nil, fmt.Errorf("destToSrc contains duplicate source %d", src)
		}
		inverse[src] = dst
		seen[src] = true
	}
	return inverse, nil
}
