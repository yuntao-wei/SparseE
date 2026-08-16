// Package workloads defines deterministic SparseE protocol inputs.
package workloads

// Workload defines a structurally representative protocol input.
type Workload struct {
	Name       string
	SparseA    [][]int64
	DenseB     [][]int64
	ExpectedCD []int
	ExpectedRP []int
}

// RepresentativeWorkloads returns small inputs for four SparseE application families.
func RepresentativeWorkloads() []Workload {
	return []Workload{
		{
			Name: "gnn_adjacency_aggregation",
			SparseA: [][]int64{
				{1, 0, 1, 0, 0, 1},
				{1, 0, 1, 0, 0, 0},
				{0, 1, 1, 1, 0, 0},
				{0, 0, 0, 0, 0, 0},
				{1, 0, 1, 1, 0, 1},
				{0, 0, 0, 0, 1, 0},
			},
			DenseB: [][]int64{
				{1, 2, 0}, {0, 1, 3}, {2, -1, 1},
				{-1, 0, 2}, {3, 1, 1}, {1, 1, -2},
			},
			ExpectedCD: []int{3, 1, 4, 2, 1, 2},
			ExpectedRP: []int{0, 3, 5, 8, 8, 12, 13},
		},
		{
			Name: "llm_one_hot_embedding",
			SparseA: [][]int64{
				{0, 0, 0, 1, 0, 0, 0, 0},
				{0, 1, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 1, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 0, 0, 1},
				{1, 0, 0, 0, 0, 0, 0, 0},
				{0, 0, 0, 1, 0, 0, 0, 0},
				{0, 0, 0, 0, 0, 1, 0, 0},
			},
			DenseB: [][]int64{
				{1, 0, 0, 1}, {0, 2, 1, -1}, {3, 1, 0, 0}, {-1, 1, 2, 0},
				{2, 2, -2, 1}, {0, -1, 3, 2}, {1, 1, 1, 1}, {4, 0, -1, 1},
			},
			ExpectedCD: []int{1, 1, 0, 3, 0, 1, 0, 1},
			ExpectedRP: []int{0, 1, 2, 3, 4, 5, 6, 7},
		},
		{
			Name: "three_d_cnn_voxel_occupancy",
			SparseA: [][]int64{
				{1, 2, 0, 0, -1, 0, 0, 0},
				{0, 1, 1, 0, 0, 1, 0, 0},
				{0, 0, 0, -1, 0, 0, 2, 0},
				{1, 0, 0, 0, 1, 0, 0, 1},
				{0, 0, 0, 0, 0, 0, 1, 0},
			},
			DenseB: [][]int64{
				{1, 0, 2}, {0, 1, 1}, {2, 1, 0}, {-1, 2, 1},
				{3, 0, -1}, {1, -1, 2}, {0, 2, 3}, {2, 2, 2},
			},
			ExpectedCD: []int{2, 2, 1, 1, 2, 1, 2, 1},
			ExpectedRP: []int{0, 3, 6, 8, 11, 12},
		},
		{
			Name: "recsys_user_item_interactions",
			SparseA: [][]int64{
				{5, 0, 1, 0, 0, 0, 2},
				{0, 3, 0, 0, 0, 0, 0},
				{0, 0, 4, 1, 0, 2, 0},
				{0, 0, 0, 0, 0, 0, 0},
				{1, 0, 0, 0, 5, 1, 1},
			},
			DenseB: [][]int64{
				{1, 0, 2, 1}, {0, 2, -1, 1}, {3, 1, 0, 0}, {1, -1, 2, 1},
				{0, 1, 1, 3}, {2, 0, 1, -1}, {1, 1, 1, 0},
			},
			ExpectedCD: []int{2, 1, 2, 1, 1, 2, 2},
			ExpectedRP: []int{0, 3, 4, 7, 7, 11},
		},
	}
}
