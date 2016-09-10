package batchnorm

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type meanResult struct {
	Input     autofunc.Result
	OutputVec linalg.Vector
	N         int
}

func computeMeans(in autofunc.Result, size int) autofunc.Result {
	var count int
	res := make(linalg.Vector, size)
	for i := 0; i < len(in.Output()); i += size {
		res.Add(in.Output()[i : i+size])
		count++
	}
	res.Scale(1 / float64(count))
	return &meanResult{
		Input:     in,
		OutputVec: res,
		N:         count,
	}
}

func (m *meanResult) Output() linalg.Vector {
	return m.OutputVec
}

func (m *meanResult) Constant(grad autofunc.Gradient) bool {
	return m.Input.Constant(grad)
}

func (m *meanResult) PropagateGradient(upstream linalg.Vector, g autofunc.Gradient) {
	if m.Input.Constant(g) {
		return
	}
	upstream.Scale(1 / float64(m.N))
	downstream := make(linalg.Vector, len(m.Input.Output()))
	for i := 0; i < len(downstream); i += len(upstream) {
		downstream[i : i+len(m.OutputVec)].Add(upstream)
	}
	m.Input.PropagateGradient(downstream, g)
}

type meanSquareResult struct {
	Input     autofunc.Result
	OutputVec linalg.Vector
	N         int
}

func computeMeanSquares(in autofunc.Result, size int) autofunc.Result {
	var count int
	res := make(linalg.Vector, size)
	for i := 0; i < len(in.Output()); i += size {
		for j, x := range in.Output()[i : i+size] {
			res[j] += x * x
		}
		count++
	}
	res.Scale(1 / float64(count))
	return &meanSquareResult{
		Input:     in,
		OutputVec: res,
		N:         count,
	}
}

func (m *meanSquareResult) Output() linalg.Vector {
	return m.OutputVec
}

func (m *meanSquareResult) Constant(grad autofunc.Gradient) bool {
	return m.Input.Constant(grad)
}

func (m *meanSquareResult) PropagateGradient(upstream linalg.Vector, g autofunc.Gradient) {
	if m.Input.Constant(g) {
		return
	}
	upstream.Scale(1 / float64(m.N))
	downstream := make(linalg.Vector, len(m.Input.Output()))
	inVec := m.Input.Output()
	for i := 0; i < len(downstream); i += len(upstream) {
		for j, u := range upstream {
			downstream[i+j] = 2 * u * inVec[i+j]
		}
	}
	m.Input.PropagateGradient(downstream, g)
}
