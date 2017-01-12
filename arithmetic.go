package batchnorm

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

type addMulResult struct {
	OutputVec linalg.Vector
	Input     autofunc.Result
	Bias      autofunc.Result
	Scale     autofunc.Result
}

// addMul repeats the bias and scale n times and computes
// (in + bias) * scale.
func addMul(in, bias, scale autofunc.Result, n int) autofunc.Result {
	outVec := make(linalg.Vector, len(in.Output()))
	copy(outVec, in.Output())

	biasSize := len(bias.Output())
	for i := 0; i < n; i++ {
		dest := outVec[i*biasSize : (i+1)*biasSize]
		dest.Add(bias.Output())
		for j, x := range scale.Output() {
			dest[j] *= x
		}
	}

	return &addMulResult{
		OutputVec: outVec,
		Input:     in,
		Bias:      bias,
		Scale:     scale,
	}
}

func (a *addMulResult) Output() linalg.Vector {
	return a.OutputVec
}

func (a *addMulResult) Constant(g autofunc.Gradient) bool {
	return a.Input.Constant(g) && a.Bias.Constant(g) && a.Scale.Constant(g)
}

func (a *addMulResult) PropagateGradient(upstream linalg.Vector, g autofunc.Gradient) {
	if !a.Scale.Constant(g) {
		downstream := make(linalg.Vector, len(a.Scale.Output()))
		inVec := a.Input.Output()
		for i := 0; i < len(upstream); i += len(downstream) {
			for j, b := range a.Bias.Output() {
				downstream[j] += upstream[i+j] * (inVec[i+j] + b)
			}
		}
		a.Scale.PropagateGradient(downstream, g)
	}
	if !a.Bias.Constant(g) {
		downstream := make(linalg.Vector, len(a.Bias.Output()))
		for i := 0; i < len(upstream); i += len(downstream) {
			for j, s := range a.Scale.Output() {
				downstream[j] += upstream[i+j] * s
			}
		}
		a.Bias.PropagateGradient(downstream, g)
	}
	if !a.Input.Constant(g) {
		for i := 0; i < len(upstream); i += len(a.Scale.Output()) {
			for j, s := range a.Scale.Output() {
				upstream[i+j] *= s
			}
		}
		a.Input.PropagateGradient(upstream, g)
	}
}
