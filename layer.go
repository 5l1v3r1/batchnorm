package batchnorm

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

// A Layer can be placed in a neuralnet.Network to apply
// Batch Normalization at that place in the network.
type Layer struct {
	InputCount int

	// These parameters are learned by the neural network
	// to allow it to potentially "undo" the effects of
	// Batch Normalization if that is optimal.
	Biases *autofunc.Variable
	Scales *autofunc.Variable

	// These parameters are updated at each mini-batch to
	// normalize the input to this layer.
	NegMeans   linalg.Vector
	InvStddevs linalg.Vector
}

// NewLayer creates a layer with pre-initialized variables
// and moment vectors.
func NewLayer(inCount int) *Layer {
	biases := &autofunc.Variable{Vector: make(linalg.Vector, inCount)}
	scales := &autofunc.Variable{Vector: make(linalg.Vector, inCount)}
	means := make(linalg.Vector, inCount)
	variances := make(linalg.Vector, inCount)
	for i := 0; i < inCount; i++ {
		variances[i] = 1
		scales.Vector[i] = 1
	}
	return &Layer{
		InputCount: inCount,
		Biases:     biases,
		Scales:     scales,
		NegMeans:   means,
		InvStddevs: variances,
	}
}

// Parameters returns a slice containing the learned
// biases and scales.
func (l *Layer) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{l.Biases, l.Scales}
}

// Apply applies batch normalization to the input.
func (l *Layer) Apply(in autofunc.Result) autofunc.Result {
	if len(in.Output()) != l.InputCount {
		panic("incorrect input size")
	}
	normalized := autofunc.Mul(autofunc.Add(in, &autofunc.Variable{Vector: l.NegMeans}),
		&autofunc.Variable{Vector: l.InvStddevs})
	return autofunc.Add(autofunc.Mul(normalized, l.Scales), l.Biases)
}

// ApplyR is like Apply but with r-operator support.
func (l *Layer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if len(in.Output()) != l.InputCount {
		panic("incorrect input size")
	}
	zeroVec := make(linalg.Vector, l.InputCount)
	negMeanVar := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: l.NegMeans},
		ROutputVec: zeroVec,
	}
	invStdVar := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: l.InvStddevs},
		ROutputVec: zeroVec,
	}
	normalized := autofunc.MulR(autofunc.AddR(in, negMeanVar), invStdVar)
	return autofunc.AddR(autofunc.MulR(normalized, autofunc.NewRVariable(l.Scales, rv)),
		autofunc.NewRVariable(l.Biases, rv))
}
