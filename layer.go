package batchnorm

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func init() {
	var l Layer
	serializer.RegisterTypedDeserializer(l.SerializerType(), DeserializeLayer)
}

// A Layer can be placed in a neuralnet.Network to apply
// Batch Normalization at that place in the network.
type Layer struct {
	// InputCount specifies the number of independently
	// normalized inputs to this layer.
	// For placement after neuralnet.DenseLayers, this
	// should be the full size of the input.
	// For placement after neuralnet.ConvLayer, this
	// should be the number of filters.
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

// DeserializeLayer deserializes a Layer.
func DeserializeLayer(d []byte) (*Layer, error) {
	var res Layer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
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
// The input's length must be divisible by l.InputCount.
func (l *Layer) Apply(in autofunc.Result) autofunc.Result {
	if len(in.Output())%l.InputCount != 0 {
		panic("invalid input size")
	}
	n := len(in.Output()) / l.InputCount

	negMean := autofunc.Repeat(&autofunc.Variable{Vector: l.NegMeans}, n)
	invStd := autofunc.Repeat(&autofunc.Variable{Vector: l.InvStddevs}, n)
	scales := autofunc.Repeat(l.Scales, n)
	biases := autofunc.Repeat(l.Biases, n)

	normalized := autofunc.Mul(autofunc.Add(in, negMean), invStd)
	return autofunc.Add(autofunc.Mul(normalized, scales), biases)
}

// ApplyR is like Apply but with r-operator support.
func (l *Layer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if len(in.Output())%l.InputCount != 0 {
		panic("invalid input size")
	}
	n := len(in.Output()) / l.InputCount

	zeroVec := make(linalg.Vector, l.InputCount)
	negMeanVar := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: l.NegMeans},
		ROutputVec: zeroVec,
	}
	invStdVar := &autofunc.RVariable{
		Variable:   &autofunc.Variable{Vector: l.InvStddevs},
		ROutputVec: zeroVec,
	}

	negMean := autofunc.RepeatR(negMeanVar, n)
	invStd := autofunc.RepeatR(invStdVar, n)
	scales := autofunc.RepeatR(autofunc.NewRVariable(l.Scales, rv), n)
	biases := autofunc.RepeatR(autofunc.NewRVariable(l.Biases, rv), n)

	normalized := autofunc.MulR(autofunc.AddR(in, negMean), invStd)
	return autofunc.AddR(autofunc.MulR(normalized, scales), biases)
}

// SerializerType returns the unique ID used to serialize
// this layer with the serializer package.
func (l *Layer) SerializerType() string {
	return "github.com/unixpickle/batchnorm.Layer"
}

// Serialize serializes the parameters of this layer.
func (l *Layer) Serialize() ([]byte, error) {
	return json.Marshal(l)
}
