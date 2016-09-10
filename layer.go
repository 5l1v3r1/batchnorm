package batchnorm

import (
	"encoding/json"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

const defaultStabilizer = 1e-3

func init() {
	var l Layer
	serializer.RegisterTypedDeserializer(l.SerializerType(), DeserializeLayer)
}

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

	// Stabilizer is used to prevent numerical stability.
	// It should be a small number.
	// If it is 0, a reasonable default is used.
	Stabilizer float64

	// These variables are used once the network has
	// been fully trained and it is time to classify
	// individual samples.
	DoneTraining  bool
	FinalMean     linalg.Vector
	FinalVariance linalg.Vector
}

// DeserializeLayer deserializes a Layer.
func DeserializeLayer(d []byte) (*Layer, error) {
	var res Layer
	if err := json.Unmarshal(d, &res); err != nil {
		return nil, err
	}
	return &res, nil
}

// NewLayer creates a layer with pre-initialized variables.
func NewLayer(inputCount int) *Layer {
	res := &Layer{
		InputCount: inputCount,
		Biases:     &autofunc.Variable{Vector: make(linalg.Vector, inputCount)},
		Scales:     &autofunc.Variable{Vector: make(linalg.Vector, inputCount)},
	}
	for i := 0; i < inputCount; i++ {
		res.Scales.Vector[i] = 1
	}
	return res
}

// Parameters returns a list containing the bias and
// scale variables.
func (l *Layer) Parameters() []*autofunc.Variable {
	return []*autofunc.Variable{l.Biases, l.Scales}
}

// Apply applies batch normalization to the single sample.
// This is ideal for classification, but not for training,
// since the variances for a single sample are always 0.
func (l *Layer) Apply(in autofunc.Result) autofunc.Result {
	if l.DoneTraining {
		if len(in.Output())%l.InputCount != 0 {
			panic("invalid input size")
		}
		n := len(in.Output()) / l.InputCount

		meanVar := &autofunc.Variable{Vector: l.FinalMean}
		varVar := &autofunc.Variable{Vector: l.FinalVariance}
		negMean := autofunc.Repeat(autofunc.Scale(meanVar, -1), n)
		invStd := autofunc.Repeat(autofunc.Pow(autofunc.AddScaler(varVar,
			l.stabilizer()), -0.5), n)
		scales := autofunc.Repeat(l.Scales, n)
		biases := autofunc.Repeat(l.Biases, n)

		normalized := autofunc.Mul(autofunc.Add(in, negMean), invStd)
		return autofunc.Add(autofunc.Mul(normalized, scales), biases)
	}
	return l.Batch(in, 1)
}

// Apply is like ApplyR, but for RResults.
func (l *Layer) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	if l.DoneTraining {
		if len(in.Output())%l.InputCount != 0 {
			panic("invalid input size")
		}
		n := len(in.Output()) / l.InputCount

		meanVar := autofunc.NewRVariable(&autofunc.Variable{Vector: l.FinalMean}, rv)
		varVar := autofunc.NewRVariable(&autofunc.Variable{Vector: l.FinalVariance}, rv)
		negMean := autofunc.RepeatR(autofunc.ScaleR(meanVar, -1), n)
		invStd := autofunc.RepeatR(autofunc.PowR(autofunc.AddScalerR(varVar,
			l.stabilizer()), -0.5), n)
		scales := autofunc.RepeatR(autofunc.NewRVariable(l.Scales, rv), n)
		biases := autofunc.RepeatR(autofunc.NewRVariable(l.Biases, rv), n)

		normalized := autofunc.MulR(autofunc.AddR(in, negMean), invStd)
		return autofunc.AddR(autofunc.MulR(normalized, scales), biases)
	}
	return l.BatchR(rv, in, 1)
}

// Batch applies batch normalization to the batch.
func (l *Layer) Batch(in autofunc.Result, n int) autofunc.Result {
	if l.DoneTraining {
		f := autofunc.FuncBatcher{F: l}
		return f.Batch(in, n)
	}
	if len(in.Output())%l.InputCount != 0 {
		panic("invalid input size")
	}
	n = len(in.Output()) / l.InputCount
	return autofunc.Pool(in, func(in autofunc.Result) autofunc.Result {
		mean := computeMeans(in, l.InputCount)
		variance := computeMeanSquares(in, l.InputCount)
		meanSquared := autofunc.Square(mean)
		variance = autofunc.Add(variance, autofunc.Scale(meanSquared, -1))

		normalized := addMul(in, autofunc.Scale(mean, -1),
			autofunc.Pow(autofunc.AddScaler(variance, l.stabilizer()), -0.5), n)
		return mulAdd(normalized, l.Scales, l.Biases, n)
	})
}

// BatchR is like Batch, but for RResults.
func (l *Layer) BatchR(rv autofunc.RVector, in autofunc.RResult, n int) autofunc.RResult {
	if l.DoneTraining {
		f := autofunc.RFuncBatcher{F: l}
		return f.BatchR(rv, in, n)
	}
	if len(in.Output())%l.InputCount != 0 {
		panic("invalid input size")
	}
	n = len(in.Output()) / l.InputCount
	return autofunc.PoolR(in, func(in autofunc.RResult) autofunc.RResult {
		var mean autofunc.RResult
		var variance autofunc.RResult
		for i := 0; i < n; i++ {
			vec := autofunc.SliceR(in, i*l.InputCount, (i+1)*l.InputCount)
			if mean == nil {
				mean = vec
			} else {
				mean = autofunc.AddR(mean, vec)
			}
			if variance == nil {
				variance = autofunc.MulR(vec, vec)
			} else {
				variance = autofunc.AddR(variance, autofunc.MulR(vec, vec))
			}
		}
		meanSquared := autofunc.SquareR(mean)
		variance = autofunc.AddR(variance, autofunc.ScaleR(meanSquared, -1))

		negMean := autofunc.RepeatR(autofunc.ScaleR(mean, -1), n)
		invStd := autofunc.RepeatR(autofunc.PowR(autofunc.AddScalerR(variance,
			l.stabilizer()), -0.5), n)
		normalized := autofunc.MulR(autofunc.AddR(in, negMean), invStd)
		scales := autofunc.RepeatR(autofunc.NewRVariable(l.Scales, rv), n)
		biases := autofunc.RepeatR(autofunc.NewRVariable(l.Biases, rv), n)
		return autofunc.AddR(autofunc.MulR(normalized, scales), biases)
	})
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

func (l *Layer) stabilizer() float64 {
	if l.Stabilizer != 0 {
		return l.Stabilizer
	}
	return defaultStabilizer
}
