package initnet

import (
	"math"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// BatchNormFirst generates a neuralnet.RescaleLayer to
// normalize input vectors from a sample set of
// neuralnet.VectorSample elements.
func BatchNormFirst(samples sgd.SampleSet) *neuralnet.RescaleLayer {
	mean, variance := batchStatisticsFirst(samples)
	return &neuralnet.RescaleLayer{
		Bias:  -mean,
		Scale: 1 / math.Sqrt(variance),
	}
}

// BatchNorm applies batch normalization to the network
// by inserting a neuralnet.RescaleLayer after every
// layer for which f returns true.
// The resulting network will be statistically normalized
// after every neuralnet.RescaleLayer.
//
// The samples parameter is used as a source for network
// inputs.
// It should contain neuralnet.VectorSample entries.
//
// The f parameter acts as a filter, picking which layers
// to normalize.
// It may make its decision based on the layer as well as
// the layer's index in the original network.
func BatchNorm(network neuralnet.Network, samples sgd.SampleSet,
	f func(i int, l neuralnet.Layer) bool) neuralnet.Network {
	var res neuralnet.Network
	for i, layer := range network {
		res = append(res, layer)
		if !f(i, layer) {
			continue
		}
		mean, variance := batchStatistics(res, samples)
		res = append(res, &neuralnet.RescaleLayer{
			Bias:  -mean,
			Scale: 1 / math.Sqrt(variance),
		})
	}
	return res
}

// ConvOrDense is a filter for BatchNorm that returns true
// for *neuralnet.ConvLayer and *neuralnet.DenseLayer.
func ConvOrDense(i int, l neuralnet.Layer) bool {
	_, ok := l.(*neuralnet.ConvLayer)
	_, ok1 := l.(*neuralnet.DenseLayer)
	return ok || ok1
}

func batchStatistics(n neuralnet.Network, samples sgd.SampleSet) (mean, variance float64) {
	var sum, squareSum float64
	var count int
	for i := 0; i < samples.Len(); i++ {
		sample := samples.GetSample(i).(neuralnet.VectorSample)
		inVar := &autofunc.Variable{Vector: sample.Input}
		out := n.Apply(inVar)
		for _, x := range out.Output() {
			sum += x
			squareSum += x * x
			count++
		}
	}
	mean = sum / float64(count)
	variance = squareSum/float64(count) - mean*mean
	return
}

func batchStatisticsFirst(samples sgd.SampleSet) (mean, variance float64) {
	var sum, squareSum float64
	var count int
	for i := 0; i < samples.Len(); i++ {
		sample := samples.GetSample(i).(neuralnet.VectorSample)
		for _, x := range sample.Input {
			sum += x
			squareSum += x * x
			count++
		}
	}
	mean = sum / float64(count)
	variance = squareSum/float64(count) - mean*mean
	return
}
