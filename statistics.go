package batchnorm

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

func batchStatistics(n neuralnet.Network, samples sgd.SampleSet) (mean, variance float64) {
	if len(n) == 0 {
		return batchStatisticsFirst(samples)
	}
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
