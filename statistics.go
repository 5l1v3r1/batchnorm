package batchnorm

import (
	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

func batchStatistics(n neuralnet.Network, samples sgd.SampleSet, outSize int) (mean,
	variance linalg.Vector) {
	mean = make(linalg.Vector, outSize)
	variance = make(linalg.Vector, outSize)
	for i := 0; i < samples.Len(); i++ {
		sample := samples.GetSample(i).(neuralnet.VectorSample)
		inVar := &autofunc.Variable{Vector: sample.Input}
		out := n.Apply(inVar).Output()
		for j, x := range out {
			mean[j] += x
			variance[j] += x * x
		}
	}
	mean.Scale(1 / float64(samples.Len()))
	variance.Scale(1 / float64(samples.Len()))
	for i, x := range mean {
		variance[i] -= x * x
	}
	return
}
