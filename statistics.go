package batchnorm

import (
	"fmt"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

func batchStatistics(n neuralnet.Network, samples sgd.SampleSet, outSize int) (mean,
	variance linalg.Vector) {
	mean = make(linalg.Vector, outSize)
	variance = make(linalg.Vector, outSize)
	var count int
	for i := 0; i < samples.Len(); i++ {
		sample := samples.GetSample(i).(neuralnet.VectorSample)
		inVar := &autofunc.Variable{Vector: sample.Input}
		out := n.Apply(inVar).Output()
		if len(out)%outSize != 0 {
			panicMsg := fmt.Sprintf("layer %d got size %d (not divisible by %d)",
				len(n), len(out), outSize)
			panic(panicMsg)
		}
		count += len(out) / outSize
		for j, x := range out {
			mean[j%outSize] += x
			variance[j%outSize] += x * x
		}
	}
	mean.Scale(1 / float64(count))
	variance.Scale(1 / float64(count))
	for i, x := range mean {
		variance[i] -= x * x
	}
	return
}
