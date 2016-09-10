// Package batchnorm implements Batch Normalization
// for neural networks.
package batchnorm

import (
	"math"

	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

// BatchNorm updates the Layer instances in the network
// based on statistics computed on the given batch.
//
// The stabilizer parameter should be a small, non-zero
// number.
// The stabilizer is added to each variance for numerical
// stability when a variance is near 0.
func BatchNorm(net neuralnet.Network, batch sgd.SampleSet, stabilizer float64) {
	for i, layer := range net {
		if l, ok := layer.(*Layer); ok {
			mean, variance := batchStatistics(net[:i], batch, l.InputCount)
			for i, x := range mean {
				l.NegMeans[i] = -x
			}
			for i, x := range variance {
				l.InvStddevs[i] = 1 / math.Sqrt(stabilizer+x)
			}
		}
	}
}
