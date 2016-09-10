// Package batchnorm implements Batch Normalization
// for neural networks.
package batchnorm

import (
	"math"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
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
//
// The cache parameter specifies the maximum number of
// float64 values to cache.
// Caching may use a lot of memory, but it can prevent
// redundant passes through the network.
func BatchNorm(net neuralnet.Network, batch sgd.SampleSet, stabilizer float64, cache int) {
	o := newOutputCache(cache, batch.Len())
	for i, layer := range net {
		if l, ok := layer.(*Layer); ok {
			mean, variance := batchStatistics(net[:i], batch, l.InputCount, o)
			for j, x := range mean {
				l.NegMeans[j] = -x
			}
			for j, x := range variance {
				l.InvStddevs[j] = 1 / math.Sqrt(stabilizer+x)
			}
		}
	}
}

type outputCache struct {
	NetLens []int
	NetOuts []*autofunc.Variable

	Capacity  int
	NumFloats int
}

func newOutputCache(capacity int, batchSize int) *outputCache {
	return &outputCache{
		NetLens:   make([]int, batchSize),
		NetOuts:   make([]*autofunc.Variable, batchSize),
		Capacity:  capacity,
		NumFloats: 0,
	}
}

func (o *outputCache) Eval(idx int, net neuralnet.Network, samples sgd.SampleSet) linalg.Vector {
	cachedVec := o.NetOuts[idx]
	cachedDepth := o.NetLens[idx]
	if cachedVec == nil || cachedDepth > len(net) {
		inVec := samples.GetSample(idx).(neuralnet.VectorSample).Input
		out := net.Apply(&autofunc.Variable{Vector: inVec}).Output()
		o.store(idx, len(net), out)
		return out
	} else if cachedDepth == len(net) {
		return cachedVec.Vector
	}

	endNet := net[cachedDepth:]
	out := endNet.Apply(cachedVec).Output()
	o.store(idx, len(net), out)
	return out
}

func (o *outputCache) store(idx int, netLen int, out linalg.Vector) {
	if netLen == 0 {
		return
	}

	newNum := o.NumFloats + len(out)
	if o.NetOuts[idx] != nil {
		newNum -= len(o.NetOuts[idx].Vector)
	}
	if newNum > o.Capacity {
		return
	}

	o.NetLens[idx] = netLen
	o.NetOuts[idx] = &autofunc.Variable{Vector: out}
	o.NumFloats = newNum
}
