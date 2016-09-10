package batchnorm

import (
	"fmt"
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/sgd"
	"github.com/unixpickle/weakai/neuralnet"
)

func TestBatchNorm(t *testing.T) {
	net := testNetwork()
	samples := testSamples()
	for _, cache := range []int{0, 11 * 30, 11 * 51, 10 * 50, 13 * 50, 13 * 51} {
		name := fmt.Sprintf("Cache%d", cache)
		t.Run(name, func(t *testing.T) {
			BatchNorm(net, samples, 1e-7, cache)
			verifyStatistics(t, net, samples)
		})
	}
}

func BenchmarkBatchNorm(b *testing.B) {
	net := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  1000,
			OutputCount: 1000,
		},
		NewLayer(1000),
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  1000,
			OutputCount: 50,
		},
		NewLayer(50),
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  50,
			OutputCount: 1000,
		},
		NewLayer(1000),
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  1000,
			OutputCount: 300,
		},
		NewLayer(300),
		&neuralnet.HyperbolicTangent{},
	}
	net.Randomize()

	var samples sgd.SliceSampleSet
	for i := 0; i < 5; i++ {
		inVec := make(linalg.Vector, 1000)
		for j := range inVec {
			inVec[j] = rand.NormFloat64()
		}
		samples = append(samples, neuralnet.VectorSample{
			Input: inVec,
		})
	}

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		BatchNorm(net, samples, 1e-4, 1000*5)
	}
}

func testNetwork() neuralnet.Network {
	net := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  16,
			OutputCount: 11,
		},
		NewLayer(11),
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  11,
			OutputCount: 13,
		},
		NewLayer(13),
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  13,
			OutputCount: 10,
		},
		NewLayer(5),
	}
	net.Randomize()
	return net
}

func testSamples() sgd.SampleSet {
	var samples sgd.SliceSampleSet
	for i := 0; i < 50; i++ {
		inVec := make(linalg.Vector, 16)
		for j := range inVec {
			inVec[j] = rand.NormFloat64()
		}
		outVec := make(linalg.Vector, 5)
		for j := range outVec {
			outVec[j] = rand.NormFloat64()
		}
		samples = append(samples, neuralnet.VectorSample{
			Input:  inVec,
			Output: outVec,
		})
	}
	return samples
}

func verifyStatistics(t *testing.T, net neuralnet.Network, samples sgd.SampleSet) {
	for i, layer := range net {
		if l, ok := layer.(*Layer); ok {
			o := newOutputCache(0, samples.Len())
			mean, variance := batchStatistics(net[:i+1], samples, l.InputCount, o)
			for _, x := range mean {
				if math.Abs(x) > 1e-5 {
					t.Errorf("bad mean: %f", x)
					return
				}
			}
			for _, x := range variance {
				if math.Abs(x-1) > 1e-5 {
					t.Errorf("bad variance: %f", x)
					return
				}
			}
		}
	}
}
