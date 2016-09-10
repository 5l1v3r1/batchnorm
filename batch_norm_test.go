package batchnorm

import (
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
	BatchNorm(net, samples, 1e-7)
	verifyStatistics(t, net, samples)
}

func testNetwork() neuralnet.Network {
	net := neuralnet.Network{
		&neuralnet.DenseLayer{
			InputCount:  16,
			OutputCount: 10,
		},
		NewLayer(10),
		&neuralnet.HyperbolicTangent{},
		&neuralnet.DenseLayer{
			InputCount:  10,
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
			mean, variance := batchStatistics(net[:i+1], samples, l.InputCount)
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
