package batchnorm

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
	"github.com/unixpickle/serializer"
)

func TestLayerOutput(t *testing.T) {
	layer := &Layer{
		InputCount: 2,
		Biases:     &autofunc.Variable{Vector: []float64{-0.39623, 0.43454}},
		Scales:     &autofunc.Variable{Vector: []float64{-0.31301, 0.15552}},
		NegMeans:   []float64{0.90751, -0.84342},
		InvStddevs: []float64{-0.97922, 0.46684},
	}
	for i := 0; i < 10; i++ {
		inVec := []float64{rand.NormFloat64(), rand.NormFloat64()}
		expected := linalg.Vector([]float64{
			(inVec[0]+layer.NegMeans[0])*layer.InvStddevs[0]*layer.Scales.Vector[0] +
				layer.Biases.Vector[0],
			(inVec[1]+layer.NegMeans[1])*layer.InvStddevs[1]*layer.Scales.Vector[1] +
				layer.Biases.Vector[1],
		})
		inVar := &autofunc.Variable{Vector: inVec}
		actual := layer.Apply(inVar).Output()
		if expected.Copy().Scale(-1).Add(actual).MaxAbs() > 1e-6 {
			t.Errorf("Apply(%v) should be %v but it's %v", inVec, expected, actual)
		}
		rv := autofunc.RVector{}
		actual = layer.ApplyR(rv, autofunc.NewRVariable(inVar, rv)).Output()
		if expected.Copy().Scale(-1).Add(actual).MaxAbs() > 1e-6 {
			t.Errorf("ApplyR(%v) should be %v but it's %v", inVec, expected, actual)
		}
	}
}

func TestLayerSerialization(t *testing.T) {
	layer := &Layer{
		InputCount: 2,
		Biases:     &autofunc.Variable{Vector: []float64{-0.39623, 0.43454}},
		Scales:     &autofunc.Variable{Vector: []float64{-0.31301, 0.15552}},
		NegMeans:   []float64{0.90751, -0.84342},
		InvStddevs: []float64{-0.97922, 0.46684},
	}

	data, err := serializer.SerializeWithType(layer)
	if err != nil {
		t.Fatal(err)
	}

	obj, err := serializer.DeserializeWithType(data)
	if err != nil {
		t.Fatal(err)
	}

	outLayer, ok := obj.(*Layer)
	if !ok {
		t.Fatalf("unexpected type: %T", obj)
	}

	paramsExpected := []linalg.Vector{layer.Biases.Vector, layer.Scales.Vector,
		layer.NegMeans, layer.InvStddevs}
	paramsActual := []linalg.Vector{outLayer.Biases.Vector, outLayer.Scales.Vector,
		outLayer.NegMeans, outLayer.InvStddevs}
	for i, expected := range paramsExpected {
		actual := paramsActual[i]
		if actual.Copy().Scale(-1).Add(expected).MaxAbs() > 1e-6 {
			t.Errorf("parameter %d should be %v but it's %v", i, expected, actual)
		}
	}
}
