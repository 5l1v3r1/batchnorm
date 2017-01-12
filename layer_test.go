package batchnorm

import (
	"math"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/autofunc/functest"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestLayerOutput(t *testing.T) {
	l := &Layer{
		InputCount: 10,
		Biases: &autofunc.Variable{
			Vector: linalg.RandVector(10),
		},
		Scales: &autofunc.Variable{
			Vector: linalg.RandVector(10),
		},
		Stabilizer: 1e-7,
	}
	in := &autofunc.Variable{
		Vector: linalg.RandVector(50),
	}
	actual := l.Apply(in).Output()

	mean := in.Vector[:10].Copy()
	meanSq := mean.Copy()
	for i, x := range meanSq {
		meanSq[i] = x * x
	}
	for i := 10; i < 50; i += 10 {
		mean.Add(in.Vector[i : i+10])
		for j, x := range in.Vector[i : i+10] {
			meanSq[j] += x * x
		}
	}
	mean.Scale(1.0 / 5)
	meanSq.Scale(1.0 / 5)

	variance := meanSq.Copy()
	for i, x := range mean {
		variance[i] -= x * x
	}

	normalized := in.Vector.Copy()
	for i := range normalized {
		normalized[i] -= mean[i%10]
		normalized[i] /= math.Sqrt(variance[i%10])
		normalized[i] *= l.Scales.Vector[i%10]
		normalized[i] += l.Biases.Vector[i%10]
	}

	for i, x := range normalized {
		a := actual[i]
		if math.IsNaN(x) || math.IsNaN(a) || math.Abs(a-x) > 1e-5 {
			t.Fatalf("expected %v but got %v", normalized, actual)
		}
	}
}

type layerTestFunc struct {
	l *Layer
	n int
}

func (l *layerTestFunc) Apply(in autofunc.Result) autofunc.Result {
	return l.l.Batch(in, l.n)
}

func (l *layerTestFunc) ApplyR(rv autofunc.RVector, in autofunc.RResult) autofunc.RResult {
	return l.l.BatchR(rv, in, l.n)
}

func TestLayerChecks(t *testing.T) {
	l := &Layer{
		InputCount: 10,
		Biases: &autofunc.Variable{
			Vector: linalg.RandVector(10),
		},
		Scales: &autofunc.Variable{
			Vector: linalg.RandVector(10),
		},
		Stabilizer: 1e-3,
	}
	in := &autofunc.Variable{
		Vector: linalg.RandVector(50),
	}
	vars := []*autofunc.Variable{l.Biases, l.Scales, in}
	rv := autofunc.RVector{}
	for _, v := range vars {
		rv[v] = linalg.RandVector(len(v.Vector))
	}
	checker := functest.RFuncChecker{
		F:     &layerTestFunc{l, 5},
		Input: in,
		Vars:  vars,
		RV:    rv,
	}
	checker.FullCheck(t)
}
