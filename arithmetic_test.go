package batchnorm

import (
	"math"
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestAddMul(t *testing.T) {
	input := &autofunc.Variable{
		Vector: make(linalg.Vector, 30),
	}
	bias := &autofunc.Variable{
		Vector: make(linalg.Vector, 6),
	}
	scale := &autofunc.Variable{
		Vector: make(linalg.Vector, 6),
	}
	for i := range scale.Vector {
		scale.Vector[i] = rand.NormFloat64()
		bias.Vector[i] = rand.NormFloat64()
	}
	for i := range input.Vector {
		input.Vector[i] = rand.NormFloat64()
	}

	actual := addMul(input, bias, scale, 5)
	expected := autofunc.Mul(autofunc.Add(input, autofunc.Repeat(bias, 5)),
		autofunc.Repeat(scale, 5))

	testEquivalence(t, actual, expected, []*autofunc.Variable{input, bias, scale})
}

func testEquivalence(t *testing.T, actual, expected autofunc.Result, params []*autofunc.Variable) {
	t.Run("Forward", func(t *testing.T) {
		for i, a := range actual.Output() {
			x := expected.Output()[i]
			if math.Abs(a-x) > 1e-5 {
				t.Fatalf("expected %v but got %v", expected, actual)
			}
		}
	})
	t.Run("Backward", func(t *testing.T) {
		actualGrad := autofunc.NewGradient(params)
		expectedGrad := autofunc.NewGradient(params)

		upstream := make(linalg.Vector, len(actual.Output()))
		for i := range upstream {
			upstream[i] = rand.NormFloat64()
		}

		actual.PropagateGradient(upstream.Copy(), actualGrad)
		expected.PropagateGradient(upstream, expectedGrad)

		for i, variable := range params {
			actualVec := actualGrad[variable]
			expectedVec := expectedGrad[variable]
			diff := actualVec.Copy().Scale(-1).Add(expectedVec).MaxAbs()
			if diff > 1e-5 {
				t.Errorf("bad gradient for variable %d", i)
			}
		}
	})
}
