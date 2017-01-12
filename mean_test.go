package batchnorm

import (
	"math/rand"
	"testing"

	"github.com/unixpickle/autofunc"
	"github.com/unixpickle/num-analysis/linalg"
)

func TestComputeMeans(t *testing.T) {
	input := &autofunc.Variable{
		Vector: make(linalg.Vector, 30),
	}
	for i := range input.Vector {
		input.Vector[i] = rand.NormFloat64()
	}
	actual := computeMeans(input, 6)
	expected := autofunc.Slice(input, 0, 6)
	for i := 6; i < len(input.Vector); i += 6 {
		expected = autofunc.Add(expected, autofunc.Slice(input, i, i+6))
	}
	expected = autofunc.Scale(expected, 0.2)

	testEquivalence(t, actual, expected, []*autofunc.Variable{input})
}

func TestComputeMeanSquares(t *testing.T) {
	input := &autofunc.Variable{
		Vector: make(linalg.Vector, 30),
	}
	for i := range input.Vector {
		input.Vector[i] = rand.NormFloat64()
	}
	actual := computeMeanSquares(input, 6)

	input2 := autofunc.Square(input)
	expected := autofunc.Slice(input2, 0, 6)
	for i := 6; i < len(input.Vector); i += 6 {
		expected = autofunc.Add(expected, autofunc.Slice(input2, i, i+6))
	}
	expected = autofunc.Scale(expected, 0.2)

	testEquivalence(t, actual, expected, []*autofunc.Variable{input})
}

func TestComputeStddev(t *testing.T) {
	input := &autofunc.Variable{
		Vector: make(linalg.Vector, 30),
	}
	for i := range input.Vector {
		input.Vector[i] = rand.NormFloat64()
	}
	mean := computeMeans(input, 6)
	meanSquare := computeMeanSquares(input, 6)
	actual := computeStddev(mean, meanSquare, 0)
	expected := autofunc.Pow(autofunc.Sub(meanSquare, autofunc.Pow(mean, 2)), 0.5)

	testEquivalence(t, actual, expected, []*autofunc.Variable{input})
}
