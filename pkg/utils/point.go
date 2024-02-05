package utils

import "math"

func NormalizePoint[T float32 | float64](_pt T, imageMax int) int {
	pt := int(math.Round(float64(_pt)))
	if pt < 0 {
		pt = 0
	}
	if pt > imageMax {
		pt = imageMax
	}
	return pt
}
