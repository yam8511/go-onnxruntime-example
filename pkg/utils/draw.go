package utils

import (
	"fmt"
	"image"
	"image/color"

	"go-onnxruntime-example/pkg/gocv"
)

func DrawBox(
	img *gocv.Mat,
	name string,
	confidence float32,
	rect image.Rectangle,
	_color color.RGBA,
	fontScale float64,
	thickness int,
	fontFace gocv.HersheyFont,
) {
	if thickness == 0 {
		thickness = 2
	}
	// 畫框框
	gocv.Rectangle(img, rect, _color, thickness)
	// 畫分類標籤
	DrawLabel(img, name, confidence, rect, _color, fontScale, thickness, fontFace)
}

func DrawLabel(
	img *gocv.Mat,
	name string,
	confidence float32,
	rect image.Rectangle,
	_color color.RGBA,
	fontScale float64,
	thickness int,
	fontFace gocv.HersheyFont,
) {
	if fontScale == 0 {
		fontScale = 0.8
	}
	if thickness == 0 {
		thickness = 2
	}

	if fontFace == 0 {
		fontFace = gocv.FontHersheyComplex
	}
	var label string
	if name == "" {
		label = fmt.Sprintf("%.2f%%", confidence*100)
	} else {
		label = fmt.Sprintf("%s (%.2f%%)", name, confidence*100)
	}
	labelSize := gocv.GetTextSize(label, fontFace, fontScale, thickness)
	padding := 16
	border := padding / 2
	_x1 := rect.Min.X
	_y2 := rect.Min.Y
	_x2 := _x1 + labelSize.X
	_y1 := _y2 - labelSize.Y

	// 畫文字的背景
	gocv.Rectangle(img, image.Rect(_x1, _y1, _x2+padding, _y2+padding), _color, -1)
	gocv.Rectangle(img, image.Rect(_x1+border, _y1+border, _x2+border, _y2+border), color.RGBA{}, -1)
	// 畫文字
	gocv.PutText(img, label, image.Pt(_x1+1, _y2+border/2), fontFace, fontScale, color.RGBA{255, 255, 255, 0}, thickness)
}
