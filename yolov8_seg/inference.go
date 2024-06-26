package main

import (
	"fmt"
	"image"
	"image/color"
	"math/rand"
	"os"
	"time"

	"go-onnxruntime-example/pkg/gocv"
	"go-onnxruntime-example/pkg/utils"

	ort "github.com/yam8511/go-onnxruntime"
)

const mask_thresh = 0.5

type SegmentObject struct {
	ID    int
	Label string
	Score float32
	Box   image.Rectangle
	Mask  []image.Point
}

type Session_SEG struct {
	session *ort.Session
	names   []string
	colors  []color.RGBA
}

func NewSession_SEG(ortSDK *ort.ORT_SDK, onnxFile string, useGPU bool) (*Session_SEG, error) {
	sess, err := ort.NewSessionWithONNX(ortSDK, onnxFile, useGPU)
	if err != nil {
		return nil, err
	}

	_names, err := sess.Metadata("names")
	if err != nil {
		sess.Release()
		return nil, err
	}
	names := utils.MetadataToNames(_names)

	colors := []color.RGBA{}
	if len(names) > 0 {
		rng := rand.New(rand.NewSource(time.Now().UnixMilli()))
		for range names {
			colors = append(colors, color.RGBA{uint8(rng.Intn(255)), uint8(rng.Intn(255)), uint8(rng.Intn(255)), 255})
		}
	}

	return &Session_SEG{
		session: sess,
		names:   names,
		colors:  colors,
	}, nil
}

func (sess *Session_SEG) predict_file(inputFile string, threshold float32) (
	gocv.Mat, []SegmentObject, error,
) {
	b, err := os.ReadFile(inputFile)
	if err != nil {
		return gocv.Mat{}, nil, err
	}
	img, err := gocv.IMDecode(b, gocv.IMReadColor)
	if err != nil {
		return gocv.Mat{}, nil, err
	}
	objs, err := sess.predict(img, threshold)
	if err != nil {
		img.Close()
	}
	return img, objs, err
}

func (sess *Session_SEG) predict(img gocv.Mat, threshold float32) (
	[]SegmentObject, error,
) {
	var preP, inferP, postP time.Duration
	now := time.Now()
	input, xFactor, yFactor, err := sess.prepare_input(img.Clone())
	if err != nil {
		return nil, err
	}
	preP = time.Since(now)

	now = time.Now()
	outputs, err := sess.run_model(input)
	if err != nil {
		return nil, err
	}
	inferP = time.Since(now)
	defer func() {
		for _, output := range outputs {
			output.Destroy()
		}
	}()

	ptr0, err := outputs[0].GetTensorMutableData()
	if err != nil {
		return nil, err
	}

	ptr1, err := outputs[1].GetTensorMutableData()
	if err != nil {
		return nil, err
	}

	sizes0 := outputs[0].GetShape().Sizes()
	output0 := gocv.NewMatWithSizesFromPtr([]int{sizes0[1], sizes0[2]}, gocv.MatTypeCV32F, ptr0)

	sizes1 := outputs[1].GetShape().Sizes()
	output1 := gocv.NewMatWithSizesFromPtr([]int{sizes1[1], sizes1[2], sizes1[3]}, gocv.MatTypeCV32F, ptr1)

	now = time.Now()
	objs, err := sess.process_output(&output0, &output1, &img, xFactor, yFactor, threshold)
	if err != nil {
		return nil, err
	}
	postP = time.Since(now)

	fmt.Printf(
		"%s pre-process, %s inference, %s post-process, total %s\n",
		preP, inferP, postP,
		preP+inferP+postP,
	)

	if len(objs) == 0 {
		return objs, nil
	}

	return objs, nil
}

func (sess *Session_SEG) prepare_input(img gocv.Mat) ([]float32, float32, float32, error) {
	// img := gocv.IMRead(inputFile, gocv.IMReadColor)
	defer img.Close()
	input0, _ := sess.session.Input("images")
	imgSize := image.Pt(int(input0.Shape[2]), int(input0.Shape[3]))
	img_width, img_height := img.Cols(), img.Rows()

	ratio := 1.0 / 255
	mean := gocv.NewScalar(0, 0, 0, 0)
	swapRGB := true
	blob := gocv.BlobFromImage(img, ratio, imgSize, mean, swapRGB, false)
	input, err := blob.DataPtrFloat32()
	if err != nil {
		return nil, 0, 0, err
	}
	inputData := make([]float32, len(input))
	copy(inputData, input)
	blob.Close()
	return inputData,
		float32(img_width) / float32(imgSize.X),
		float32(img_height) / float32(imgSize.Y),
		nil
}

func (sess *Session_SEG) run_model(input []float32) (
	[2]*ort.Tensor[float32], error,
) {
	ret := [2]*ort.Tensor[float32]{}
	inputTensor, err := ort.NewInputTensor(sess.session, "", input)
	if err != nil {
		return ret, err
	}
	defer inputTensor.Destroy()

	output0Tensor, err := ort.NewEmptyOutputTensor[float32](sess.session, "output0")
	if err != nil {
		return ret, err
	}
	output1Tensor, err := ort.NewEmptyOutputTensor[float32](sess.session, "output1")
	if err != nil {
		output0Tensor.Destroy()
		return ret, err
	}

	err = sess.session.RunDefault(
		[]ort.AnyTensor{inputTensor},
		[]ort.AnyTensor{output0Tensor, output1Tensor},
	)
	if err != nil {
		output0Tensor.Destroy()
		output1Tensor.Destroy()
		return ret, err
	}

	ret[0] = output0Tensor
	ret[1] = output1Tensor

	return ret, nil
}

func (sess *Session_SEG) process_output(_output0, output1, img *gocv.Mat, xFactor, yFactor, accu_thresh float32) (
	objs []SegmentObject, err error,
) {
	output0 := _output0.T() // [116 8400] => [8400 116]
	_output0.Close()
	defer func() {
		output0.Close()
		output1.Close()
	}()

	// fmt.Println("output0 shape = ", output0.Size())  // [8400 116]
	// fmt.Println("output1 shape = ", _output1.Size()) // [32 160 160]

	objs = []SegmentObject{}
	sizes_0 := output0.Size() // [8400 116]
	sizes_1 := output1.Size() // [32 160 160]

	rows := sizes_0[0]               // 8400
	totalSize := sizes_0[1]          // 116
	maskSize := sizes_1[0]           // 32
	maskHeight := sizes_1[1]         // 160
	nameSize := totalSize - maskSize // 116 - 32
	imageWidth := img.Cols()
	imageHeight := img.Rows()

	boxes := make([]image.Rectangle, 0, rows)
	scores := make([]float32, 0, rows)
	classIds := make([]int, 0, rows)
	originIdx := make([]int, 0, rows)

	for index := 0; index < rows; index++ {
		func() {
			row := output0.RowRange(index, index+1)
			defer row.Close()

			classes := row.ColRange(4, nameSize)
			defer classes.Close()

			_, maxScore, _, maxLoc := gocv.MinMaxLoc(classes)
			if maxScore < accu_thresh {
				return
			}
			// mask := row.ColRange(nameSize, row.Cols())

			xc := row.GetFloatAt(0, 0)
			yc := row.GetFloatAt(0, 1)
			w := row.GetFloatAt(0, 2)
			h := row.GetFloatAt(0, 3)

			x1 := utils.NormalizePoint((xc-w*0.5)*xFactor, imageWidth)
			y1 := utils.NormalizePoint((yc-h*0.5)*yFactor, imageHeight)
			x2 := utils.NormalizePoint((xc+w*0.5)*xFactor, imageWidth)
			y2 := utils.NormalizePoint((yc+h*0.5)*yFactor, imageHeight)

			boxes = append(boxes, image.Rect(x1, y1, x2, y2))
			scores = append(scores, maxScore)
			classIds = append(classIds, maxLoc.X)
			originIdx = append(originIdx, index)
		}()
	}

	if len(boxes) == 0 {
		return
	}

	indices := gocv.NMSBoxes(boxes, scores, accu_thresh, 0.5)

	output1_reshape := output1.Reshape(output1.Channels(), maskSize) // [32 160 160] => [32 25600]
	defer output1_reshape.Close()
	for _, idx := range indices {
		originIndex := originIdx[idx]
		box := boxes[idx]
		func() {
			row := output0.RowRange(originIndex, originIndex+1) // [8400 116] => [1 116]
			col := row.ColRange(nameSize, totalSize)            // [1 116] => [1 32]
			row.Close()

			mask := col.MultiplyMatrix(output1_reshape)
			col.Close()

			mask_reshape := mask.Reshape(mask.Channels(), maskHeight)
			mask.Close()

			gocv.Resize(mask_reshape, &mask_reshape, image.Pt(imageWidth, imageHeight), 0, 0, gocv.InterpolationDefault)

			mask_region := mask_reshape.Region(box)
			mask_reshape.Close()
			defer mask_region.Close()

			gocv.Threshold(mask_region, &mask_region, 1, 255, gocv.ThresholdBinary)
			mask_region.ConvertTo(&mask_region, gocv.MatTypeCV8U)
			// if !gocv.IMWrite(fmt.Sprintf("mask_%d.jpg", idx), mask_region) {
			// 	fmt.Println("存圖失敗")
			// }

			pts_vec := gocv.FindContours(mask_region, gocv.RetrievalExternal, gocv.ChainApproxSimple)
			defer pts_vec.Close()

			maxAreaIdx := -1
			maxArea := -1.0
			for i := 0; i < pts_vec.Size(); i++ { // 找出最大面積
				contour := pts_vec.At(i)
				area := gocv.ContourArea(contour)
				// fmt.Println(idx, "面積 ==> ", area, "|", i)
				if area > maxArea {
					maxAreaIdx = i
					maxArea = area
				}
			}

			// mask_draw := mask_region.Clone()
			// defer mask_draw.Close()
			// gocv.CvtColor(mask_draw, &mask_draw, gocv.ColorGrayToBGR)

			pts := []image.Point{}
			if maxAreaIdx >= 0 {
				contour := pts_vec.At(maxAreaIdx)
				pts = contour.ToPoints()
				// gocv.DrawContours(&mask_draw, pts_vec, maxAreaIdx, color.RGBA{R: 255}, 4)
				// fmt.Println(idx, "最終選擇 ", maxAreaIdx, " ==> ", maxArea)
			}

			// if !gocv.IMWrite(fmt.Sprintf("mask_draw_%d.jpg", idx), mask_draw) {
			// 	fmt.Println("存圖失敗")
			// }

			for i := range pts {
				pts[i].X += box.Min.X
				pts[i].Y += box.Min.Y
			}

			// return pts
			obj := SegmentObject{
				ID:    classIds[idx],
				Label: sess.names[classIds[idx]],
				Score: scores[idx],
				Box:   box,
				Mask:  pts,
			}
			objs = append(objs, obj)
		}()
	}

	return
}

func (sess *Session_SEG) release() { sess.session.Release() }

func (sess *Session_SEG) draw(
	img *gocv.Mat, objs []SegmentObject,
) {
	for _, obj := range objs {
		_color := sess.colors[obj.ID]
		pt := gocv.NewPointVectorFromPoints(obj.Mask)
		pts := gocv.NewPointsVector()
		pts.Append(pt)
		gocv.Polylines(img, pts, true, _color, 8)
		pts.Close()
	}

	for _, obj := range objs {
		_color := sess.colors[obj.ID]
		utils.DrawLabel(
			img,
			obj.Label,
			obj.Score,
			obj.Box,
			_color,
			0, 0, 0,
		)
	}
}
