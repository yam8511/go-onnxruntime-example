package main

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"strings"
	"time"

	ort "github.com/yam8511/go-onnxruntime"

	"go-onnxruntime-example/gocv"
)

const mask_thresh = 0.5

type SegmentObject struct {
	ID         int
	Label      string
	Confidence float32
	Box        image.Rectangle
	Mask       gocv.Mat
}

type Session_SEG struct {
	session *ort.Session
	names   []string
}

func NewSession_SEG(ortSDK *ort.ORT_SDK, onnxFile, namesFile string, useGPU bool) (*Session_SEG, error) {
	sess, err := ort.NewSessionWithONNX(ortSDK, onnxFile, true)
	if err != nil {
		return nil, err
	}

	b, err := os.ReadFile(namesFile)
	if err != nil {
		sess.Release()
		return nil, err
	}

	names := []string{}
	lines := strings.Split(string(b), "\n")
	for _, v := range lines {
		v = strings.TrimSpace(v)
		if v == "" {
			continue
		}
		names = append(names, v)
	}

	return &Session_SEG{
		session: sess,
		names:   names,
	}, nil
}

func (sess *Session_SEG) predict(inputFile string, threshold float32) (
	[]SegmentObject, error,
) {
	img := gocv.IMRead(inputFile, gocv.IMReadColor)
	defer img.Close()

	var preP, inferP, postP time.Duration
	now := time.Now()
	input, xFactor, yFactor, err := sess.prepare_input(img.Clone())
	if err != nil {
		return nil, err
	}
	preP = time.Since(now)

	now = time.Now()
	output0, output1, err := sess.run_model(input)
	_ = output1
	if err != nil {
		return nil, err
	}
	inferP = time.Since(now)

	fmt.Printf(
		"%s pre-process, %s inference, total %s\n",
		preP, inferP,
		preP+inferP+postP,
	)

	fmt.Println("output0 shape = ", output0.Size()) // [116 8400]
	fmt.Println("output1 shape = ", output1.Size()) // [32 160 160]
	// return []SegmentObject{}, nil

	now = time.Now()
	// ptr := gocv.DecodeOutput_Segment(img, *output0, *output1, img.Cols(), img.Rows())
	// _ = ptr
	// _ = xFactor
	// _ = yFactor
	objs, err := sess.process_output(output0, output1, &img, xFactor, yFactor, threshold)
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

	sess.drawBox(&img, objs)
	gocv.IMWrite("result.jpg", img)

	return objs, nil
}

func (sess *Session_SEG) prepare_input(img gocv.Mat) ([]float32, float32, float32, error) {
	// img := gocv.IMRead(inputFile, gocv.IMReadColor)
	defer img.Close()
	input0, _ := sess.session.Input("images")
	imgSize := image.Pt(int(input0.Shape[2]), int(input0.Shape[3]))
	img_width, img_height := img.Cols(), img.Rows()
	gocv.Resize(img, &img, imgSize, 0, 0, gocv.InterpolationDefault)

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
	*gocv.Mat, *gocv.Mat, error,
) {
	inputTensor, err := ort.NewInputTensor(sess.session, "", input)
	if err != nil {
		return nil, nil, err
	}
	defer inputTensor.Destroy()

	output0Tensor, err := ort.NewEmptyOutputTensor[float32](sess.session, "output0")
	if err != nil {
		return nil, nil, err
	}
	defer output0Tensor.Destroy()
	output1Tensor, err := ort.NewEmptyOutputTensor[float32](sess.session, "output1")
	if err != nil {
		return nil, nil, err
	}
	defer output1Tensor.Destroy()

	err = sess.session.RunDefault(
		[]ort.AnyTensor{inputTensor},
		[]ort.AnyTensor{output0Tensor, output1Tensor},
	)
	if err != nil {
		return nil, nil, err
	}

	ptr0, err := output0Tensor.GetTensorMutableData()
	if err != nil {
		return nil, nil, err
	}

	ptr1, err := output1Tensor.GetTensorMutableData()
	if err != nil {
		return nil, nil, err
	}

	sizes0 := output0Tensor.GetShape().Sizes()
	mat0 := gocv.NewMatWithSizeAndPtr([]int{sizes0[1], sizes0[2]}, gocv.MatTypeCV32F, ptr0)

	sizes1 := output1Tensor.GetShape().Sizes()
	mat1 := gocv.NewMatWithSizeAndPtr([]int{1, sizes1[1], sizes1[2], sizes1[3]}, gocv.MatTypeCV32F, ptr1)

	return &mat0, &mat1, nil
}

func (sess *Session_SEG) process_output(output, output1, img *gocv.Mat, xFactor, yFactor, accu_thresh float32) (
	objs []SegmentObject, err error,
) {
	objs = []SegmentObject{}
	sizes_0 := output.Size()
	sizes_1 := output1.Size()

	fmt.Printf("sizes: %v\n", sizes_0)
	fmt.Printf("sizes: %v\n", sizes_1)
	rows := sizes_0[1] // 8400
	// cols := output.Size()[0] // 116
	// nameSize := sizes[0]
	nameSize := sizes_0[0] - sizes_1[1]
	fmt.Printf("nameSize: %v\n", nameSize)
	boxes := make([]image.Rectangle, 0, rows)
	scores := make([]float32, 0, rows)
	classIds := make([]int, 0, rows)
	// maskPredicts := make([]gocv.Mat, 0, rows)

	for index := 0; index < rows; index++ {
		row := output.ColRange(index, index+1)
		row = row.T()

		_, maxScore, _, maxLoc := gocv.MinMaxLoc(row.ColRange(4, nameSize))
		// mask := row.ColRange(nameSize, row.Cols())

		xc := row.GetFloatAt(0, 0)
		yc := row.GetFloatAt(0, 1)
		w := row.GetFloatAt(0, 2)
		h := row.GetFloatAt(0, 3)

		x1 := (xc - w*0.5) * xFactor
		y1 := (yc - h*0.5) * yFactor
		x2 := (xc + w*0.5) * xFactor
		y2 := (yc + h*0.5) * yFactor

		boxes = append(boxes, image.Rect(int(x1), int(y1), int(x2), int(y2)))
		scores = append(scores, maxScore)
		classIds = append(classIds, maxLoc.X)
		// maskPredicts = append(maskPredicts, mask)
	}

	if len(boxes) == 0 {
		return
	}

	indices := gocv.NMSBoxes(boxes, scores, accu_thresh, 0.5)

	// raw_width := img.Cols()
	// raw_height := img.Rows()
	for _, idx := range indices {
		// boxes[idx] = boxes[idx].Intersect(image.Rect(0, 0, raw_width, raw_height))
		obj := SegmentObject{
			ID:         classIds[idx],
			Label:      sess.names[classIds[idx]],
			Confidence: scores[idx],
			Box:        boxes[idx],
		}
		// mask := maskPredicts[idx]
		// mask = mask.T()
		// sess.get_mask(output1, img, &mask, boxes[idx])
		// obj.Mask = mask
		objs = append(objs, obj)
	}

	return
}

func (sess *Session_SEG) get_mask(output1, img, maskInfo *gocv.Mat, box image.Rectangle) gocv.Mat {
	sizes := output1.Size()
	fmt.Printf("squeeze mask_output: %v\n", sizes)
	sizeInfo := maskInfo.Size()
	fmt.Printf("遮罩形狀: %v\n", sizeInfo)
	seg_ch := sizes[0]
	seg_h := sizes[1]
	seg_w := sizes[2]
	r_x := int(math.Floor(float64(box.Min.X) / float64(box.Dx()) * float64(seg_w)))
	r_y := int(math.Floor(float64(box.Min.Y) / float64(box.Dy()) * float64(seg_h)))
	r_w := int(math.Ceil(float64(box.Min.X+box.Dx())/float64(img.Cols())*float64(seg_w))) - r_x
	r_h := int(math.Ceil(float64(box.Min.Y+box.Dy())/float64(img.Rows())*float64(seg_w))) - r_y
	r_w = int(math.Max(float64(r_w), 1))
	r_h = int(math.Max(float64(r_h), 1))
	if r_x+r_w > seg_w {
		if seg_w-r_x > 0 {
			r_w = seg_w - r_x
		} else {
			r_x -= 1
		}
	}
	if r_y+r_h > seg_h {
		if seg_h-r_y > 0 {
			r_h = seg_h - r_y
		} else {
			r_y -= 1
		}
	}

	roi_rangs := gocv.NewRangeVector()
	roi_rangs.Append(gocv.NewRangeFrom(0, 1))
	roi_rangs.Append(gocv.NewRangeFromAll())
	roi_rangs.Append(gocv.NewRangeFrom(r_y, r_h+r_y))
	roi_rangs.Append(gocv.NewRangeFrom(r_x, r_w+r_x))
	fmt.Printf("roi_rangs.Size(): %v\n", roi_rangs.Size()) // 4
	temp_mask := output1.Ranges(roi_rangs)
	temp_mask = temp_mask.Clone()
	fmt.Printf("temp_mask.Size(): %v\n", temp_mask.Size()) // 1 160 1
	fmt.Printf("seg_ch: %v\n", seg_ch)
	fmt.Printf("r_w: %v\n", r_w)
	fmt.Printf("r_h: %v\n", r_h)
	fmt.Println(" ====> ", r_w*r_h)
	protos := temp_mask.ReshapeWithSize(0, []int{seg_ch, r_w * r_h})
	matmul_res := gocv.NewMat()
	gocv.Multiply(*maskInfo, protos, &matmul_res)
	matmul_res = matmul_res.T()
	masks_feature := matmul_res.ReshapeWithSize(1, []int{r_h, r_w})

	dest := gocv.NewMat()
	gocv.Exp(masks_feature.Negative(), &dest) // sigmoid
	dest = dest.OpAdd_Float(1.0)
	dest = dest.OpDivByNum_Float(1.0)

	left := int(math.Floor(float64(img.Cols() * r_x / seg_w)))
	top := int(math.Floor(float64(img.Rows() * r_y / seg_h)))
	width := int(math.Ceil(float64(img.Cols() * r_w / seg_w)))
	height := int(math.Ceil(float64(img.Rows() * r_h / seg_h)))

	mask := gocv.NewMat()
	gocv.Resize(dest, &mask, image.Pt(width, height), 0, 0, gocv.InterpolationLinear)
	box.Min.X -= left
	box.Min.Y -= top
	mask2 := mask.Region(box)
	mast_out := mask2.OpGreat_Float(mask_thresh)
	return mast_out
}

func (sess *Session_SEG) release() { sess.session.Release() }

func (sess *Session_SEG) drawBox(
	img *gocv.Mat, objs []SegmentObject,
) {
	// mask := img.Clone()
	for _, v := range objs {
		gocv.Rectangle(img, v.Box, color.RGBA{255, 0, 0, 0}, 3)
		// if v.Mask.Rows() > 0 && v.Mask.Cols() > 0 {
		// 	m := mask.Region(v.Box)
		// 	m.SetToWithMat(gocv.NewScalar(255, 0, 0, 0), v.Mask)
		// }
		label := fmt.Sprintf("%s:%.2f", v.Label, v.Confidence)
		gocv.PutText(img, label, v.Box.Min, gocv.FontHersheySimplex, 2, color.RGBA{255, 0, 0, 0}, 4)
	}
	// gocv.AddWeighted(*img, 0.6, mask, 0.4, 0, img) // add mask to src
}

func (sess *Session_SEG) draw_bounding_box(
	img *gocv.Mat,
	name string,
	confidence float32,
	rect image.Rectangle,
	_color color.RGBA,
	fontScale float64,
	thickness int,
	fontFace gocv.HersheyFont,
	// mask: "np.ndarray | None" = None,
) {
	if fontScale == 0 {
		fontScale = 1
	}
	if thickness == 0 {
		thickness = 3
	}

	if fontFace == 0 {
		fontFace = gocv.FontHersheyComplex
	}
	label := fmt.Sprintf("%s (%.2f)%%", name, confidence*100)
	labelSize := gocv.GetTextSize(label, fontFace, fontScale, thickness)
	_x1 := rect.Min.X
	_y1 := rect.Min.Y
	_x2 := _x1 + labelSize.X
	_y2 := _y1 - labelSize.Y
	// 畫框框
	gocv.Rectangle(img, rect, _color, thickness)
	// 畫文字的背景
	gocv.RectangleWithParams(img, image.Rect(_x1, _y1, _x2, _y2), _color, thickness, gocv.Filled, 0)
	// 畫文字
	gocv.PutText(img, label, rect.Min, fontFace, fontScale, color.RGBA{255, 255, 255, 0}, thickness)
}
