package main

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"time"

	"go-onnxruntime-example/pkg/gocv"

	ort "github.com/yam8511/go-onnxruntime"
)

type PoseObject struct {
	Box       image.Rectangle
	Score     float32
	Keypoints [17]Keypoint
}

type Keypoint struct {
	X, Y, Score float32
}

type Session_Pose struct {
	session *ort.Session
}

func NewSession_Pose(ortSDK *ort.ORT_SDK, onnxFile string, useGPU bool) (*Session_Pose, error) {
	sess, err := ort.NewSessionWithONNX(ortSDK, onnxFile, true)
	if err != nil {
		return nil, err
	}

	return &Session_Pose{
		session: sess,
	}, nil
}

func (sess *Session_Pose) predict_file(inputFile string, thresholdPerson, thresholdPose float32) (
	gocv.Mat, []PoseObject, error,
) {
	img := gocv.IMRead(inputFile, gocv.IMReadColor)
	objs, err := sess.predict(img, thresholdPerson, thresholdPose)
	if err != nil {
		img.Close()
	}
	return img, objs, err
}

func (sess *Session_Pose) predict(img gocv.Mat, thresholdPerson, thresholdPose float32) (
	[]PoseObject, error,
) {
	var preP, inferP, postP time.Duration
	now := time.Now()
	input, xFactor, yFactor, err := sess.prepare_input(img.Clone())
	if err != nil {
		return nil, err
	}
	preP = time.Since(now)

	now = time.Now()
	output, err := sess.run_model(input)
	if err != nil {
		return nil, err
	}
	inferP = time.Since(now)

	now = time.Now()
	objs := sess.process_output(output, thresholdPerson, thresholdPose, xFactor, yFactor)
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

func (sess *Session_Pose) prepare_input(img gocv.Mat) ([]float32, float32, float32, error) {
	// img := gocv.IMRead(inputFile, gocv.IMReadColor)
	defer img.Close()
	input0, _ := sess.session.Input("images")
	// fmt.Printf("input0: %v\n", input0)
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

func (sess *Session_Pose) run_model(input []float32) ([]float32, error) {
	inputTensor, err := ort.NewInputTensor(sess.session, "", input)
	if err != nil {
		return nil, err
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyOutputTensor[float32](sess.session, "")
	if err != nil {
		return nil, err
	}
	defer outputTensor.Destroy()

	err = sess.session.RunDefault(
		[]ort.AnyTensor{inputTensor},
		[]ort.AnyTensor{outputTensor},
	)
	if err != nil {
		return nil, err
	}
	return outputTensor.GetData(), nil
}

func (sess *Session_Pose) process_output(output []float32, thresholdPerson, thresholdPose, xFactor, yFactor float32) (
	objs []PoseObject,
) {
	output0, ok := sess.session.Output("output0")
	if !ok {
		return
	}

	size := int(output0.Shape[2]) // 8400
	boxes := make([]image.Rectangle, 0, size)
	scores := make([]float32, 0, size)
	keypoints := make([][17]Keypoint, 0, size)
	for index := 0; index < size; index++ {
		score := output[4*size+index]

		if score < thresholdPerson {
			continue
		}

		kps := [17]Keypoint{}
		for i := 0; i < 17; i++ {
			kp_score := output[(7+i*3)*size+index]
			if kp_score < thresholdPose {
				kps[i] = Keypoint{-1, -1, kp_score}
				continue
			}
			kp_x := output[(5+i*3)*size+index] * xFactor
			kp_y := output[(6+i*3)*size+index] * yFactor
			kps[i] = Keypoint{kp_x, kp_y, kp_score}
		}

		xc := output[0*size+index]
		yc := output[1*size+index]
		w := output[2*size+index]
		h := output[3*size+index]

		x1 := math.Round(float64((xc - w*0.5) * xFactor))
		y1 := math.Round(float64((yc - h*0.5) * yFactor))
		x2 := math.Round(float64((xc + w*0.5) * xFactor))
		y2 := math.Round(float64((yc + h*0.5) * yFactor))

		boxes = append(boxes, image.Rect(int(x1), int(y1), int(x2), int(y2)))
		scores = append(scores, score)
		keypoints = append(keypoints, kps)
	}

	objs = []PoseObject{}
	if len(boxes) == 0 {
		return
	}

	indices := gocv.NMSBoxes(boxes, scores, thresholdPerson, 0.5)
	for _, idx := range indices {
		objs = append(objs, PoseObject{
			Box:       boxes[idx],
			Score:     scores[idx],
			Keypoints: keypoints[idx],
		})
	}
	return
}

func (sess *Session_Pose) release() { sess.session.Release() }

func (sess *Session_Pose) draw(
	img *gocv.Mat,
	objs []PoseObject,
) {
	for _, obj := range objs {
		_color := color.RGBA{200, 200, 200, 0}

		// 畫框框
		gocv.Rectangle(img, obj.Box, _color, 4)

		// 畫肢體
		sess.draw_body(
			img, obj.Keypoints,
			0, 0,
		)
	}
}

func (sess *Session_Pose) draw_body(
	img *gocv.Mat,
	kps [17]Keypoint,
	thickness, radius int,
) {
	if thickness == 0 {
		thickness = 1
	}

	if radius == 0 {
		radius = 3
	}

	// 畫頭部圓圈
	for i := 0; i < 7; i++ {
		kpt := image.Pt(int(kps[i].X), int(kps[i].Y))
		gocv.Circle(img, kpt, radius, color.RGBA{200, 200, 0, 0}, -1)
	}

	for i := 5; i < 13; i++ {
		gocv.Circle(img, image.Pt(int(kps[i].X), int(kps[i].Y)), radius, color.RGBA{200, 0, 200, 0}, -1)
	}
	for i := 11; i < 17; i++ {
		gocv.Circle(img, image.Pt(int(kps[i].X), int(kps[i].Y)), radius, color.RGBA{0, 200, 200, 0}, -1)
	}

	kpset := []struct {
		kp_index []int
		color    color.RGBA
		closed   bool
	}{
		{
			kp_index: []int{5, 3, 1},
			color:    color.RGBA{200, 200, 0, 125},
		},
		{
			kp_index: []int{6, 4, 2},
			color:    color.RGBA{200, 200, 0, 125},
		},
		{
			kp_index: []int{0, 1, 2},
			color:    color.RGBA{200, 200, 0, 125},
			closed:   true,
		},
		{
			kp_index: []int{5, 7, 9},
			color:    color.RGBA{0, 0, 200, 125},
		},
		{
			kp_index: []int{6, 8, 10},
			color:    color.RGBA{0, 0, 200, 125},
		},
		{
			kp_index: []int{11, 13, 15},
			color:    color.RGBA{0, 200, 0, 125},
		},
		{
			kp_index: []int{12, 14, 16},
			color:    color.RGBA{0, 200, 0, 125},
		},
		{
			kp_index: []int{5, 6, 12, 11},
			color:    color.RGBA{255, 0, 0, 125},
			closed:   true,
		},
	}

	for _, kpi := range kpset {
		func() { // 0~6
			pts := gocv.NewPointsVector()
			defer pts.Close()
			pv := gocv.NewPointVector()
			defer pv.Close()
			for _, i := range kpi.kp_index {
				kp := kps[i]
				if kp.X < 0 || kp.Y < 0 {
					continue
				}
				kpt := image.Pt(int(kp.X), int(kp.Y))
				pv.Append(kpt)
			}
			pts.Append(pv)
			if kpi.closed {
				gocv.Polylines(img, pts, true, kpi.color, thickness)
			} else {
				gocv.Polylines(img, pts, false, kpi.color, thickness)
			}
		}()
	}
}
