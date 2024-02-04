package main

import (
	"fmt"
	"image"
	"image/color"
	"math"
	"math/rand"
	"os"
	"strings"
	"time"

	"go-onnxruntime-example/pkg/gocv"
	"go-onnxruntime-example/pkg/utils"

	ort "github.com/yam8511/go-onnxruntime"
)

type DetectObject struct {
	ID    int
	Label string
	Score float32
	Box   image.Rectangle
}

type Session_OD struct {
	session *ort.Session
	names   []string
	colors  []color.RGBA
}

func NewSession_OD(ortSDK *ort.ORT_SDK, onnxFile, namesFile string, useGPU bool) (*Session_OD, error) {
	sess, err := ort.NewSessionWithONNX(ortSDK, onnxFile, useGPU)
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

	colors := []color.RGBA{}
	if len(names) > 0 {
		rng := rand.New(rand.NewSource(time.Now().UnixMilli()))
		for range names {
			colors = append(colors, color.RGBA{uint8(rng.Intn(255)), uint8(rng.Intn(255)), uint8(rng.Intn(255)), 255})
		}
	}

	return &Session_OD{
		session: sess,
		names:   names,
		colors:  colors,
	}, nil
}

func (sess *Session_OD) predict_file(inputFile string, threshold float32) (
	gocv.Mat, []DetectObject, error,
) {
	img := gocv.IMRead(inputFile, gocv.IMReadColor)
	objs, err := sess.predict(img, threshold)
	if err != nil {
		img.Close()
	}
	return img, objs, err
}

func (sess *Session_OD) predict(img gocv.Mat, threshold float32) (
	[]DetectObject, error,
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
	objs := sess.process_output(output, threshold, xFactor, yFactor)
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

func (sess *Session_OD) prepare_input(img gocv.Mat) ([]float32, float32, float32, error) {
	// img := gocv.IMRead(inputFile, gocv.IMReadColor)
	defer img.Close()
	input0, _ := sess.session.Input("images")
	// fmt.Printf("input0: %v\n", input0)
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

func (sess *Session_OD) run_model(input []float32) ([]float32, error) {
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

func (sess *Session_OD) process_output(output []float32, threshold, xFactor, yFactor float32) (
	objs []DetectObject,
) {
	output0, ok := sess.session.Output("output0")
	if !ok {
		return
	}
	// fmt.Printf("output0.Shape: %v\n", output0.Shape)
	size := int(output0.Shape[2])
	// fmt.Printf("size: %v\n", size)
	nameSize := int(output0.Shape[1])
	// fmt.Printf("nameSize: %v\n", nameSize)
	boxes := make([]image.Rectangle, 0, size)
	scores := make([]float32, 0, size)
	classIds := make([]int, 0, size)

	for index := 0; index < size; index++ {
		class_id, prob := 0, float32(0.0)
		for col := 0; col < (nameSize - 4); col++ {
			if output[size*(col+4)+index] > prob {
				prob = output[size*(col+4)+index]
				class_id = col
			}
		}

		if prob < threshold {
			continue
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
		scores = append(scores, prob)
		classIds = append(classIds, class_id)
	}

	objs = []DetectObject{}
	if len(boxes) == 0 {
		return
	}

	indices := gocv.NMSBoxes(boxes, scores, threshold, 0.5)
	for _, idx := range indices {
		objs = append(objs, DetectObject{
			ID:    classIds[idx],
			Label: sess.names[classIds[idx]],
			Score: scores[idx],
			Box:   boxes[idx],
		})
	}

	return
}

func (sess *Session_OD) release() { sess.session.Release() }

func (sess *Session_OD) draw(
	img *gocv.Mat,
	objs []DetectObject,
) {
	for _, obj := range objs {
		utils.DrawBox(
			img,
			obj.Label,
			obj.Score,
			obj.Box,
			sess.colors[obj.ID],
			0, 0, 0,
		)
	}
}
