package main

import (
	"fmt"
	"image"
	"os"
	"sort"
	"time"

	"go-onnxruntime-example/pkg/gocv"
	"go-onnxruntime-example/pkg/utils"

	ort "github.com/yam8511/go-onnxruntime"
)

type Session_CLS struct {
	session *ort.Session
	names   []string
}

func NewSession_CLS(ortSDK *ort.ORT_SDK, onnxFile string, useGPU bool) (*Session_CLS, error) {
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

	return &Session_CLS{
		session: sess,
		names:   names,
	}, nil
}

func (sess *Session_CLS) predict_file(inputFile string, threshold float32) (
	gocv.Mat, string, float32, error,
) {
	b, err := os.ReadFile(inputFile)
	if err != nil {
		return gocv.Mat{}, "", 0, err
	}
	img, err := gocv.IMDecode(b, gocv.IMReadColor)
	if err != nil {
		return gocv.Mat{}, "", 0, err
	}
	label, score, err := sess.predict(img, threshold)
	if err != nil {
		img.Close()
	}
	return img, label, score, err
}

func (sess *Session_CLS) predict(img gocv.Mat, threshold float32) (string, float32, error) {
	var preP, inferP, postP time.Duration
	now := time.Now()
	input, err := sess.prepare_input(img.Clone())
	if err != nil {
		return "", 0, err
	}
	preP = time.Since(now)

	now = time.Now()
	output, err := sess.run_model(input)
	if err != nil {
		return "", 0, err
	}
	inferP = time.Since(now)

	now = time.Now()
	label, confidence, err := sess.process_output(output, threshold)
	if err != nil {
		return "", 0, err
	}
	postP = time.Since(now)

	fmt.Printf(
		"%s pre-process, %s inference, %s post-process, total %s\n",
		preP, inferP, postP,
		preP+inferP+postP,
	)

	return label, confidence, nil
}

func (sess *Session_CLS) prepare_input(img gocv.Mat) ([]float32, error) {
	// img := gocv.IMRead(inputFile, gocv.IMReadColor)
	defer img.Close()
	input0, _ := sess.session.Input("images")
	// fmt.Printf("input0: %v\n", input0)
	imgSize := image.Pt(int(input0.Shape[2]), int(input0.Shape[3]))
	gocv.Resize(img, &img, imgSize, 0, 0, gocv.InterpolationDefault)

	ratio := 1.0 / 255
	mean := gocv.NewScalar(0, 0, 0, 0)
	swapRGB := true
	blob := gocv.BlobFromImage(img, ratio, imgSize, mean, swapRGB, false)
	input, err := blob.DataPtrFloat32()
	if err != nil {
		return nil, err
	}

	inputData := make([]float32, len(input))
	copy(inputData, input)
	blob.Close()
	return inputData, nil
}

func (sess *Session_CLS) run_model(input []float32) ([]float32, error) {
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

type ScoreIdx struct {
	index int
	score float32
}

func (sess *Session_CLS) process_output(output []float32, threshold float32) (string, float32, error) {
	if len(output) == 0 {
		return "", 0, nil
	}

	scores := []ScoreIdx{}
	for i, v := range output {
		scores = append(scores, ScoreIdx{index: i, score: v})
	}

	sort.SliceStable(scores, func(i, j int) bool { return scores[i].score > scores[j].score })

	if si := scores[0]; si.score > threshold {
		return sess.names[si.index], si.score, nil
	}

	return "", 0, nil
}

func (sess *Session_CLS) release() { sess.session.Release() }
