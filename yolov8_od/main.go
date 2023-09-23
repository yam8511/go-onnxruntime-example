package main

import (
	"encoding/json"
	"flag"
	"fmt"
	"image"
	"image/color"
	"math"
	"os"
	"runtime"
	"strings"
	"time"

	ort "github.com/yam8511/go-onnxruntime"

	"gocv.io/x/gocv"
)

func main() {
	dllPath := ""
	if runtime.GOOS == "windows" {
		flag.StringVar(&dllPath, "lib", "onnxruntime.dll", "onnxruntime DLL")
	}
	useGPU := flag.Bool("gpu", true, "inference using CUDA")
	flag.Parse()

	sdk, err := ort.New_ORT_SDK(dllPath)
	if err != nil {
		panic(err)
	}
	defer sdk.Release()

	fmt.Printf("sdk.ORT_API_VERSION(): %v\n", sdk.ORT_API_VERSION())
	fmt.Printf("sdk.GetVersionString(): %v\n", sdk.GetVersionString())

	detect(sdk, *useGPU)
}

// Handler of /detect POST endpoint
// Receives uploaded file with a name "image_file", passes it
// through YOLOv8 object detection network and returns and array
// of bounding boxes.
// Returns a JSON array of objects bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
func detect(sdk *ort.ORT_SDK, useGPU bool) {
	session, err := ort.NewSessionWithONNX(sdk, "./yolov8n.onnx", useGPU)
	if err != nil {
		panic(err)
	}

	for i := 0; i < 5; i++ {
		boxes, scores, classIds := detect_objects_on_image(session)

		b, err := json.Marshal([]any{
			boxes, scores, classIds,
		})
		if err != nil {
			panic(err)
		}
		fmt.Println(string(b))
	}
}

// Function receives an image,
// passes it through YOLOv8 neural network
// and returns an array of detected objects
// and their bounding boxes
// Returns Array of bounding boxes in format [[x1,y1,x2,y2,object_type,probability],..]
func detect_objects_on_image(session *ort.Session) (
	boxes []image.Rectangle,
	scores []float32,
	classIds []int,
) {
	var preP, inferP, postP time.Duration
	now := time.Now()
	input, img_width, img_height := prepare_input()
	preP = time.Since(now)

	now = time.Now()
	output := run_model(session, input)
	inferP = time.Since(now)
	// fmt.Printf("origin ==> %+v\n", output)

	now = time.Now()
	boxes, scores, classIds = process_output(output)
	postP = time.Since(now)

	fmt.Printf(
		"%s pre-process, %s inference, %s post-process, total %s\n",
		preP, inferP, postP,
		preP+inferP+postP,
	)

	img := gocv.IMRead("bus.jpg", gocv.IMReadUnchanged)
	drawBox(&img, float32(img_width)/640.0, float32(img_height)/640.0, boxes, scores, classIds)
	gocv.IMWrite("bus_result.jpg", img)
	return
}

// Function used to convert input image to tensor,
// required as an input to YOLOv8 object detection
// network.
// Returns the input tensor, original image width and height
func prepare_input() ([]float32, int64, int64) {
	img := gocv.IMRead("bus.jpg", gocv.IMReadUnchanged)
	// defer debug.FreeOSMemory()
	defer img.Close()
	imgSize := image.Pt(640, 640)
	img_width, img_height := img.Cols(), img.Rows()
	gocv.Resize(img, &img, imgSize, 0, 0, gocv.InterpolationDefault)
	// size := img.Bounds().Size()
	// img_width, img_height := int64(size.X), int64(size.Y)
	// img = resize.Resize(640, 640, img, resize.Lanczos3)

	ratio := 1.0 / 255
	mean := gocv.NewScalar(0, 0, 0, 0)
	swapRGB := true
	blob := gocv.BlobFromImage(img, ratio, imgSize, mean, swapRGB, false)
	input, err := blob.DataPtrFloat32()
	if err != nil {
		panic(err)
	}
	// fmt.Printf("input: %v\n", input)
	return input, int64(img_width), int64(img_height)
}

// Function used to pass provided input tensor to
// YOLOv8 neural network and return result
// Returns raw output of YOLOv8 network as a single dimension
// array
func run_model(session *ort.Session, input []float32) []float32 {
	inputTensor, err := ort.NewInputTensor(session, "", input)
	if err != nil {
		panic(err)
	}
	defer inputTensor.Destroy()

	outputTensor, err := ort.NewEmptyOutputTensor[float32](session, "")
	if err != nil {
		panic(err)
	}
	defer outputTensor.Destroy()

	err = session.RunDefault(
		[]ort.AnyTensor{inputTensor},
		[]ort.AnyTensor{outputTensor},
	)
	if err != nil {
		panic(err)
	}
	return outputTensor.GetData()
}

// Function used to convert RAW output from YOLOv8 to an array
// of detected objects. Each object contain the bounding box of
// this object, the type of object and the probability
// Returns array of detected objects in a format [[x1,y1,x2,y2,object_type,probability],..]
func process_output(output []float32) (
	boxes []image.Rectangle,
	scores []float32,
	classIds []int,
) {
	boxes = make([]image.Rectangle, 0, 8400)
	scores = make([]float32, 0, 8400)
	classIds = make([]int, 0, 8400)

	// boxes := [][]interface{}{}
	for index := 0; index < 8400; index++ {
		xc := output[index]
		yc := output[8400+index]
		w := output[2*8400+index]
		h := output[3*8400+index]
		class_id, prob := 0, float32(0.0)
		out := []float32{xc, yc, w, h}
		for col := 0; col < 80; col++ {
			out = append(out, output[8400*(col+4)+index])
			if output[8400*(col+4)+index] > prob {
				prob = output[8400*(col+4)+index]
				class_id = col
			}
		}
		// fmt.Println(index, "==>", out)
		if prob < 0.8 {
			continue
		}
		// label := yolo_classes[class_id]
		x1 := xc - w/2
		y1 := yc - h/2
		x2 := xc + w/2
		y2 := yc + h/2
		// boxes = append(boxes, []interface{}{float64(x1), float64(y1), float64(x2), float64(y2), label, prob})
		box := image.Rect(
			int(math.Round(float64(x1))),
			int(math.Round(float64(y1))),
			int(math.Round(float64(x2))),
			int(math.Round(float64(y2))),
		)
		boxes = append(boxes, box)
		scores = append(scores, prob)
		classIds = append(classIds, class_id)
	}

	indices := gocv.NMSBoxes(boxes, scores, 0.8, 0.5)
	tmp_boxes := make([]image.Rectangle, 0, len(indices))
	tmp_scores := make([]float32, 0, len(indices))
	tmp_classIds := make([]int, 0, len(indices))
	for _, idx := range indices {
		tmp_boxes = append(tmp_boxes, boxes[idx])
		tmp_scores = append(tmp_scores, scores[idx])
		tmp_classIds = append(tmp_classIds, classIds[idx])
	}
	boxes = tmp_boxes
	scores = tmp_scores
	classIds = tmp_classIds

	return
}

// Function calculates "Intersection-over-union" coefficient for specified two boxes
// https://pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/.
// Returns Intersection over union ratio as a float number
func iou(box1, box2 []interface{}) float64 {
	return intersection(box1, box2) / union(box1, box2)
}

// Function calculates union area of two boxes
// Returns Area of the boxes union as a float number
func union(box1, box2 []interface{}) float64 {
	box1_x1, box1_y1, box1_x2, box1_y2 := box1[0].(float64), box1[1].(float64), box1[2].(float64), box1[3].(float64)
	box2_x1, box2_y1, box2_x2, box2_y2 := box2[0].(float64), box2[1].(float64), box2[2].(float64), box2[3].(float64)
	box1_area := (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
	box2_area := (box2_x2 - box2_x1) * (box2_y2 - box2_y1)
	return box1_area + box2_area - intersection(box1, box2)
}

// Function calculates intersection area of two boxes
// Returns Area of intersection of the boxes as a float number
func intersection(box1, box2 []interface{}) float64 {
	box1_x1, box1_y1, box1_x2, box1_y2 := box1[0].(float64), box1[1].(float64), box1[2].(float64), box1[3].(float64)
	box2_x1, box2_y1, box2_x2, box2_y2 := box2[0].(float64), box2[1].(float64), box2[2].(float64), box2[3].(float64)
	x1 := math.Max(box1_x1, box2_x1)
	y1 := math.Max(box1_y1, box2_y1)
	x2 := math.Min(box1_x2, box2_x2)
	y2 := math.Min(box1_y2, box2_y2)
	return (x2 - x1) * (y2 - y1)
}

// Array of YOLOv8 class labels
var yolo_classes = []string{
	"person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
	"traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse",
	"sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie",
	"suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove",
	"skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
	"bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
	"cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse",
	"remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book",
	"clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
}

func drawBox(
	img *gocv.Mat, xFactor, yFactor float32,
	boxes []image.Rectangle,
	scores []float32,
	classIds []int,
) {
	b, err := os.ReadFile("names.txt")
	if err != nil {
		panic(err)
	}
	lines := strings.Split(string(b), "\n")
	for i, v := range lines {
		lines[i] = strings.TrimSpace(v)
	}
	for idx, box := range boxes {
		// if idx == 0 {
		// 	continue
		// }
		// box := boxes[idx]
		// fmt.Printf("######################: %v\n", idx)
		// fmt.Printf("x1 = %v, y1 = %v\n", box.Min.X, box.Min.Y)
		// fmt.Printf("x2 = %v, y2 = %v\n", box.Max.X, box.Max.Y)
		// fmt.Printf("class id: %v\n", classIds[idx])
		// box := boxes[idx]
		box = image.Rect(
			int(math.Round(float64(box.Min.X)*float64(xFactor))),
			int(math.Round(float64(box.Min.Y)*float64(yFactor))),
			int(math.Round(float64(box.Max.X)*float64(xFactor))),
			int(math.Round(float64(box.Max.Y)*float64(yFactor))),
		)
		draw_bounding_box(
			img,
			fmt.Sprint(lines[classIds[idx]]),
			scores[idx],
			box,
			color.RGBA{255, 0, 0, 0},
			0, 0, 0,
		)
		fmt.Println("==>", idx, ":", lines[classIds[idx]], scores[idx])
	}
}

func draw_bounding_box(
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
