package main

import (
	"flag"
	"fmt"
	"log"
	"runtime"

	ort "github.com/yam8511/go-onnxruntime"
)

var threshold float64

func main() {
	dllPath := ""
	if runtime.GOOS == "windows" {
		flag.StringVar(&dllPath, "lib", "onnxruntime.dll", "onnxruntime DLL")
	}
	useGPU := flag.Bool("gpu", true, "inference using CUDA")
	input := flag.String("input", "bus.jpg", "inference input image")
	onnxFile := flag.String("onnx", "yolov8n-cls.onnx", "inference onnx model")
	nameFile := flag.String("names", "names_cls.txt", "inference names.txt")
	flag.Float64Var(&threshold, "conf", 0.0, "inference confidence threshold")
	flag.Parse()

	ortSDK, err := ort.New_ORT_SDK(func(opt *ort.OrtSdkOption) {
		opt.Version = ort.ORT_API_VERSION
		opt.WinDLL_Name = dllPath
		opt.LoggingLevel = ort.ORT_LOGGING_LEVEL_WARNING
	})
	if err != nil {
		log.Println("初始化 onnxruntime sdk 失敗: ", err)
		return
	}
	defer ortSDK.Release()

	log.Println("onnxruntime version " + ortSDK.GetVersionString())

	sess, err := NewSession_CLS(ortSDK, *onnxFile, *nameFile, *useGPU)
	if err != nil {
		log.Println("建立分類 Session 失敗: ", err)
		return
	}
	defer sess.release()

	for i := 0; i < 5; i++ {
		img, label, confidence, err := sess.predict_file(*input, float32(threshold))
		if err != nil {
			log.Println("inference failed:", err)
			return
		}
		img.Close()
		fmt.Printf("label: %v, confidence: %v\n", label, confidence)
	}
}
