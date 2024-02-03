package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os/signal"
	"runtime"
	"syscall"

	"go-onnxruntime-example/pkg/gocv"

	ort "github.com/yam8511/go-onnxruntime"
)

var thresholdPerson, thresholdPose float64

func main() {
	dllPath := ""
	if runtime.GOOS == "windows" {
		flag.StringVar(&dllPath, "lib", "onnxruntime.dll", "onnxruntime DLL")
	}
	useGPU := flag.Bool("gpu", true, "inference using CUDA")
	input := flag.String("input", "bus.jpg", "inference input image")
	onnxFile := flag.String("onnx", "yolov8n-pose.onnx", "inference onnx model")
	flag.Float64Var(&thresholdPerson, "conf_person", 0.25, "inference confidence threshold of person")
	flag.Float64Var(&thresholdPose, "conf_pose", 0.5, "inference confidence threshold of pose")
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

	sess, err := NewSession_Pose(ortSDK, *onnxFile, *useGPU)
	if err != nil {
		log.Println("建立姿態偵測 Session 失敗: ", err)
		return
	}
	defer sess.release()

	sig, _ := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	for i := 0; i < 5; i++ {
		select {
		case <-sig.Done():
			return
		default:
		}
		img, objs, err := sess.predict_file(*input, float32(thresholdPerson), float32(thresholdPose))
		if err != nil {
			log.Println("inference failed:", err)
			return
		}
		sess.draw(&img, objs)
		gocv.IMWrite("result_pose.jpg", img)
		img.Close()
		fmt.Printf("detect %d person. and saved to result_pose.jpg\n", len(objs))
	}
}
