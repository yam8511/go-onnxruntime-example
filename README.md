# Examples

## Package

```shell
go get github.com/yam8511/go-onnxruntime@v1.1.0
```

## YOLOv8 Object Detection

- [YOLOv8](https://docs.ultralytics.com/tasks/detect/)

```shell
go build -v -o run_od ./yolov8_od

# Windows
./run_od.exe -lib your/onnxruntime.dll
# Linux
./run_od
```

## YOLOv8 Classify

- [YOLOv8](https://docs.ultralytics.com/tasks/classify/)

```shell
go build -v -o run_cls ./yolov8_cls

# Windows
./run_cls.exe -lib your/onnxruntime.dll
# Linux
./run_cls
```
