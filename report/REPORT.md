# iw05 Obeject Detection with Bounding Box

> 杨茂琛 191180164

运用CoreML开发一个利用TinyYOLO进行目标检测的iOS App

演示视频：https://www.bilibili.com/video/BV18L411j7DR

数据集：https://box.nju.edu.cn/f/0fde100e73b2405abe1e/?dl=1

## 模型训练

使用的是同第4次作业的模型。之前我还奇怪训练集为什么还需要有csv文件，不是有图片就够了吗？

csv文件及其中的标识object位置的左边将在模型训练中用到

补全函数`load_images_with_annotations`以构造一个完整的训练集：

```python
def load_images_with_annotations(images_dir, annotations_file):
    # Load the images into a Turi SFrame.
    data = tc.image_analysis.load_images(images_dir, with_path=True)
    
    # Load the annotations CSV file into a Pandas dataframe.
    csv = pd.DataFrame(pd.read_csv(annotations_file))

    # Loop through all the images and match these to the annotations from the
    # CSV file, if annotations are available for the image.
    all_annotations = []
    for i, item in enumerate(data):
        # Grab image info from the SFrame.
        img_path = item["path"]
        img_width = item["image"].width
        img_height = item["image"].height

        # Find the corresponding row(s) in the CSV's dataframe.
        image_id = os.path.basename(img_path)[:-4]
        rows = csv[csv["image_id"] == image_id]

        # Turi expects a list for every image that contains a dictionary for
        # every bounding box that we have an annotation for.
        img_annotations = []
        # The CSV file stores the coordinate as numbers between 0 and 1,
        # but Turi wants pixel coordinates in the image.
        for i in rows.itertuples():
            img_annotations.append({"coordinates": {"height": (i[5] - i[4]) * img_height, 
                                                    "width": (i[3] - i[2]) * img_width, 
                                                    "x": (i[3] + i[2]) / 2 * img_width, 
                                                    "y": (i[5] + i[4]) / 2 * img_height}, 
                                    "label": i[6]})
        # A bounding box in Turi is given by a center coordinate and the
        # width and height, we have them as the four corners of the box.

            
            # img_annotations.append({"coordinates": {"height": height, 
            #                                         "width": width, 
            #                                         "x": x, 
            #                                         "y": y}, 
            #                         "label": class_name})

        # If there were no annotations for this image, then append a None
        # so that we can filter out those images from the SFrame.
        if len(img_annotations) > 0:
            all_annotations.append(img_annotations)
        else:
            all_annotations.append(None)

    data["annotations"] = tc.SArray(data=all_annotations, dtype=list)
    return data.dropna()
```

csv文件中存放的是归一化的位置，但Turi Create使用的是像素坐标

Turi Create描述位置的方法为“中心位置+大小”，根据需求进行一定运算即可



生成的训练集长下面这个样子：

```json
{'path': '/Users/nju/Desktop/iw05-syhien/snacks/train/apple/007a0bec00a90a66.jpg',
 'image': Height: 341px
 Width: 256px
 Channels: 3,
 'annotations': [{'coordinates': {'height': 94.81198099999997,
    'width': 111.797248,
    'x': 73.18144,
    'y': 113.12794349999997},
   'label': 'apple'},
  {'coordinates': {'height': 91.041203,
    'width': 116.11827199999999,
    'x': 75.881856,
    'y': 118.2456715},
   'label': 'apple'},
  {'coordinates': {'height': 103.43143799999999,
    'width': 113.95788799999998,
    'x': 75.34182399999999,
    'y': 111.51177399999997},
   'label': 'apple'}]}
```



notebook中加载了预览图，可以看到标的位置是正确的



训练结果如下：

```json
	{
		"metadata": {
			"outputType": "display_data",
			"metadata": {}
		},
		"outputItems": [
			{
				"mimeType": "text/html",
				"data": "<pre>| 26000        | 2.11781      | 4h 17m       |</pre>"
			},
			{
				"mimeType": "text/plain",
				"data": "| 26000        | 2.11781      | 4h 17m       |"
			}
		]
	},
```

```json
{'average_precision_50': {'apple': 0.35483407974243164,
  'banana': 0.17502447962760925,
  'cake': 0.22721879184246063,
  'candy': 0.17134447395801544,
  'carrot': 0.13678358495235443,
  'cookie': 0.1488942950963974,
  'doughnut': 0.2704440653324127,
  'grape': 0.1721615344285965,
  'hot dog': 0.31497931480407715,
  'ice cream': 0.12797603011131287,
  'juice': 0.41162046790122986,
  'muffin': 0.297859251499176,
  'orange': 0.2578592300415039,
  'pineapple': 0.25265851616859436,
  'popcorn': 0.39622601866722107,
  'pretzel': 0.2407742589712143,
  'salad': 0.38455572724342346,
  'strawberry': 0.22313976287841797,
  'waffle': 0.3182065188884735,
  'watermelon': 0.3446015417575836},
 'mean_average_precision_50': 0.2613580822944641}
```

机房的设备性能说好不好说差不差，这样应该够用了。从验证集的数据来看，基本都能正确识别到并且标注框。

## 应用开发

模板项目中基本已经实现完全了。

首先，将训练导出的mlmodel导入项目

需要我编写代码的地方主要有两处：

### func processObservations()

在此处`call show function`

```swift
func processObservations(for request: VNRequest, error: Error?) {
    //call show function
    DispatchQueue.main.async {
        let results = request.results as? [VNRecognizedObjectObservation]
        if results?.isEmpty == false {
            self.show(predictions: results!)
        } else {
            print("results empty")
        }
    }
}
```

`VNRecognizedObjectObservation`是CoreML提供的非常方便的结果的封装，下面的`show()`可以看到

### func show()

此处主要是把结果处理一下，然后用boundingBox显示

```swift
func show(predictions: [VNRecognizedObjectObservation]) {
    //process the results, call show function in BoundingBoxView
    for boxViewCount in 0..<boundingBoxViews.count {
        guard boxViewCount < predictions.count else {
            boundingBoxViews[boxViewCount].hide()
            return
        }

        let prediction = predictions[boxViewCount]
        let width = view.bounds.width
        let height = width * 1280 / 720
        let offsetY = (view.bounds.height - height) / 2
        let scale = CGAffineTransform.identity.scaledBy(x: width, y: height)
        let transform = CGAffineTransform(scaleX: 1, y: -1).translatedBy(x: 0, y: -height - offsetY)
        let rect = prediction.boundingBox.applying(scale).applying(transform)

        let label: String = prediction.labels[0].identifier + "\(prediction.labels[0].confidence * 100)%"

        let color = colors[prediction.labels[0].identifier] ?? UIColor.yellow
        boundingBoxViews[boxViewCount].show(frame: rect, label: label, color: color)
    }
}
```

首先，ViewController类有以下成员：

```swift
let maxBoundingBoxViews = 10
var boundingBoxViews = [BoundingBoxView]()
var colors: [String: UIColor] = [:]
```

`maxBoundingBoxViews`表示框的最大显示数量，`boundingBoxViews`是`BoundingBoxView`的集合，`colors`是一个Object name到UIColor的字典



如果结果的数量少于`BoundingBoxViews.count`，则将多出来的隐藏起来（如果是我我可能会选择周期性地把所有subView删了）

对于需要显示的，先对Observation中的boundingBox成员进行一些转换，转换成我们2D绘图常用的样式；根据置信度最高的结果构造label；选择颜色，如果在字典中没有就用黄色

最后，调用以下show方法

**不需要手动调用addToLayer()方法**，因为这些工作已经在`setUpCamera`里交付给`VideoCapture`类来实现了：

```swift
func setUpCamera() {
    videoCapture = VideoCapture()
    videoCapture.delegate = self

    // Change this line to limit how often the video capture delegate gets
    // called. 1 means it is called 30 times per second, which gives realtime
    // results but also uses more battery power.
    videoCapture.frameInterval = 1

    videoCapture.setUp(sessionPreset: .hd1280x720) { success in
        if success {
            // Add the video preview into the UI.
            if let previewLayer = self.videoCapture.previewLayer {
                self.videoPreview.layer.addSublayer(previewLayer)
                self.resizePreviewLayer()
            }

            // Add the bounding box layers to the UI, on top of the video preview.
            for box in self.boundingBoxViews {
                box.addToLayer(self.videoPreview.layer)
            }

            // Once everything is set up, we can start capturing live video.
            self.videoCapture.start()
        }
    }
}
```

