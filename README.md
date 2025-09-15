# Gun Detection System
Gun detection system, it...detects guns and people, assigns id for each, determinates if someone's holding a gun. It can also send frame and text summary message to Telegram.

[![video](https://img.youtube.com/vi/InMwlxTmGhQ/0.jpg)](https://www.youtube.com/watch?v=InMwlxTmGhQ)

## Setup
  - install python 3.11+ (version I use)
  - ```pip install -r requirements.txt```
  - create `.env` file
    ```
    BOT_TOKEN=""
    CHAT_ID=""
    ```

## Packages
```
filterpy==1.4.5
numpy==1.26.0
opencv_contrib_python==4.10.0.84
opencv_python==4.9.0.80
opencv_python_headless==4.9.0.80
Pillow==11.3.0
python-dotenv==1.1.1
Requests==2.32.3
sahi==0.11.22
tqdm==4.65.0
ultralytics==8.3.89
```

## Project Structure
```
├───DatasetPrepTools/
│   ├───class_counts.py
│   ├───dataset_cleaner.py
│   ├───move_files.py
│   ├───get_data.py
│   └───setup_dataset_folder.py
├───models/
├───TestData/
├───train_data/
│   ├───images/
│   │   ├───train/
│   │   └───val/
│   └───labels/
│       ├───train/
│       └───val/
├───Videos/
├───models/
│   ├───classes.txt
│   ├───yolo11_1_tuned.pt
│   ├───yolo11_1.pt
│   └───yolov5_1.pt
│ 
├───.env
├───config.py
├───custom_decorators.py
├───custom_logger.py
├───main.py
├───sort_tracker.py
├───requirements.txt
└───yolo_detector.py
```

## Configuration

### Main parameters:
 - `TRACK` - bool, track objects or not
 - `DRAW_TRACK` - bool, draw tracking
 - `MIN_DET_FRAMES` - int, minimum number of occurrences (frames) of an object for it to be taken into account - reduction of false positives
 - `SORT_MAX_AGE` - int, max age of tracked object - for example if set to 5 object will be lost if not seen for 5 frames
 - `SORT_MIN_HITS`- int, how many times object has to be seen in order to be tracked
 - `SORT_IOU_THRESHOLD` - int, Intersection Over Union (IoU) threshold - it's used to "recognize" if it's the same object, for example if set to 0.3 boxes have to align in 30% in order to tell if it's the same object.
 - `FONT_SIZE` - int
 - `FONT_THICK` - int
 - `BBOX_THICK` - int
 - `GUN_THRESHOLD` - int, minimum number of occurrences (frames) of already detected gun that meet the requirements (being close enough to person), so person can be classified as a gunner
 - `DEBUG` - bool
 - `TEMP_FOLDER` - Path
 - `ERROR_MARGIN` - int, expands bbox of each person, because sometimes guns is really close, literally in hand of the person but bboxes of both objects do not overlap, so with that parameter bbox is expanded
 - `FRAME_BATCH_SIZE` - int, how many frames to process at once

 ### Model
 - `CLASSES_PATH` - Path 
 - `MODEL_PATH` - Path
 - `DEVICE` - str

### Regular
 - `IOU` - float, Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates
 - `CONF_THRESH` - float, detection confidence threshold (0-1.0)
 - `AUGMENT` - bool, enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed
 - `AGNOSTIC_NMS` - bool, enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.

### Sahi
 - `USE_SAHI` - bool
 - `SAHI_CONF_THRESH` - float
 - `SAHI_SLICE_HEIGHT` - int
 - `SAHI_SLICE_WIDTH` - int
 - `SAHI_OVERLAP_HEIGHT_RATIO` - float
 - `SAHI_OVERLAP_WIDTH_RATIO` -  float

 ## Detectors
### <u>***yolo_detector_old.py***</u>
Detector for yolov5, nothing fancy.
#### **Attributes:**
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | - | Path to the YOLOv5 model weights file (required) |
| `conf_threshold` | `float` | 0.1 | Confidence threshold for filtering detections |
| `ultralytics_path` | `str` | "ultralytics/yolov5" | Path to the ultralytics YOLOv5 implementation |
| `model_type` | `str` | "custom" | Type of model to load ("custom" or pre-trained) |
| `force_reload` | `bool` | True | Whether to force reload the model |

#### **Methods:**
`detect` - detection function that processes an image and returns detection results.
#### **Parameters:**
 - `img`: `Union[str, np.array]` - Numpy image or path to an image.

#### **Returns:**
- `Tuple[np.array, pd.DataFrame]` - Image with drawn detections, dataframe with detections.

### **Basic usage:**
```python
yolo_predictor = YoloDetector(
        model_path=model_path,
)
image = cv2.imread(image_path)
converted = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
image_draw, res = detector.detect(img=converted)

for detection in res:
    bbox, conf, obj_id, class_id = detection[:4], *detection[4:]
cv2.imshow("res", image_draw)
cv2.waitKey(0)
```

### <u>***yolo_detector.py***</u>
Detector for yolov11, for both regular and sahi version.

| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | - | Path to the YOLOv11 model weights file (required) |
| `classes_path` | `str` | - | Path to model classes (required) |
| `device` | `str` |  `"cpu"` | Device ("cpu", "cuda:0") |

### **Methods:**
`detect` - detection function that processes an image and returns detection results.

#### **Parameters:**
 - `images`: `List[np.array]` - List of numpy images.
 - `conf`: `float` - Detection confidence threshold (0-1.0).
 - `iou`: `float` - Intersection Over Union (IoU) threshold for Non-Maximum Suppression (NMS). Lower values result in fewer detections by eliminating overlapping boxes, useful for reducing duplicates.
 - `augment`: `bool` - Enables test-time augmentation (TTA) for predictions, potentially improving detection robustness at the cost of inference speed.
 - `agnostic_nms`: `float` - Enables class-agnostic Non-Maximum Suppression (NMS), which merges overlapping boxes of different classes. Useful in multi-class detection scenarios where class overlap is common.

#### **Returns:**
  - `List[tuple]` - return list of tuples, each tuple contains pair of detection data and image with drawn detections.

<br>
<hr>

`detect_with_sahi` - detection function that processes an image and returns detection results BUT with **SAHI** (Slicing Aided Hyper Inference - https://docs.ultralytics.com/guides/sahi-tiled-inference/), which helps a lot in detecting really small objects.

#### **Parameters:**
 - `images`: `List[np.ndarray]` - List of numpy images.
 - `conf`: `float` - Confidence threshold.
 - `slice_height`, `slice_width`: `int` - The larger the better detection of smaller objects but longer processing time.
 - `overlap_height_ratio`, `overlap_width_ratio`: `int` - Slice overlay.

#### **Returns:**
 - `List[tuple]` - return list of tuples, each tuple contains pair of detection data and image with drawn detections.

<br>
<hr>

`yield_data` - Method for yielding detection data from `detect` method.
#### **Parameters:**
 - `bbox`: `Boxes` - Detection data from `detect`.
#### **Returns:**
 - `Generator` - processed detection data into: `cls_id, class_name, conf,  (x1, y1, x2, y2)`

 <br>
<hr>

`yield_sahi_data` - Method for yielding detection data from `detect_with_sahi` method.
#### **Parameters:**
 - `sahi_result`: `PredictionResult` - Detection data from `detect_with_sahi`.
#### **Returns:**
- `Generator` - processed detection data into: `cls_id, class_name, conf, (x1, y1, x2, y2)`

### **Basic usage:**
```python
yolo_predictor = YoloDetector(
        model_path=model_path,
        classes_path=classes_path
)
image = cv2.imread(image_path)
res, res_img = yolo_predictor.detect(images=[image])[0]
#res, res_img = self.yolo_detector.detect_with_sahi(images=[frame])[0]

detection_res_gen = self.yolo_detector.yield_data(bbox=res)
# detection_res_gen = self.yolo_detector.yield_sahi_data(sahi_result=res)
for detection in detection_res_gen:
    class_id, _, conf, x1, y1, x2, y2 = *detection[:3], *detection[3]
cv2.imshow("res", res_img)
cv2.waitKey(0)
```

## System
Main functionality sits in **main.py** file, which uses few other modules:
 - **config.py** - managing configuration
 - **sort_tracker.py** - providing SORT algorithm for tracking detection object.
 - **yolo_detector.py** - YOLO detector
 - **custom_decorators.py** - providing extra functionality, mostly for logging
 - **custom_logger** - providing logger
 - **telegram_tool** - providing ability to send telegram messages, both text and files (images)

### <u>***main.py***</u>
Main file...yes

#### **Attributes:**
| Attribute | Type | Default | Description |
|-----------|------|---------|-------------|
| `model_path` | `str` | value from `Config.MODEL_PATH` | Path to the YOLOv11 model.
| `classes_path` | `str` | value from `Config.CLASSES_PATH` | Path to the classes path.
| `device` | `str` | value from `Config.DEVICE` | Device.
| `sort_max_age` | `int` | value from `Config.SORT_MIN_HITS` | Max age of tracked object - for example if set to 5 object will be lost if not seen for 5 frames.
| `sort_min_hits` | `int` |  value from `Config.SORT_MIN_HITS`| How many times obejct has to be seen in order to be tracked. 
| `sort_iou_threshold` | `int` | value from `Config.SORT_IOU_THRESHOLD` |Intersection Over Union (IoU) threshold - it's used to "recognize" if it's the same object, for example if set to 0.3 boxes have to align in 30% in order to tell if it's the same object.
| `temp_folder_path` | `Path` | value from `Config.TEMP_FOLDER` | Folder for temp files, in this cases images that are being sent via Telegram.
| `telegram_api_token` | `str` | value from `Config.TELEGRAM_API_KEY` | Telegram api token, suprised?
| `telegram_chat_id` | `str` | value from `Config.TELEGRAM_CHAT_ID` | Telegram chat id :)


#### **Methods:**

**`process_frame`** - method for processing multiple frames - making detections and tracking.

#### **Parameters:**
 - `frame`: ` np.array`
 - `use_sahi`: `bool`
 - `conf`: `float` - Confidence threshold for regular YOLO models.
 - `iou`: `float` - IOU for regular YOLO models.
 - `augment`: `bool` - Augment for regular YOLO models.
 - `sahi_conf`: `float`
 - `sahi_slice_height`: `int`
 - `sahi_slice_width`: `int`
 - `sahi_overlap_height_ratioi_conf`: `float`
 - `sahi_overlap_height_ratio`: `float`
 - `detection_history`: `defaultdict[int]` - Detection history, used to avoid counting the same object over and over again.
 - `track`: `bool` - Tracking objects with SORT algorithm.
 - `min_det_frames`: `int` - Used to avoid counting objects that appeared for a frame or two, so object has to appear X times in order to be counted, **if you're working with just one image not frame stream you can set it to 0.**
 - `font_size`: `float`
 - `font_thick`: `int`
 - `bbox_thick`: `int`
 - `persons_data`: `defaultdict[str]` - provide defaultdict with person data (empty one in the first step), so method can return data, and in later you can overwrite defaultdict you've provided and update it again to keep up continuation.
 - `guns_data`: `defaultdict[str]` - same thing like above but for guns.
 - `gun_threshold`: `int` - minimum number of occurrences (frames) of already detected gun that meet the requirements (being close enough to person), so person can be classified as a gunner.
 - `error_margin`: `int` - expands bbox of each person, because sometimes guns is really close, literally in hand of the person but bboxes of both objects do not overlap, so with that parameter bbox is expanded.

 #### **Returns:**
 - `Tuple[List[tuple[np.array, np.array, dict, np.array]], list, list]` - Returns list of frames raw with detections, frames with processed detections, summaries, tracking data, person data, gun data.

<br>
<hr>

**`process_video`** - same thing as `process_frame` but for video and with drawing FPS and count summary.

#### **Parameters:**
 - `vid_cap`: `Union[int, Path, str]`
 - `use_sahi`: `bool`
 - `conf`: `float` - Confidence threshold for regular YOLO models.
 - `iou`: `float` - IOU for regular YOLO models.
 - `augment`: `bool` - Augment for regular YOLO models.
 - `sahi_conf`: `float`
 - `sahi_slice_height`: `int`
 - `sahi_slice_width`: `int`
 - `sahi_overlap_height_ratioi_conf`: `float`
 - `sahi_overlap_height_ratio`: `float`
 - `detection_history`: `defaultdict[int]` - Detection history, used to avoid counting the same object over and over again.
 - `track`: `bool` - Tracking objects with SORT algorithm.
 - `min_det_frames`: `int` - Used to avoid counting objects that appeared for a frame or two, so object has to appear X times in order to be counted, **if you're working with just one image not frame stream you can set it to 0.**
 - `font_size`: `float`
 - `font_thick`: `int`
 - `bbox_thick`: `int`
 - `gun_threshold`: `int` - minimum number of occurrences (frames) of already detected gun that meet the requirements (being close enough to person), so person can be classified as a gunner.
 - `error_margin`: `int` - expands bbox of each person, because sometimes guns is really close, literally in hand of the person but bboxes of both objects do not overlap, so with that parameter bbox is expanded.
 - `batch_size`: `int` - how many frames to process at once
#### **Returns:**
 - `None`
   
<br>
<hr>

 **`draw_summary`** - drawing count summary.
 #### **Parameters:**
  - `frame`: `np.array`
  - `summary`: `dict`
  - `x`: `int` - Where summary starts in x axis.
  - `y`: `int` - Where summary starts in y axis.
  - `font_size`: `float`
  - `font_thick`: `int`
  - `color`: `tuple[int, int, int]`
  - `y_step`: `int` - Step between each class count.
  - `title`: `str`
#### **Returns:**
 - `None`

 <br>
<hr>

**`draw_bbox`** - drawing object bbox.
#### **Parameters:**
  - `img`: `np.ndarray`
  - `bbox`: `Tuple[int, int, int, int]`
  - `class_name`: `str`
  - `obj_id`: `int`
  - `conf`: `float`
  - `font_thick`: `int`
  - `font_size`: `float`
  - `colors`: `dict` - Colors for each class.
  - `bbox_thick`: `int`
#### **Returns:**
 - `None`

**`get_center`** - gets center of bbox.
#### **Parameters:**
  - `bbox`: `Tuple[int, int, int, int]`
#### **Returns:**
 - `Tuple[int, int]`

**`rectangles_intersect`** - checks if bboxes overlap on each other.
#### **Parameters:**
  - `frame`: `np.array`
  - `rect1`: `tuple`
  - `rect2`: `tuple`
  - `error_margin`: `int`
#### **Returns:**
 - `Tuple[int, int]`

**`check_for_owners`** - checks if there is some person bbox that overlap with gun's bbox, if so, breaks the loop and returns True.
#### **Parameters:**
  - `frame`: `np.array`
  - `gun_bb`: `Tuple[int]` - bbox
  - `detections`: `List[tuple]` - list of person bboxes
  - `error_margin`: `int`
#### **Returns:**
 - `bool`

**`switch_states`** - based on result of `check_for_owners` adds 1 when result is True and -1 when False, if `FramesCount` is larger than `threshold (Config.GUN_THRESHOLD)` then person is a gunner, if lower than -threshold then is no longer a gunner or never was a gunner.
#### **Parameters:**
  - `frame`: `np.ndarray`
  - `person_data`: `dict`
  - `guns_data`: `dict`
  - `threshold`: `int`
  - `error_margin`: `int`
#### **Returns:**
 - `bool`

## Logging and decorators

### Logging
Logging tools are handled by `custom_logger.py` it logs to both file and CLI.

### Decorators
Logging decroators because I like logging - `custom_decorators.py`.

**`timeit`**
#### **Parameters:**
 - `logger`: `Logger` - you can log execution time.
 - `print_time`: `bool` - or not, you can just print it.
 - `return_val`: `bool` - you can also return execution time.

 <hr>
 <br>

**`log_call`**
#### **Parameters:**
 - `logger`: `Logger` - you can log execution time.
 - `log_params`: `list` - list of parameters of method you want to log, sometimes you don't want to log some stuff, so you can choose only the ones you are interested in or you can set it to `[""]` to not log anything.
 - `hide_res`: `bool` - show (or not) retuned data.
