from time import time
import os
from typing import Union, Tuple, List
from itertools import zip_longest
from collections import defaultdict
from pathlib import Path
from dataclasses import dataclass
import cv2
import numpy as np

from config import Config
from yolo_detector import YoloDetector
from sort_tracker import Sort
from custom_decorators import timeit, log_call
from custom_logger import logger
from telegram_tool import TelegramTool


@dataclass
class GunDetection:
    model_path: Path = Config.MODEL_PATH
    classes_path: Path = Config.CLASSES_PATH
    device: str =  Config.DEVICE # "cpu", "cuda:0"
    sort_max_age: int = Config.SORT_MAX_AGE
    sort_min_hits: int = Config.SORT_MIN_HITS
    sort_iou_threshold: float = Config.SORT_IOU_THRESHOLD
    temp_folder_path: Path = Config.TEMP_FOLDER
    telegram_api_token: str = Config.TELEGRAM_API_KEY
    telegram_chat_id: str = Config.TELEGRAM_CHAT_ID
    debug: bool = False

    def __post_init__(self) -> None:
        os.makedirs(self.temp_folder_path, exist_ok=True)

        self.yolo_detector = YoloDetector(
            model_path=self.model_path,
            classes_path=self.classes_path,
            device=self.device
        )
        logger.info("Model loaded")

        self.sorttr = Sort(
            max_age=self.sort_max_age,
            min_hits=self.sort_min_hits,
            iou_threshold=self.sort_iou_threshold
        )
        logger.info("Sort alg loaded")

        if self.telegram_api_token and self.telegram_chat_id:
            self.tg = TelegramTool(
                bot_token=self.telegram_api_token
            )
        
        self.switch = 0
        self.class_colors = self.yolo_detector.colors
        self.class_colors["person"] = (0, 200, 0)
    
    def create_empty_summary(self) -> dict:
        return {class_name: set() for class_name in self.yolo_detector.classes_list}
    
    @timeit(logger=logger)
    def draw_bbox(self, img: np.ndarray, bbox: Tuple[int, int, int, int], class_name: str, 
                  obj_id: int, conf: float, colors: dict = None, font_size: float = 1.4,
                  font_thick: int = 2, bbox_thick: int = 2, force_color: Tuple[int, int, int] = None
                  ) -> None:
        if colors is None:
            colors = self.class_colors
        
        if not force_color:
            color = colors[class_name]
        else:
            color = force_color

        cv2.rectangle(img, bbox[:2], bbox[2:4], color, bbox_thick)
        cv2.putText(img, f"{class_name} {conf}-{obj_id}", (bbox[0], bbox[1]-10), cv2.FONT_HERSHEY_PLAIN,
                    font_size, color, font_thick
                    )
    
    @staticmethod
    def draw_summary(frame: np.array, summary: dict, x: int, y: int, font_size: float = 1.4, 
                     font_thick: int = 2, color: tuple[int, int, int] = (200, 200, 200), y_step: int = 20,
                     title: str = "Current summary") -> None:
        y_start = y
        cv2.putText(frame, title, (x, y_start), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_thick)
        y_start += y_step
        for key, val in summary.items():
            cv2.putText(frame, f"{key}: {len(val)}", (x, y_start), cv2.FONT_HERSHEY_PLAIN, font_size, color, font_thick)
            y_start += y_step
    
    @staticmethod
    def get_center(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
        return bbox[0] + (abs(bbox[2]-bbox[0])//2), bbox[1] + (abs(bbox[3]-bbox[1])//2)

    def rectangles_intersect(self, frame: np.array, rect1: tuple, rect2: tuple, error_margin: int):
        """
        Checks if the bboxes cross or not
        :param rect1:
        :param rect2:
        :param error_margin: in pixels
        :return:
        """
        x1_min, y1_min, x1_max, y1_max = rect1
        x2_min, y2_min, x2_max, y2_max = rect2

        x1_min -= error_margin
        y1_min -= error_margin
        x1_max += error_margin
        y1_max += error_margin

        # for debug, add frame param
        if self.debug:
            cv2.rectangle(frame, (x1_min, y1_min), (x1_max, y1_max), (0, 255, 255), 2)
        if x1_max < x2_min or x2_max < x1_min:
            return False

        if y1_max < y2_min or y2_max < y1_min:
            return False

        return True
    
    def check_for_owners(self, frame: np.array, gun_bb: Tuple[int], detections: List[tuple], error_margin: int) -> bool:
        """
        Loops through all people to check if person's bbox crossed with suitcase's bbox, if so, then we found owner
        :param suitcase_bb:
        :param detections:
        :return:
        """
        for detection in detections:
            if self.rectangles_intersect(frame, gun_bb, detection, error_margin) and not self.switch:
                return True
        return False
    
    def switch_states(self, frame: np.ndarray, person_data: dict, guns_data: dict, threshold: int, error_margin: int) -> dict:        
        check = self.check_for_owners(frame, person_data["Bbox"], [gun["Bbox"] for gun in guns_data.values()], error_margin)
        if check:
            person_data["FramesCount"] += 1
        else:
            person_data["FramesCount"] -= 1

        if person_data["FramesCount"] > threshold:
            person_data["FramesCount"] = threshold + 1
            person_data["Gunner"] = True
            person_data["Color"] = (0, 0, 200)
        
        if person_data["FramesCount"] < -threshold:
            person_data["FramesCount"] = -threshold - 1
            person_data["Gunner"] = False
            person_data["Color"] = (0, 200, 0)
        
        return person_data

    @timeit(logger=logger)
    def process_frames(self, frames: List[np.array], use_sahi: bool = True, conf: float = .2, iou: float = .35, 
                      augment: bool = True, sahi_conf: float = 0.2, sahi_slice_height: int = 256, 
                      sahi_slice_width: int = 256, sahi_overlap_height_ratio: float = 0.2, 
                      sahi_overlap_width_ratio: float = 0.2, detection_history: defaultdict[int] = None,
                      track: bool = True, draw_track: bool = True, min_det_frames: int = 5, 
                      font_size: float = 1.4, font_thick: int = 2, bbox_thick: int = 2,
                      persons_data: defaultdict[str] = None, guns_data: defaultdict[str] = None,
                      gun_threshold: int = 5, error_margin: int = 20
                      ) -> List[tuple[np.array, np.array, dict, np.array]]:
        frames_data = []

        frame_summary = self.create_empty_summary()
        if use_sahi:
            res = self.yolo_detector.detect_with_sahi(
                images=frames,
                conf=sahi_conf, 
                slice_height=sahi_slice_height, 
                slice_width=sahi_slice_width, 
                overlap_height_ratio=sahi_overlap_height_ratio,
                overlap_width_ratio=sahi_overlap_width_ratio
            )
            detection_results, detection_frames = map(list, zip(*res)) if res else ([], [])
            detection_gens = [self.yolo_detector.yield_sahi_data(bbox=detection_res) for detection_res in detection_results]
        else:
            # input(self.yolo_detector.detect(images=frames, conf=conf, iou=iou, augment=augment))
            res = self.yolo_detector.detect(images=frames, conf=conf, iou=iou, augment=augment)
            detection_results, detection_frames = map(list, zip(*res)) if res else ([], [])
            detection_gens = [self.yolo_detector.yield_data(bbox=detection_res) for detection_res in detection_results]

        #(2, 'block cracking', 0.44653183221817017, (769, 159, 966, 234)) sahi
        #(0, 'hole', 0.22668276727199554, (735, 166, 819, 191)) reg yolo
        total_track_data = []
        if track:
            for detection_gen in detection_gens:
                track_data = []
                for detection in detection_gen:
                    class_id, _, conf, x1, y1, x2, y2 = *detection[:3], *detection[3]
                    track_data.append([x1, y1, x2, y2, conf, class_id])

                updated_tracks = self.sorttr.update(track_data).astype(float) # rozpierdalasz sobie conf jak cos. EDIT: juz nie
                total_track_data.append(updated_tracks)

            for frame, updated_tracks, detection_frame in zip(frames, total_track_data, detection_frames):
                current_detections = set()
                for track_data in updated_tracks:
                    x1, y1, x2, y2, conf, obj_id, class_id = track_data
                    conf = round(track_data[-3], 1)
                    x1, y1, x2, y2, obj_id, class_id = int(x1), int(y1), int(x2), int(y2), int(obj_id), int(class_id)
                    class_name = self.yolo_detector.classes_list[class_id]

                    detection_key = f"{class_name[:3]}_{obj_id}"

                    # checking if object was seen X times to avoid counting same shit
                    current_detections.add(detection_key)

                    detection_history[detection_key] += 1
                    if detection_history[detection_key] >= min_det_frames:
                        frame_summary[class_name].add(obj_id)
                        if class_id == 0: # person
                            if detection_key not in list(persons_data.keys()):
                                persons_data[detection_key] = {
                                    "Bbox": (x1, y1, x2, y2),
                                    "FramesCount": 0,
                                    "Gunner": False,
                                    "Color": (0, 200, 0),
                                    "Conf": conf
                                }
                            # update kurwa
                            else:
                                persons_data[detection_key]["Bbox"] = (x1, y1, x2, y2)
                        else:
                            if detection_key not in list(guns_data.keys()):
                                guns_data[detection_key] = {
                                    "Bbox": (x1, y1, x2, y2),
                                    "FramesCount": 0
                                }
                            else:
                                # tu tez :PPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP
                                guns_data[detection_key]["Bbox"] = (x1, y1, x2, y2)
                                guns_data[detection_key]["Conf"] = conf
                    
                    if draw_track:
                        self.draw_bbox(img=frame, bbox=(x1, y1, x2, y2), class_name=class_name, obj_id=detection_key, 
                                    conf=conf, font_size=font_size, font_thick=font_thick, bbox_thick=bbox_thick)
                keys_to_remove = [key for key in detection_history if key not in current_detections]
                for key in keys_to_remove:
                    del detection_history[key]
                    try:
                        if "gun" in key:
                            # input(guns_data)
                            del guns_data[key]
                        else:
                            del persons_data[key]
                    except KeyError:
                        pass

                for person in persons_data.items():
                    person_id, person_data = person
                    
                    person_data = self.switch_states(frame=frame,
                                                     person_data=person_data,
                                                     guns_data=guns_data,
                                                     threshold=gun_threshold,
                                                     error_margin=error_margin
                                                     )
                    if person_data["Gunner"]:
                        self.draw_bbox(img=frame, bbox=person_data["Bbox"], class_name="person", obj_id=person_id, 
                                    conf=person_data["Conf"], font_size=font_size, font_thick=font_thick, bbox_thick=bbox_thick, force_color=person_data["Color"])

                frames_data.append((detection_frame, frame, frame_summary, total_track_data))
        else:
            for frame, detection_frame in zip(frames, detection_frames):
                frames_data.append((detection_frame, frame, {}, []))
        return frames_data, persons_data, guns_data
    
    @log_call(logger=logger, log_params=["vid_cap", "use_sahi", "conf", "iou", "augment", "sahi_conf", 
                                        "sahi_slice_height", "sahi_slice_width", "sahi_overlap_height_ratio", 
                                        "sahi_overlap_width_ratio", "track", "draw_track", "min_det_frames", 
                                        "font_size", "font_thick", "bbox_thick"])
    @timeit(logger=logger) 
    def process_video(self, vid_cap: Union[int, Path, str], use_sahi: bool = True, conf: float = .2, 
             iou: float = .35, augment: bool = True, sahi_conf: float = 0.2, 
             sahi_slice_height: int = 256, sahi_slice_width: int = 256, 
             sahi_overlap_height_ratio: float = 0.2, sahi_overlap_width_ratio: float = 0.2, 
             track: bool = True, draw_track: bool = True, min_det_frames: int = 5,
             font_size: float = 1.4, font_thick: int = 2, bbox_thick: int = 2, batch_size: int = 16,
             gun_threshold: int = 5, error_margin: int = 20
             ) -> None:
        
        summary = self.create_empty_summary()
        detection_hist = defaultdict(int)
        persons_data = defaultdict(str)
        guns_data = defaultdict(str)

        cap = cv2.VideoCapture(vid_cap)
        total_frames = 0
        start_time = time()
        alert_sent_time = None

        while cap.isOpened():
            batch_frames = []

            for _ in range(batch_size):
                success, frame = cap.read()
                if not success:
                    break
                batch_frames.append(frame)

            if not batch_frames:
                break
            batch_start = time()
            res, persons_data, guns_data = self.process_frames(
                frames=batch_frames, 
                use_sahi=use_sahi, 
                conf=conf, 
                iou=iou, 
                augment=augment, 
                sahi_conf=sahi_conf, 
                sahi_slice_height=sahi_slice_height, 
                sahi_slice_width=sahi_slice_width, 
                sahi_overlap_height_ratio=sahi_overlap_height_ratio, 
                sahi_overlap_width_ratio=sahi_overlap_width_ratio,
                detection_history=detection_hist,
                track=track,
                draw_track=draw_track,
                min_det_frames=min_det_frames,
                font_size=font_size,
                font_thick=font_thick,
                bbox_thick=bbox_thick,
                persons_data=persons_data,
                guns_data=guns_data,
                gun_threshold=gun_threshold,
                error_margin=error_margin
            )
            res = [res[-1]] # taking only last thing :p
            detection_frames, frames, summary_frames, track_data = map(list, zip(*res)) if res else ([], [], [], [])
            batch_end = time()
            batch_time = batch_end - batch_start

            """
            defaultdict(<class 'int'>, {'person_0': 4, 'person_1': 4, 'person_2': 4, 'person_3': 4, 'person_4': 4, 'person_5': 4, 'person_6': 4, 'person_7': 4, 'gun_8': 4, 'gun_9': 4, 'gun_10': 4, 'gun_11': 4, 'gun_12': 4, 'gun_13': 4, 'gun_14': 2})
            defaultdict(<class 'int'>, {'person_0': 8, 'person_1': 8, 'person_2': 8, 'person_3': 8, 'person_4': 8, 'person_5': 8, 'person_6': 8, 'person_7': 8, 'gun_8': 8, 'gun_9': 8, 'gun_10': 8, 'gun_11': 8, 'gun_12': 8, 'gun_13': 8, 'gun_14': 6})

            
            """

            batch_fps = len(batch_frames) / batch_time if batch_time > 0 else 0
            total_frames += len(batch_frames)

            show_frame = None
            reg_frame = None
            for frame, summary_frame, detection_frame in zip_longest(frames, summary_frames, detection_frames):
                reg_frame = detection_frame
                if track:
                    # Update
                    for key, val in summary_frame.items():
                        summary[key].update(val)
                    
                    self.draw_summary(frame=frame, summary=summary, x=10, y=75, title="Overall summary")
                    self.draw_summary(frame=frame, summary=summary_frame, x=10, y=140)
                    show_frame = frame
                else:
                    show_frame = detection_frame

                total_time = time() - start_time
                overall_fps = int(total_frames / total_time) if total_time > 0 else 0

                cv2.putText(show_frame, f"Batch FPS: {int(batch_fps)}", (10, 25), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)
                cv2.putText(show_frame, f"Overall FPS: {overall_fps}", (10, 50), cv2.FONT_HERSHEY_PLAIN, 1.4, (100, 0, 255), 2)
            
            gunners = []
            for key, val in persons_data.items():
                if val["Gunner"]:
                    gunners.append(key)

            if gunners and self.telegram_api_token and self.telegram_chat_id:
                if alert_sent_time:
                    diff = int(time() - alert_sent_time)
                    if diff >= 50:
                        alert_sent_time = None
                else:
                    file_path = f"{self.temp_folder_path}/{time()}.png"
                    cv2.imwrite(filename=file_path, img=show_frame)
                    msg = "Gunners: \n"
                    for gunner in gunners:
                        msg += f"- {gunner.replace('_', '')}\n"
                    self.tg.send_tg_message(msg=msg, chat_id=self.telegram_chat_id)
                    self.tg.send_tg_photo(photo_path=file_path, chat_id=self.telegram_chat_id)
                    alert_sent_time = time()
                    os.remove(file_path)

            key = cv2.waitKey(1)
            if self.debug:
                cv2.putText(show_frame, "Debug Mode", (10, 210), cv2.FONT_HERSHEY_PLAIN, 1.5, (255, 255, 255), 2)
                cv2.putText(show_frame, "Press 'f' to return False when owner checking", (10, 235), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 200, 200), 2)
                cv2.putText(show_frame, "Press 'g' to return True when owner checking", (10, 260), cv2.FONT_HERSHEY_PLAIN, 1.2, (0, 200, 200), 2)
                cv2.imshow("OriginalDetection", reg_frame)

                if key == ord("f"):
                    self.switch = 1
                if key == ord("g"):
                    self.switch = 0
            if key == 27:
                break
            
            cv2.imshow("Result", show_frame)
        cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    mocne = GunDetection(
        model_path=Config.MODEL_PATH,
        classes_path=Config.CLASSES_PATH,
        debug=Config.DEBUG
    )
    mocne.process_video(
        vid_cap=rf"{Config.VIDEOS_FOLDER}/6581397-hd_1280_720_24fps.mp4",
        use_sahi=Config.USE_SAHI,
        conf=Config.CONF_THRESH,
        iou=Config.IOU,
        augment=Config.AUGMENT,
        sahi_conf=Config.SAHI_CONF_THRESH,
        sahi_slice_height=Config.SAHI_SLICE_HEIGHT,
        sahi_slice_width=Config.SAHI_SLICE_WIDTH,
        sahi_overlap_height_ratio=Config.SAHI_OVERLAP_HEIGHT_RATIO,
        sahi_overlap_width_ratio=Config.SAHI_OVERLAP_WIDTH_RATIO,
        track=Config.TRACK,
        draw_track=Config.DRAW_TRACK,
        min_det_frames=Config.MIN_DET_FRAMES,
        font_size=Config.FONT_SIZE,
        font_thick=Config.FONT_THICK,
        bbox_thick=Config.BBOX_THICK,
        batch_size=Config.FRAME_BATCH_SIZE,
        gun_threshold=Config.GUN_THRESHOLD,
        error_margin=Config.ERROR_MARGIN
    )   
