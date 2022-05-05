import os
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import nms
from tqdm import tqdm
import cv2
from time import perf_counter_ns
import numpy as np
from coco_classes import class_dict
import pdb


class MaskRCNN_R50:
    def __init__(self, score_threshold=0.5, proba_threshold=0.5, nms_iou_threshold=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        t1_setup = perf_counter_ns()
        self.model = maskrcnn_resnet50_fpn(
            pretrained=True,
            box_nms_thresh=nms_iou_threshold,
            box_score_thresh=score_threshold,
        )
        self.model.eval().to(self.device)
        t2_setup = perf_counter_ns()
        print(f"Model setup time: {(t2_setup - t1_setup)*1e-9:.3f}")
        self.transform_pre = T.Compose(
            [
                T.ToTensor(),
                T.ConvertImageDtype(torch.float32),
            ]
        )
        self.transform_post = T.Compose(
            [
                T.ToTensor(),
                T.ConvertImageDtype(torch.uint8),
            ]
        )
        self.proba_threshold = proba_threshold  # probability threshold for converting masks to binary masks
        self.inference_time = 0

    def video_inference(self, video_path: str, save_video=True, save_video_path=None):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        videowriter_initialize = False
        with torch.no_grad():
            with tqdm(total=total_frames) as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        print("Can't receive frame...")
                        break
                    frame_rgb_orig = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_rgb_norm = frame_rgb_orig / 255
                    frame_height, frame_width, _ = frame_rgb_orig.shape
                    frame_rgb = [self.transform_pre(frame_rgb_norm).to(self.device)]
                    t1 = perf_counter_ns()
                    predictions = self.model(frame_rgb)
                    if len(predictions[0]["masks"]) > 0:
                        masks = predictions[0]["masks"] > self.proba_threshold
                        masks = masks.squeeze(1)
                        boxes = predictions[0]["boxes"]
                        labels = [
                            class_dict[label.item()]
                            for label in predictions[0]["labels"]
                        ]

                        t2 = perf_counter_ns()
                        img_out = draw_segmentation_masks(
                            self.transform_post(frame_rgb_orig), masks, alpha=0.7
                        )
                        img_out = draw_bounding_boxes(img_out, boxes, labels=labels)
                        img_out = np.moveaxis(img_out.numpy(), 0, -1)
                        img_out = cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)
                        if save_video:
                            if not videowriter_initialize:
                                videowriter_initialize = True
                                assert (
                                    save_video_path is not None
                                ), "Please enter the path for the inferred video to be saved"
                                videowriter = self.save_video(
                                    (frame_height, frame_width), save_video_path, fps=30
                                )
                    else:
                        t2 = perf_counter_ns()
                        img_out = cv2.cvtColor(frame_rgb_orig, cv2.COLOR_RGB2BGR)
                    self.inference_time += t2 - t1
                    if save_video:
                        videowriter.write(img_out)

                    pbar.update(1)

                if save_video:
                    videowriter.release()
                    print("Video saved successfully")
                cap.release()
                print("****************      INFERENCE DONE      *************")
                print(f"  TOTAL INFERENCE TIME : {self.inference_time*1e-9:.3f}s")
                print(f"  FPS: {total_frames/(self.inference_time*1e-9):.2f}")

    def save_video(self, frame_size: tuple, save_path: str, fps=30):
        height, width = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(save_path, fourcc, fps, (width, height))


if __name__ == "__main__":
    video_path = "../data/videos/single_person.mp4"
    directory, filename = os.path.split(video_path)
    filename_wo_ext, ext = os.path.splitext(filename)
    filename_to_save = filename_wo_ext + "_infer_r50" + ext
    save_dir = "inferenced_videos/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_video_path = os.path.join(save_dir, filename_to_save)
    MRCNN50 = MaskRCNN_R50()
    MRCNN50.video_inference(
        video_path, save_video=True, save_video_path=save_video_path
    )
