import os
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from torchvision.ops import nms
from tqdm import tqdm
import cv2
import numpy as np
import pdb


class MaskRCNN_R50:
    def __init__(self, score_threshold=0.5, proba_threshold=0.5, nms_iou_threshold=0.5):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        self.model.eval().to(self.device)
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
        self.score_threshold = score_threshold  # Score threshold to filter bboxes
        self.proba_threshold = proba_threshold  # probability threshold for converting masks to binary masks
        self.nms_iou_threshold = nms_iou_threshold

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

                    predictions = self.model(frame_rgb)
                    out_indices = nms(
                        predictions[0]["boxes"],
                        predictions[0]["scores"],
                        self.nms_iou_threshold,
                    )
                    masks = predictions[0]["masks"][out_indices][
                        predictions[0]["scores"][out_indices] > self.score_threshold
                    ]
                    if len(predictions[0]["masks"]) > 0:
                        masks = masks > self.proba_threshold
                        masks = masks.squeeze(1)
                        boxes = predictions[0]["boxes"][out_indices]
                        boxes = boxes[
                            predictions[0]["scores"][out_indices] > self.score_threshold
                        ]
                        img_out = draw_segmentation_masks(
                            self.transform_post(frame_rgb_orig), masks, alpha=0.7
                        )
                        img_out = draw_bounding_boxes(img_out, boxes)
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
                        img_out = cv2.cvtColor(frame_rgb_orig, cv2.COLOR_RGB2BGR)
                    videowriter.write(img_out)

                    pbar.update(1)

    def save_video(self, frame_size: tuple, save_path: str, fps=30):
        height, width = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(save_path, fourcc, fps, (width, height))


if __name__ == "__main__":
    video_path = "../data/videos/multiple_people.mp4"
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
