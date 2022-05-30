import os
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn
import torchvision.transforms as T
from torchvision.utils import draw_segmentation_masks, draw_bounding_boxes
from tqdm import tqdm
import cv2
from time import perf_counter_ns
import numpy as np
from coco_classes import class_dict
import pickle


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

    def image_inference(
        self,
        image_path: str,
        draw_bbox=False,
        save_image=True,
        save_image_path=None,
        save_outputs=True,
        output_path=None,
    ):
        img = cv2.imread(image_path)
        with torch.no_grad():
            img_rgb, img_rgb_orig = self.preprocess(img)
            predictions = self.model(img_rgb)
            if len(predictions[0]["masks"]) > 0:
                masks, boxes, labels = self.process_predictions(predictions)
                img_out = self.visualization(
                    img_rgb_orig, masks, boxes, labels, draw_bbox, alpha=0.7
                )

                if save_image:
                    self.save_image(
                        img_out,
                        save_image_path,
                        save_outputs,
                        output_path,
                        masks,
                        boxes,
                        labels,
                        predictions[0]["scores"],
                    )

    def video_inference(
        self, video_path: str, draw_bbox=False, save_video=True, save_video_path=None
    ):
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
                    frame_height, frame_width, _ = frame.shape
                    frame_rgb, frame_rgb_orig = self.preprocess(frame)
                    t1 = perf_counter_ns()
                    predictions = self.model(frame_rgb)
                    if len(predictions[0]["masks"]) > 0:
                        masks, boxes, labels = self.process_predictions(predictions)

                        t2 = perf_counter_ns()
                        img_out = self.visualization(
                            frame_rgb_orig, masks, boxes, labels, draw_bbox, alpha=0.7
                        )
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

    def visualization(
        self, img_rgb_orig, masks, boxes, labels, draw_bbox=False, alpha=0.7
    ):
        img_out = draw_segmentation_masks(
            self.transform_post(img_rgb_orig), masks, alpha=alpha
        )
        if draw_bbox:
            img_out = draw_bounding_boxes(
                img_out, boxes, labels=labels, colors=(0, 0, 255)
            )
        img_out = np.moveaxis(img_out.numpy(), 0, -1)
        return cv2.cvtColor(img_out, cv2.COLOR_RGB2BGR)

    def preprocess(self, img):
        img_rgb_orig = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_rgb_norm = img_rgb_orig / 255
        return [self.transform_pre(img_rgb_norm).to(self.device)], img_rgb_orig

    def process_predictions(self, predictions):
        masks = predictions[0]["masks"] > self.proba_threshold
        masks = masks.squeeze(1)
        boxes = predictions[0]["boxes"]
        labels = [class_dict[label.item()] for label in predictions[0]["labels"]]
        return masks, boxes, labels

    def save_video(self, frame_size: tuple, save_path: str, fps=30):
        height, width = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    def save_image(
        self,
        img,
        save_path,
        save_outputs=False,
        output_path=None,
        masks=None,
        boxes=None,
        labels=None,
        scores=None,
    ):
        cv2.imwrite(save_path, img)
        if save_outputs:
            rows, cols, _ = img.shape
            boxes_np = boxes.cpu().numpy()
            boxes_np[:, [0, 2]] /= cols
            boxes_np[:, [1, 3]] /= rows
            outputs = {
                "masks": masks.cpu().numpy(),
                "bboxes": boxes_np,
                "bbox_labels": np.array(labels),
                "bbox_scores": scores.cpu().numpy(),
            }
            directory, _ = os.path.split(output_path)
            if not os.path.exists(directory):
                os.makedirs(directory)
            with open(output_path, "wb") as fh:
                pickle.dump(outputs, fh, pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    input_path = "../data/videos/single_person.mp4"
    # input_path = "../data/images/2-persons-white-bg.jpg"
    directory, filename = os.path.split(input_path)
    filename_wo_ext, ext = os.path.splitext(filename)
    filename_to_save = filename_wo_ext + "_infer_r50" + ext
    save_dir = "inferenced_videos/"
    # save_dir = "inferenced_images/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_visual_path = os.path.join(save_dir, filename_to_save)
    MRCNN50 = MaskRCNN_R50()
    MRCNN50.video_inference(
        input_path, draw_bbox=True, save_video=True, save_video_path=save_visual_path
    )
    # save_output_path = "outputs/outputs.pkl"
    # MRCNN50.image_inference(
    #     input_path,
    #     draw_bbox=True,
    #     save_image=True,
    #     save_image_path=save_visual_path,
    #     save_outputs=True,
    #     output_path=save_output_path,
    # )
