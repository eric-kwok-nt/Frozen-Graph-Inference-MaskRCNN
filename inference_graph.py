import tensorflow as tf
import os
import time
import numpy as np
from saved_model_preprocess import ForwardModel
import cv2
from mrcnn.config import Config
import visualize_custom as visualize
from tqdm import tqdm
import pdb


class_names = [
    "BG",
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


class InferenceConfig(Config):
    """Configuration for training on MS COCO.
    Derives from the base Config class and overrides values specific
    to the COCO dataset.
    """

    # Give the configuration a recognizable name
    NAME = "coco"

    # We use a GPU with 12GB memory, which can fit two images.
    # Adjust down if you use a smaller GPU.
    IMAGES_PER_GPU = 1

    # Uncomment to train on 8 GPUs (default is 1)
    GPU_COUNT = 1

    # Number of classes (including background)
    NUM_CLASSES = 1 + 80  # COCO has 80 classes


class GraphInference:
    def __init__(self, graph_path):
        self.graph_path = graph_path
        self.inference_time = 0
        t1_setup = time.time()
        self.coco_config = InferenceConfig()
        self.coco_config.display()

        # PreProcess Model
        self.preprocess_obj = ForwardModel(self.coco_config)  # config , outputs

        # Create frozen function
        self.frozen_func = None
        self._build_graph()
        t2_setup = time.time()
        print(f"Model setup time: {t2_setup - t1_setup:.3f}")

    def _wrap_frozen_graph(self, graph_def, inputs, outputs, print_graph=False):
        def _imports_graph_def():
            tf.compat.v1.import_graph_def(graph_def, name="")

        wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
        import_graph = wrapped_import.graph

        layers = [op.name for op in import_graph.get_operations()]
        if print_graph == True:
            print("-" * 50)
            print("Frozen model layers: ")
            for layer in layers:
                print(layer)
            print("-" * 50)

        return wrapped_import.prune(
            tf.nest.map_structure(import_graph.as_graph_element, inputs),
            tf.nest.map_structure(import_graph.as_graph_element, outputs),
        )

    def _preprocess_image(self, image):
        images = np.expand_dims(image, axis=0)
        molded_images, image_metas, windows = self.preprocess_obj.mold_inputs(images)
        molded_images = tf.convert_to_tensor(molded_images, dtype=tf.float32)
        image_metas = tf.convert_to_tensor(image_metas, dtype=tf.float32)
        image_shape = molded_images[0].shape

        for g in molded_images[1:]:
            assert (
                g.shape == image_shape
            ), "After resizing , all images must have the same size , Check IMAGE_RESIZE_MODE and image sizes"

        anchors = self.preprocess_obj.get_anchors(image_shape)
        anchors = np.broadcast_to(anchors, (images.shape[0],) + anchors.shape)
        anchors = tf.convert_to_tensor(anchors)

        return images, molded_images, image_metas, anchors, windows

    def _build_graph(self):
        graph_pb = self.graph_path

        with tf.compat.v2.io.gfile.GFile(graph_pb, "rb") as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())

        self.frozen_func = self._wrap_frozen_graph(
            graph_def=graph_def,
            inputs=["input_image:0", "input_image_meta:0", "input_anchors:0"],
            outputs=["mrcnn_detection/Reshape_1:0", "mrcnn_mask/Reshape_1:0"],
            print_graph=False,
        )

    def video_inference(self, video_path: str, save_video=True, save_video_path=None):
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        videowriter_initialize = False
        with tqdm(total=total_frames) as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Can't receive frame...")
                    break
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result_dict = self.inference(frame_rgb)
                infer_image = visualize.display_instances(
                    frame_rgb,
                    result_dict["rois"],
                    result_dict["mask"],
                    result_dict["class"],
                    class_names,
                    result_dict["scores"],
                    title="Predictions",
                )
                if save_video:
                    if not videowriter_initialize:
                        videowriter_initialize = True
                        frame_height, frame_width, _ = frame.shape
                        assert (
                            save_video_path is not None
                        ), "Please enter the path for the inferred video to be saved"
                        videowriter = self.save_video(
                            (frame_height, frame_width), save_video_path, fps=30
                        )
                    videowriter.write(infer_image)
                pbar.update(1)

        if save_video:
            videowriter.release()
            print("Video saved successfully")
        cap.release()
        print("****************      INFERENCE DONE      *************")
        print(f"  TOTAL INFERENCE TIME : {self.inference_time:.3f}s")
        print(f"  FPS: {total_frames/self.inference_time:.2f}")

    def inference(self, frame):
        (
            images,
            molded_images,
            image_metas,
            anchors,
            windows,
        ) = self._preprocess_image(frame)
        result = dict()
        t1 = time.time()

        output = self.frozen_func(molded_images, image_metas, anchors)
        result["mrcnn_detection/Reshape_1"] = output[0].numpy()
        result["mrcnn_mask/Reshape_1"] = output[1].numpy()

        t2 = time.time()
        self.inference_time += t2 - t1

        result_dict = self.preprocess_obj.result_to_dict(
            images, molded_images, windows, result
        )[0]

        return result_dict

    def save_video(self, frame_size: tuple, save_path: str, fps=30):
        height, width = frame_size
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        return cv2.VideoWriter(save_path, fourcc, fps, (width, height))

    # @staticmethod
    # def get_ax(rows=1, cols=1, size=16):
    #     """Return a Matplotlib Axes array to be used in
    #     all visualizations . Provide a
    #     central point to control graph sizes.

    #     Adjust the size attribute to control how big to render images
    #     """
    #     _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))

    #     return ax


if __name__ == "__main__":
    graph_path = "./model/frozen_graph.pb"
    video_path = "../data/videos/single_person.mp4"
    directory, filename = os.path.split(video_path)
    filename_wo_ext, ext = os.path.splitext(filename)
    filename_to_save = filename_wo_ext + "_infer" + ext
    save_dir = "inferenced_videos/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_video_path = os.path.join(save_dir, filename_to_save)
    InferObj = GraphInference(graph_path=graph_path)
    InferObj.video_inference(
        video_path, save_video=False, save_video_path=save_video_path
    )
