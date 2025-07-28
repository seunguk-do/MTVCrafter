import os
import pickle
from pathlib import Path

import cv2
import tyro
import torch
import torchvision  # Don't Delete this line:  https://github.com/pytorch/pytorch/issues/48932
from tqdm import tqdm


def process_video(model, video_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(
        output_dir, os.path.splitext(os.path.basename(video_path))[0] + ".pkl"
    )
    if os.path.exists(output_file):
        print(f"file exists: {output_file}")
        return
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_results = {
        "joints3d_nonparam": [],
    }

    with torch.inference_mode(), torch.device("cuda"):
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).cuda()
            frame_batch = frame_tensor.unsqueeze(0).permute(0, 3, 1, 2)
            # Model inference
            pred = model.detect_smpl_batched(frame_batch)
            # Collect pose data
            for key in pose_results.keys():
                if key in pred:
                    # pose_results[key].append(pred[key].cpu().numpy())
                    pose_results[key].append(pred[key])
                else:
                    pose_results[key].append(None)

            frame_idx += 1

    cap.release()

    # Prepare output data
    output_data = {
        "video_path": video_path,
        "video_length": frame_count,
        "video_width": video_width,
        "video_height": video_height,
        "pose": pose_results,
    }

    # Save to pkl file
    with open(output_file, "wb") as f:
        pickle.dump(output_data, f)


def process_directory(video_dir: str):
    model_path = (
        Path(os.environ["DATA_DIR"])
        / "pretrained_weights/nlf_l_multi_0.3.2.torchscript"
    )

    # Load the pre-trained model
    print(f"Loading NLF model from: {model_path}")

    model = torch.jit.load(model_path).cuda().eval()

    for root, _, files in os.walk(video_dir):
        for file in tqdm(files, desc="Processing files", unit="file"):
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_path = os.path.join(root, file)
                output_dir = root + "_smpl"
                print(video_path)
                process_video(model, video_path, output_dir)


if __name__ == "__main__":
    tyro.cli(process_directory)
