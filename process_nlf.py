import os
import sys
import cv2
import torch
import pickle
from tqdm import tqdm

# Load the pre-trained model
model = torch.jit.load('nlf_l_multi.torchscript').cuda().eval()

def process_video(video_path, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, os.path.splitext(os.path.basename(video_path))[0] + '.pkl')
    if os.path.exists(output_file):
        print(f"file exists: {output_file}")
        return
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    pose_results = {
        'joints3d_nonparam': [],
    }

    with torch.inference_mode(), torch.device('cuda'):
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Convert frame to tensor
            frame_tensor = torch.from_numpy(frame).cuda()
            frame_batch = frame_tensor.unsqueeze(0).permute(0,3,1,2)
            # Model inference
            pred = model.detect_smpl_batched(frame_batch)
            # Collect pose data
            for key in pose_results.keys():
                if key in pred:
                    #pose_results[key].append(pred[key].cpu().numpy())
                    pose_results[key].append(pred[key])
                else:
                    pose_results[key].append(None)

            frame_idx += 1

    cap.release()

    # Prepare output data
    output_data = {
        'video_path': video_path,
        'frame_count': frame_count,
        'video_width': video_width,
        'video_height': video_height,
        'poses': pose_results
    }

    # Save to pkl file
    with open(output_file, 'wb') as f:
        pickle.dump(output_data, f)


def check_video_file(output_video_path):
    if os.path.exists(output_video_path):
        print(f"File {output_video_path} saved.")
        print(f"File size: {os.path.getsize(output_video_path) / 1024} KB")
    else:
        print(f"File {output_video_path} not found.")


def generate_video(image_paths, output_path):
    frame = cv2.imread(image_paths[0])
    if frame is None:
        return
    height, width, layers = frame.shape

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, 30, (width, height))

    if not video.isOpened():
        return

    for image_path in image_paths:
        frame = cv2.imread(image_path)
        if frame is None:
            continue
        video.write(frame)

    video.release()
    check_video_file(output_path)


def process_directory(video_dir):
    for root, _, files in os.walk(video_dir):
        has_video = False
        images = []
        for file in tqdm(files, desc="Processing files", unit="file"):
            if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                video_path = os.path.join(root, file)
                output_dir = root + '_smpl'
                # print(output_dir)
                print(video_path)
                process_video(video_path, output_dir)



if __name__ == '__main__':
    video_directory = sys.argv[-1]
    process_directory(video_directory)
