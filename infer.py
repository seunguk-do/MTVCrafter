import os
import copy
import torch
import random
import pickle
import decord
import imageio
import numpy as np

from PIL import Image
from tqdm import tqdm
from torchvision import transforms
from torchvision.transforms import functional as F
from torchvision.transforms import ToPILImage, InterpolationMode

from models import MTVCrafterPipeline
from models import VectorQuantizer, Encoder, Decoder, SMPL_VQVAE
from draw_pose import get_pose_images

    

def concat_images(images, direction='horizontal', pad=0, pad_value=0):
    if len(images) == 1:
        return images[0]
    is_pil = isinstance(images[0], Image.Image)
    if is_pil:
        images = [np.array(image) for image in images]
    if direction == 'horizontal':
        height = max([image.shape[0] for image in images])
        width = sum([image.shape[1] for image in images]) + pad * (len(images) - 1)
        new_image = np.full((height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype)
        begin = 0
        for image in images:
            end = begin + image.shape[1]
            new_image[: image.shape[0], begin:end] = image
            begin = end + pad
    elif direction == 'vertical':
        height = sum([image.shape[0] for image in images]) + pad * (len(images) - 1)
        width = max([image.shape[1] for image in images])
        new_image = np.full((height, width, images[0].shape[2]), pad_value, dtype=images[0].dtype)
        begin = 0
        for image in images:
            end = begin + image.shape[0]
            new_image[begin:end, : image.shape[1]] = image
            begin = end + pad
    else:
        assert False
    if is_pil:
        new_image = Image.fromarray(new_image)
    return new_image


def concat_images_grid(images, cols, pad=0, pad_value=0):
    new_images = []
    while len(images) > 0:
        new_image = concat_images(images[:cols], pad=pad, pad_value=pad_value)
        new_images.append(new_image)
        images = images[cols:]
    new_image = concat_images(new_images, direction='vertical', pad=pad, pad_value=pad_value)
    return new_image

def sample_video(video, indexes, method=2):
    if method == 1:
        frames = video.get_batch(indexes)
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
    elif method == 2:
        max_idx = indexes.max() + 1
        all_indexes = np.arange(max_idx, dtype=int)
        frames = video.get_batch(all_indexes)
        frames = frames.numpy() if isinstance(frames, torch.Tensor) else frames.asnumpy()
        frames = frames[indexes]
    else:
        assert False
    return frames

def get_sample_indexes(video_length, num_frames, stride):
    assert num_frames * stride <= video_length
    sample_length = min(video_length, (num_frames - 1) * stride + 1)
    start_idx = 0 + random.randint(0, video_length - sample_length)
    start_idx = 0
    sample_indexes = np.linspace(start_idx, start_idx + sample_length - 1, num_frames, dtype=int)
    return sample_indexes

def get_new_height_width(data_dict, dst_height, dst_width):
    height = data_dict['video_height']
    width = data_dict['video_width']
    print("original:", height, width)
    if float(dst_height) / height < float(dst_width) / width:
        new_height = int(round(float(dst_width) / width * height))
        new_width = dst_width
    else:
        new_height = dst_height
        new_width = int(round(float(dst_height) / height * width))
    assert dst_width <= new_width and dst_height <= new_height
    return new_height, new_width


def inference(device, motion_data_path, ref_image_path='', output_dir='inference_output'):
    dst_width, dst_height = (512, 512)
    num_frames = 49
    to_pil = ToPILImage()
    normalize = transforms.Normalize([0.5], [0.5])
    os.makedirs(output_dir, exist_ok=True)

    with open(motion_data_path, 'rb') as f1:
        data = pickle.load(f1)
    
    with open('data/statistics.pkl', 'rb') as f2:
        ref_data = pickle.load(f2)
        mean = ref_data['mean']
        std = ref_data['std']

    # load model
    pipe = MTVCrafterPipeline.from_pretrained(
        model_path='THUDM-CogVideoX-5b',
        transformer_model_path = 'MTVCrafter/transformer_epoch_100',
        torch_dtype=torch.bfloat16,
        scheduler_type='dpm',
    ).to(device)
    params = [(name, param) for name, param in pipe.transformer.named_parameters()]

    # load vqvae
    state_dict = torch.load("VQVAE/mp_rank_00_model_states.pt", map_location="cpu")
    motion_encoder = Encoder(in_channels=3, mid_channels=[128, 512], out_channels=3072, downsample_time=[2, 2], downsample_joint=[1, 1])
    motion_quant = VectorQuantizer(nb_code=8192, code_dim=3072, is_train=False)
    motion_decoder = Decoder(in_channels=3072, mid_channels=[512, 128], out_channels=3, upsample_rate=2.0, frame_upsample_rate=[2.0, 2.0], joint_upsample_rate=[1.0, 1.0])
    vqvae = SMPL_VQVAE(motion_encoder, motion_decoder, motion_quant).to(device)
    print(vqvae.load_state_dict(state_dict['module'], strict=True))

    # inference
    for index, data_dict in enumerate(tqdm(data)):
        new_height, new_width = get_new_height_width(data_dict, dst_height, dst_width)
        print("new:", new_height, new_width)
        x1 = (new_width - dst_width) // 2
        y1 = (new_height - dst_height) // 2
        sample_indexes = get_sample_indexes(data_dict['video_length'], num_frames, 1)
        print("sample_indexes:", sample_indexes)

        input_images = sample_video(decord.VideoReader(data_dict['video_path']), sample_indexes, method=2)
        input_images = torch.from_numpy(input_images).permute(0, 3, 1, 2).contiguous()
        input_images = F.resize(input_images, (new_height, new_width), InterpolationMode.BILINEAR)
        input_images = F.crop(input_images, y1, x1, dst_height, dst_width)

        if ref_image_path != '':
            ref_image = Image.open(ref_image_path).convert("RGB")
            ref_image = torch.from_numpy(np.array(ref_image)).permute(2, 0, 1).contiguous()
            ref_images = torch.stack([ref_image.clone() for _ in range(num_frames)]) 
            ref_images = F.resize(ref_images, (new_height, new_width), InterpolationMode.BILINEAR)
            ref_images = F.crop(ref_images, y1, x1, dst_height, dst_width)
        else:
            ref_images = copy.deepcopy(input_images)
            frame0 = input_images[0]
            ref_images[:, :, :, :] = frame0

        input_images = input_images / 255.0
        ref_images = ref_images / 255.0

        input_images = normalize(input_images)
        ref_images = normalize(ref_images)

        height, width = data_dict['video_height'], data_dict['video_width']
        offset = [height, width, 0]

        poses = data_dict['pose'][sample_indexes]
        poses = torch.tensor((poses - mean) / std)
        pose_images_before = get_pose_images(copy.deepcopy(poses), offset, mean, std)
        pose_images_before = [image.resize((new_width, new_height)).crop((x1, y1, x1+dst_width, y1+dst_height)) for image in pose_images_before]
        input_smpl_joints = poses.unsqueeze(0).to(device)
        motion_tokens, vq_loss = vqvae(input_smpl_joints, return_vq=True)
        print(f"vq loss: {vq_loss}")
        output_motion, _ =  vqvae(input_smpl_joints)
        pose_images_after = get_pose_images(output_motion[0].cpu().detach(), offset, mean, std)
        pose_images_after = [image.resize((new_width, new_height)).crop((x1, y1, x1+dst_width, y1+dst_height)) for image in pose_images_after]

        # inference
        guidance_scale = 3.0
        output_images = pipe(
            height=dst_height,
            width=dst_width,
            num_frames=num_frames,
            num_inference_steps=50,
            guidance_scale=guidance_scale,
            seed=6666,
            ref_images=ref_images,
            motion_embeds=motion_tokens,
            joint_mean=mean,
            joint_std=std,
        ).frames[0]

        # save results
        save_path = f"{output_dir}/{ref_image_path.split('/')[-1].split('.png')[0]}_data_{index}_guidance_{guidance_scale}.mp4"
        vis_images = []
        for k in range(len(pose_images_before)):
            vis_image = [to_pil(((ref_images[k] + 1) * 127.5).clamp(0, 255).to(torch.uint8)), to_pil(((input_images[k] + 1) * 127.5).clamp(0, 255).to(torch.uint8)), pose_images_before[k], pose_images_after[k], output_images[k]]
            vis_image = concat_images_grid(vis_image, cols=len(vis_image), pad=2)
            vis_images.append(vis_image)
        imageio.mimsave(save_path, vis_images, fps=20)
        print(f"save data {index} to {save_path}")


def main():
    data_path = 'data/sample_data.pkl'
    ref_image_path = ''
    # ref_image_path = '/ref_images/human.png'
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--ref_image_path', type=str, default="", required=False, help="path to the reference character image")
    parser.add_argument('--motion_data_path', type=str, default="data/sample_data.pkl", required=False, help='path to the motion sequence')
    parser.add_argument('--output_path', type=str, default="inference_output", required=False, help="where to save the generated video")
    
    args = parser.parse_args()
    inference(device='cuda:0', motion_data_path=args.motion_data_path, ref_image_path=args.ref_image_path, output_dir=args.output_path)


if __name__ == '__main__':
    main()

