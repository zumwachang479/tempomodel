'''
MIT License

Copyright (c) 2023 OpenGVLab

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''

import os
import torch
import numpy as np
from decord import VideoReader, cpu
import decord
from easydict import EasyDict
from transformers import AutoModel, StoppingCriteria, StoppingCriteriaList
from torchvision import transforms
import argparse

import os
import torch
import numpy as np
from decord import VideoReader, cpu
import decord
from easydict import EasyDict
from transformers import AutoModel
import argparse
from torch.multiprocessing import Process, set_start_method
import math


decord.bridge.set_bridge("torch")

def HD_transform_padding(frames, image_size=224, hd_num=6):
    def _padding_224(frames):
        _, _, H, W = frames.shape
        tar = int(np.ceil(H / 224) * 224)
        top_padding = (tar - H) // 2
        bottom_padding = tar - H - top_padding
        left_padding = 0
        right_padding = 0

        padded_frames = torch.nn.functional.pad(
            frames, 
            pad=[left_padding, right_padding, top_padding, bottom_padding], 
            mode='constant', value=255
        )
        return padded_frames

    _, _, H, W = frames.shape
    trans = False
    if W < H:
        frames = frames.flip(-2, -1)
        trans = True
        width, height = H, W
    else:
        width, height = W, H

    ratio = width / height
    scale = 1
    while scale * np.ceil(scale / ratio) <= hd_num:
        scale += 1
    scale -= 1
    new_w = int(scale * image_size)
    new_h = int(new_w / ratio)

    resized_frames = torch.nn.functional.interpolate(
        frames, size=(new_h, new_w), 
        mode='bicubic', 
        align_corners=False
    )
    padded_frames = _padding_224(resized_frames)

    if trans:
        padded_frames = padded_frames.flip(-2, -1)

    return padded_frames

def find_closest_aspect_ratio(aspect_ratio, target_ratios, width, height, image_size):
    best_ratio_diff = float('inf')
    best_ratio = (1, 1)
    area = width * height
    for ratio in target_ratios:
        target_aspect_ratio = ratio[0] / ratio[1]
        ratio_diff = abs(aspect_ratio - target_aspect_ratio)
        if ratio_diff < best_ratio_diff:
            best_ratio_diff = ratio_diff
            best_ratio = ratio
        elif ratio_diff == best_ratio_diff:
            if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                best_ratio = ratio
    return best_ratio

def HD_transform_no_padding(frames, image_size=224, hd_num=6):
    min_num = 1
    max_num = hd_num
    _, _, orig_height, orig_width = frames.shape
    aspect_ratio = orig_width / orig_height

    target_ratios = set(
        (i, j) for n in range(min_num, max_num + 1) 
        for i in range(1, n + 1) 
        for j in range(1, n + 1) 
        if i * j <= max_num and i * j >= min_num
    )
    target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

    target_aspect_ratio = find_closest_aspect_ratio(
        aspect_ratio, target_ratios, orig_width, orig_height, image_size
    )

    target_width = image_size * target_aspect_ratio[0]
    target_height = image_size * target_aspect_ratio[1]

    resized_frame = torch.nn.functional.interpolate(
        frames, size=(target_height, target_width), 
        mode='bicubic', align_corners=False
    )
    return resized_frame

def get_prompt(conv):
    ret = conv.system + conv.sep
    for role, message in conv.messages:
        if message:
            ret += role + " " + message + " " + conv.sep
        else:
            ret += role
    return ret


def get_prompt2(conv):
    ret = conv.system + conv.sep
    count = 0
    for role, message in conv.messages:
        count += 1
        if count == len(conv.messages):
            ret += role + " " + message
        else:
            if message:
                ret += role + " " + message + " " + conv.sep
            else:
                ret += role
    return ret


def get_context_emb(gpu_id, conv, model, img_list, answer_prompt=None, print_res=False):
    if answer_prompt:
        prompt = get_prompt2(conv)
    else:
        prompt = get_prompt(conv)
    if print_res:
        print(prompt)
    if '<VideoHere>' in prompt:
        prompt_segs = prompt.split('<VideoHere>')
    else:
        prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + 1, "Unmatched numbers of image placeholders and images."
    with torch.no_grad():
        seg_tokens = [
            model.mistral_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(f"cuda:{gpu_id}").input_ids
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.mistral_model.model.embed_tokens(seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list) for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


def ask(text, conv):
    conv.messages.append([conv.roles[0], text])

def get_index(num_frames, num_segments):
    seg_size = float(num_frames - 1) / num_segments
    start = int(seg_size / 2)
    offsets = np.array([
        start + int(np.round(seg_size * idx)) for idx in range(num_segments)
    ])
    return offsets


def load_video(video_path, num_segments=8, return_msg=False, resolution=224, hd_num=6, padding=False):
    vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
    num_frames = len(vr)
    frame_indices = get_index(num_frames, num_segments)

    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)

    transform = transforms.Compose([
        transforms.Lambda(lambda x: x.float().div(255.0)),
        transforms.Normalize(mean, std)
    ])

    frames = vr.get_batch(frame_indices)
    frames = frames.permute(0, 3, 1, 2)

    if padding:
        frames = HD_transform_padding(frames.float(), image_size=resolution, hd_num=hd_num)
    else:
        frames = HD_transform_no_padding(frames.float(), image_size=resolution, hd_num=hd_num)

    frames = transform(frames)
    print(frames.shape)
    
    if return_msg:
        fps = float(vr.get_avg_fps())
        sec = ", ".join([str(round(f / fps, 1)) for f in frame_indices])
        msg = f"The video contains {len(frame_indices)} frames sampled at {sec} seconds."
        return frames, msg
    else:
        return frames

def get_sinusoid_encoding_table(n_position=784, d_hid=1024, cur_frame=8, ckpt_num_frame=4, pre_n_position=784): 
    sinusoid_table = np.array([[(pos_i / np.power(10000, 2 * (hid_j // 2) / d_hid)) for hid_j in range(d_hid)] for pos_i in range(pre_n_position)]) 
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])
    sinusoid_table = torch.tensor(sinusoid_table, dtype=torch.float, requires_grad=False).unsqueeze(0)
    
    print(f"n_position: {n_position}")
    print(f"pre_n_position: {pre_n_position}")
    
    if n_position != pre_n_position:
        T = ckpt_num_frame
        P = 14
        C = d_hid
        new_P = int((n_position // cur_frame) ** 0.5)
        if new_P != 14:
            print(f'Pretraining uses 14x14, but current version is {new_P}x{new_P}')
            print(f'Interpolate the position embedding')
            sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
            sinusoid_table = sinusoid_table.reshape(-1, P, P, C).permute(0, 3, 1, 2)
            sinusoid_table = torch.nn.functional.interpolate(
                sinusoid_table, size=(new_P, new_P), mode='bicubic', align_corners=False)
            sinusoid_table = sinusoid_table.permute(0, 2, 3, 1).reshape(-1, T, new_P, new_P, C)
            sinusoid_table = sinusoid_table.flatten(1, 3)
    
    if cur_frame != ckpt_num_frame:
        print(f'Pretraining uses 4 frames, but current frame is {cur_frame}')
        print(f'Interpolate the position embedding')
        T = ckpt_num_frame
        new_T = cur_frame
        P = int((n_position // cur_frame) ** 0.5)
        C = d_hid
        sinusoid_table = sinusoid_table.reshape(-1, T, P, P, C)
        sinusoid_table = sinusoid_table.permute(0, 2, 3, 4, 1).reshape(-1, C, T)
        sinusoid_table = torch.nn.functional.interpolate(sinusoid_table, size=new_T, mode='linear')
        sinusoid_table = sinusoid_table.reshape(1, P, P, C, new_T).permute(0, 4, 1, 2, 3)
        sinusoid_table = sinusoid_table.flatten(1, 3)
        
    return sinusoid_table


class StoppingCriteriaSub(StoppingCriteria):
    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True
        return False
    
    
def answer(gpu_id, conv, model, img_list, do_sample=True, max_new_tokens=200, num_beams=1, min_length=1, top_p=0.9,
               repetition_penalty=1.0, length_penalty=1, temperature=1.0, answer_prompt=None, print_res=False):
    stop_words_ids = [
        torch.tensor([2]).to(f"cuda:{gpu_id}"),
        torch.tensor([29871, 2]).to(f"cuda:{gpu_id}")]
    stopping_criteria = StoppingCriteriaList([StoppingCriteriaSub(stops=stop_words_ids)])
    
    conv.messages.append([conv.roles[1], answer_prompt])
    embs = get_context_emb(gpu_id, conv, model, img_list, answer_prompt=answer_prompt, print_res=print_res)
    with torch.no_grad():
        outputs = model.mistral_model.generate(
            inputs_embeds=embs,
            max_new_tokens=max_new_tokens,
            stopping_criteria=stopping_criteria,
            num_beams=num_beams,
            do_sample=do_sample,
            min_length=min_length,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
            length_penalty=length_penalty,
            temperature=temperature,
        )
    output_token = outputs[0]
    if output_token[0] == 0:
            output_token = output_token[1:]
    if output_token[0] == 1:
            output_token = output_token[1:]
    output_text = model.mistral_tokenizer.decode(output_token, add_special_tokens=False)
    output_text = output_text.split('</s>')[0]
    conv.messages[-1][1] = output_text + '</s>'
    return output_text, output_token.cpu().numpy()


def setup_model(gpu_id, resolution=224, num_frame=16):
    device = f"cuda:{gpu_id}"
    model = AutoModel.from_pretrained(
        "OpenGVLab/VideoChat2_HD_stage4_Mistral_7B_hf",
        trust_remote_code=True
    ).to(device)
    
    new_pos_emb = get_sinusoid_encoding_table(
        n_position=(resolution//16)**2*num_frame,
        cur_frame=num_frame
    )
    model.vision_encoder.encoder.pos_embed = new_pos_emb
    
    return model

from tqdm import tqdm

def process_video_batch(gpu_id, video_files, args):
    device = f"cuda:{gpu_id}"
    model = setup_model(gpu_id, args.resolution)
    
    for video_path in tqdm(video_files):
        filename = os.path.basename(video_path)
        output_path = os.path.join(args.directory, filename[:-4] + '.txt')
        
        print(f"Processing {filename} on GPU {gpu_id}...")
        try:
            chat = EasyDict({
                "system": "",
                "roles": ("[INST]", "[/INST]"),
                "messages": [],
                "sep": ""
            })
            
            vid, msg = load_video(
                video_path, 
                num_segments=args.num_frame,
                return_msg=True,
                resolution=args.resolution,
                hd_num=args.hd_num,
                padding=args.padding
            )
            
            T_, C, H, W = vid.shape
            video = vid.reshape(1, T_, C, H, W).to(device)
            
            img_list = []
            with torch.no_grad():
                image_emb, _, _ = model.encode_img(video, "Watch the video and answer the question.")
            img_list.append(image_emb[0])
            
            chat.messages.append([chat.roles[0], "<Video><VideoHere></Video> [/INST]"])
            ask("Describe the video in details.", chat)
            
            description = answer(
                gpu_id=gpu_id,
                conv=chat,
                model=model,
                do_sample=False,
                img_list=img_list,
                max_new_tokens=512,
                print_res=True
            )[0]
            print("### Desc:", description)
            
            with open(output_path, 'w') as f:
                f.write(description)
            print(f"Description saved to {output_path}")
            
        except Exception as e:
            print(f"Error processing {filename} on GPU {gpu_id}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Generate descriptions for MP4 videos using multiple GPUs')
    parser.add_argument('directory', type=str, help='Directory containing MP4 files')
    parser.add_argument('--padding', action='store_true', help='Use padding in HD transformation')
    parser.add_argument('--hd-num', type=int, default=12, help='HD transformation number')
    parser.add_argument('--resolution', type=int, default=224, help='Target resolution')
    parser.add_argument('--num-frame', type=int, default=16, help='Number of frames')
    parser.add_argument('--gpus', type=int, nargs='+', default=[0], help='GPU IDs to use')
    args = parser.parse_args()

    video_files = [
        os.path.join(args.directory, f)
        for f in os.listdir(args.directory)
        if f.endswith('.mp4') and not f.endswith('.recon.mp4')
    ]
    
    num_gpus = len(args.gpus)
    files_per_gpu = math.ceil(len(video_files) / num_gpus)
    file_batches = [
        video_files[i:i + files_per_gpu] 
        for i in range(0, len(video_files), files_per_gpu)
    ]
    
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
        
    processes = []
    for gpu_id, batch in zip(args.gpus, file_batches):
        p = Process(
            target=process_video_batch,
            args=(gpu_id, batch, args)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()

if __name__ == "__main__":
    main()
