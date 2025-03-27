import argparse
from pathlib import Path
import torch
import os
import random
import numpy as np
import torchaudio
import math

from genmo.lib.utils import save_video
from genmo.mochi_preview.pipelines import (
    DecoderModelFactory, DitModelFactory, MochiSingleGPUPipeline,
    T5ModelFactory, linear_quadratic_schedule,
)
from moviepy.editor import VideoFileClip, AudioFileClip
from transformers import Wav2Vec2Processor, Wav2Vec2Model

def set_seed(seed=33):
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

def extract_audio_from_video(video_path, output_dir):
    video_filename = os.path.basename(video_path).split('.')[0]
    output_path = os.path.join(output_dir, f"{video_filename}.mp3")
    try:
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(output_path)
        duration = video.duration
        video.close()
        return output_path, duration
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None, 0

def save_cropped_audio(waveform, sample_rate, start_idx, length, output_path):
    cropped_waveform = waveform[:, start_idx:start_idx + length]
    # Create temp file in output directory to avoid writing to current directory
    output_dir = os.path.dirname(output_path)
    temp_filename = os.path.join(output_dir, f"temp_{os.urandom(4).hex()}.wav")
    
    try:
        torchaudio.save(temp_filename, cropped_waveform, sample_rate, format="wav")
        audio_clip = AudioFileClip(temp_filename)
        audio_clip.write_audiofile(output_path, fps=sample_rate, nbytes=4, bitrate="320k")
        audio_clip.close()
        os.remove(temp_filename)
        return True
    except Exception as e:
        print(f"Error saving cropped audio: {str(e)}")
        if os.path.exists(temp_filename):
            os.remove(temp_filename)
        return False

def process_audio_wav2vec(audio_path, duration, output_dir, use_cuda=True, save_crop=True):
    print(f"Processing audio from {audio_path}")
    processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
    model = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-base-960h")
    if use_cuda and torch.cuda.is_available():
        model = model.cuda()
    
    original_waveform, original_sample_rate = torchaudio.load(audio_path)
    original_desired_length = math.ceil(original_sample_rate * duration)
    crop_start_idx = 0
    crop_saved = False
    input_filename = os.path.basename(audio_path).split('.')[0]
    
    if original_waveform.shape[1] > original_desired_length:
        max_start_idx = original_waveform.shape[1] - original_desired_length
        crop_start_idx = random.randint(0, max_start_idx)
        if save_crop:
            crop_output_path = os.path.join(output_dir, f"{input_filename}_cropped_hq.mp3")
            cropped_original = original_waveform[:, crop_start_idx:crop_start_idx + original_desired_length]
            temp_wav_path = os.path.join(output_dir, f"temp_{input_filename}_hq.wav")
            try:
                torchaudio.save(temp_wav_path, cropped_original, original_sample_rate, format="wav")
                audio_clip = AudioFileClip(temp_wav_path)
                audio_clip.write_audiofile(crop_output_path, fps=original_sample_rate, nbytes=4, bitrate="320k")
                audio_clip.close()
                os.remove(temp_wav_path)
                crop_saved = True
            except Exception as e:
                print(f"Error saving high-quality crop: {str(e)}")
    
    if original_sample_rate != 16000:
        waveform = torchaudio.transforms.Resample(original_sample_rate, 16000)(original_waveform)
    else:
        waveform = original_waveform    
    
    if waveform.shape[0] > 1:
        waveform = waveform[0:1]
    
    desired_length = math.ceil(16000 * duration)
    
    if original_waveform.shape[1] > original_desired_length:
        resampled_start_idx = int((crop_start_idx / original_sample_rate) * 16000)
        resampled_end_idx = min(resampled_start_idx + desired_length, waveform.shape[1])
        if save_crop and not crop_saved:
            crop_output_path = os.path.join(output_dir, f"{input_filename}_cropped_lq.mp3")
            save_cropped_audio(waveform, 16000, resampled_start_idx, desired_length, crop_output_path)
        waveform = waveform[:, resampled_start_idx:resampled_end_idx]
        if waveform.shape[1] < desired_length:
            waveform = torch.nn.functional.pad(waveform, (0, desired_length - waveform.shape[1]))
    else:
        waveform = torch.nn.functional.pad(waveform, (0, desired_length - waveform.shape[1]))
    
    inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt")
    if use_cuda and torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    return outputs.last_hidden_state

def process_audio(input_path, num_frames, output_dir):
    input_ext = os.path.splitext(input_path)[1].lower()
    if input_ext in ['.mp4', '.mov']:
        audio_path, _ = extract_audio_from_video(input_path, output_dir)
        if audio_path is None:
            raise ValueError(f"Failed to extract audio from {input_path}")
    elif input_ext in ['.mp3', '.wav']:
        # Copy audio file to output directory if needed
        audio_filename = os.path.basename(input_path)
        audio_path_in_output = os.path.join(output_dir, audio_filename)
        if input_path != audio_path_in_output:
            audio_clip = AudioFileClip(input_path)
            audio_clip.write_audiofile(audio_path_in_output)
            audio_clip.close()
            audio_path = audio_path_in_output
        else:
            audio_path = input_path
    else:
        raise ValueError(f"Unsupported file format: {input_ext}")
    
    duration = num_frames / 30 + 0.09
    audio_embeddings = process_audio_wav2vec(
        audio_path, duration=duration, output_dir=output_dir,
        use_cuda=torch.cuda.is_available(), save_crop=True
    )
    
    input_filename = os.path.basename(input_path).split('.')[0]
    cropped_audio_path_hq = os.path.join(output_dir, f"{input_filename}_cropped_hq.mp3")
    cropped_audio_path_lq = os.path.join(output_dir, f"{input_filename}_cropped_lq.mp3")
    
    if os.path.exists(cropped_audio_path_hq):
        audio_path = cropped_audio_path_hq
    elif os.path.exists(cropped_audio_path_lq):
        audio_path = cropped_audio_path_lq
    
    return audio_embeddings, audio_path

def save_video_with_audio(video_frames, output_path, audio_path):
    output_dir = os.path.dirname(output_path)
    temp_video_path = os.path.join(output_dir, f"{os.path.basename(output_path).replace('.mp4', '_temp.mp4')}")
    temp_audio_path = os.path.join(output_dir, "temp-audio.m4a")
    
    save_video(video_frames, temp_video_path)
    
    try:
        video = VideoFileClip(temp_video_path)
        audio = AudioFileClip(audio_path)
        
        if audio.duration > video.duration:
            audio = audio.subclip(0, video.duration)
        
        video.set_audio(audio).write_videofile(
            str(output_path), 
            codec='libx264',
            audio_codec='aac',
            temp_audiofile=temp_audio_path,
            remove_temp=True
        )
        
        video.close()
        audio.close()
        os.remove(temp_video_path)
        
        # Ensure temp audio is removed if it still exists
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)
            
    except Exception as e:
        print(f"Error processing video with audio: {str(e)}")
        if os.path.exists(temp_video_path):
            os.rename(temp_video_path, output_path)

def create_pipeline(Tempo_model_path, mochi_dir):
    if Tempo_model_path == 'base':
        return MochiSingleGPUPipeline(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(model_path=f"{mochi_dir}/dit.safetensors", model_dtype="bf16"),
            decoder_factory=DecoderModelFactory(model_path=f"{mochi_dir}/decoder.safetensors"),
            cpu_offload=True,
            decode_type="full",
        )
    else:
        return MochiSingleGPUPipeline(
            text_encoder_factory=T5ModelFactory(),
            dit_factory=DitModelFactory(
                model_path=f"{mochi_dir}/dit.safetensors", 
                lora_path=f"{Tempo_model_path}", 
                model_dtype="bf16",
                audio_mode="cross_attn"
            ),
            decoder_factory=DecoderModelFactory(model_path=f"{mochi_dir}/decoder.safetensors"),
            cpu_offload=True,
            decode_type="full",
            fast_init=False,
            strict_load=False,
        )

def run_inference(args):
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    
    try:
        audio_embedding, audio_path = process_audio(args.input_file, args.num_frames, args.output_dir)
        pipeline = create_pipeline(args.Tempo_model_path, args.mochi_dir)
        
        video = pipeline(
            height=480,
            width=848,
            num_frames=args.num_frames,
            num_inference_steps=64,
            sigma_schedule=linear_quadratic_schedule(64, 0.025),
            cfg_schedule=[args.cfg_scale] * 64,
            batch_cfg=False,
            prompt=args.prompt,
            negative_prompt="",
            seed=args.seed,
            audio_feat=audio_embedding.cuda(),
        )
        
        base_filename = os.path.basename(args.input_file).split('.')[0]
        output_path = output_dir / f"{base_filename}_generated.mp4"
        
        save_video_with_audio(video[0], str(output_path), audio_path)
        print(f"✓ Generated video saved to: {output_path}")
            
    except Exception as e:
        print(f"✗ Error processing {args.input_file}: {str(e)}")
        import traceback
        traceback.print_exc()

def main():
    parser = argparse.ArgumentParser(description="Mochi dancer generation with audio input")
    parser.add_argument("--Tempo_model-path", type=str, 
                        default="weights/Tempo_model.safetensors", 
                        help="Path to the Tempo_model model or 'base' for base model")
    parser.add_argument("--mochi-dir", type=str, 
                        default="weights",
                        help="Directory with Mochi model weights")
    parser.add_argument("--output-dir", type=str, 
                        default="./outputs",
                        help="Output directory for all generated files")
    parser.add_argument("--input-file", type=str,
                        default="example.mp4",
                        help="Input file (MP3 or MP4) to extract audio from")
    parser.add_argument("--prompt", type=str, 
                        default="a professional female dancer dancing K-pop in an advanced dance setting in a studio with a white background, captured from a front view",
                        help="Prompt for dance video generation")
    parser.add_argument("--cfg-scale", type=float, 
                        default=6.0,
                        help="Classifier-Free Guidance (CFG) scale")
    parser.add_argument("--num-frames", type=int, 
                        default=145,
                        help="Number of frames to generate")
    parser.add_argument("--seed", type=int, 
                        default=None,
                        help="Random seed")
    
    args = parser.parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    run_inference(args)

if __name__ == "__main__":
    main()
