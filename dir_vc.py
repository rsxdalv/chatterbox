import os
import subprocess
import argparse
import sys
import shutil
import re
import datetime
import math
import shlex
from pydub import AudioSegment

import torch
import torchaudio as ta

from chatterbox.vc import ChatterboxVC

RVC_PATH = os.path.dirname(os.path.realpath(__file__))
LOG_FILE = os.path.join(RVC_PATH, "run_infer.log")
log_file = None

# (Optional) Duration constraints—you may use them later if needed.
MIN_SEGMENT_DURATION = 120  # 2 minutes
MAX_SEGMENT_DURATION = 180  # 3 minutes

def log_message(message, level="INFO"):
    global log_file
    if log_file is None:
        log_file = LOG_FILE
        print(f"[WARNING] using default log file: {log_file}")
    with open(log_file, "a", encoding='utf-8') as log_fd:
        log_fd.write(f"[{level}] {message}\n")
        
def safe_print(message):
    """Prints a message to the console, ensuring it is safe for all environments."""
    try:
        print(message)
    except UnicodeEncodeError:
        sys.stdout.reconfigure(encoding='utf-8')
        # If there's an encoding error, replace problematic characters
        print(message.encode('utf-8', errors='replace').decode('utf-8', errors='replace'))

def get_audio_detail(input_file):
    """_summary_

    Args:
        input_file (_str_): path of audio file (wav or mp3)

    Returns:
        tuple: (duration: float, sample_rate: int, channel_str: str)
        duration: unit second
        sample_rate: unit Hz
        channel_str: mono, stereo, 5.1, 7.1, unknown
    """
    #using ffprobe to get the duration of the audio file
    detail_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 
        # ffprobe use it's own order. even you write duration,sample_rate,channels, it still return data in the order of sample_rate, channels, duration
        'stream=sample_rate,channels,duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    try:
        output = subprocess.run(detail_cmd, stdout=subprocess.PIPE, text=True, check=True, encoding='utf-8').stdout.strip()
        sample_rate, channel, duration = output.split('\n')

        #set channel str
        # mono if 1
        # stereo if 2
        # 5.1 if 6
        # 7.1 if 8
        channel_str = 'mono' if channel == '1' else 'stereo' if channel == '2' else '5.1' if channel == '6' else '7.1' if channel == '8' else 'unknown'

        log_message(f"get_audio_detail {input_file} => \n\tduration: {duration}, \n\tsample_rate: {sample_rate}, \n\tchannel: {channel}")

        return float(duration), int(sample_rate), channel_str
    except (subprocess.CalledProcessError, ValueError) as e:
        return -1.0, -1, 'unknown'

def append_silence(input_file: str, append_duration: float, sample_rate: int, channel: str, continue_job) -> bool:
    """
    Append silence to the end of the input audio file.

    Args:
        input_file (str): input audio file path
        append_duration (float): duration of silence to append in seconds
        sample_rate (int): sample rate of the audio file
        channel (str): channel of the audio file
    Returns:
        str: output file path (if fail will return same path as input_file)
    """
    log_message(f'append silence to {input_file} by {append_duration} seconds')
    # output file name = input file name appended with '_append'
    output_file = input_file
    if '.wav' in input_file:
        output_file = input_file.rsplit('.wav', 1)[0] + '_append.wav'
        
    #if output_file already exists, skip it
    if continue_job and os.path.exists(output_file):
        log_message(f"Skipping appending silence to {input_file} because {output_file} already exists", level='WARNING')
        safe_print(f"[WARNING] Skipping appending silence to {input_file} because {output_file} already exists")
        return output_file
    
    log_message(f'append silence to {input_file} by {append_duration} seconds')
    ffmpeg_silence_cmd = [
        'ffmpeg', '-y', '-i', input_file, '-f', 'lavfi', '-t', str(append_duration), '-i', f'anullsrc=channel_layout={channel}:sample_rate={sample_rate}', '-filter_complex', '[0][1]concat=n=2:v=0:a=1', output_file
    ]
    try: 
        subprocess.run(ffmpeg_silence_cmd, check=True, encoding='utf-8', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        log_message(f"[Error] appending silence: {e}")
        return input_file

    return output_file

# Practical Examples
# Setting	Expected Behavior
# -40dB, d=0.5	Very aggressive: Cuts on very small gaps, even if there’s low noise.
# -30dB, d=1	Moderate (your current setting): Good for clean recordings, but can overcut in noisy ones.
# -25dB, d=2	More tolerant: Best if your audio has background hum or short pauses.
# -20dB, d=3	Very tolerant: Cuts only on long, obvious silence.
def split_audio_into_segments(input_file: str, temp_dir: str, noise_threshold=-20, silence_duration=1.5, continue_job=False, padding_sec=0.1):
    # Get total duration of the input file.
    duration_cmd = [
        'ffprobe', '-v', 'error', '-show_entries', 'format=duration',
        '-of', 'default=noprint_wrappers=1:nokey=1', input_file
    ]
    total_duration = float(subprocess.run(duration_cmd, stdout=subprocess.PIPE, text=True, check=True).stdout.strip())
    log_message(f'BEGIN file => {input_file}')
    log_message(f'\t=> total_duration: {total_duration}')

    """
    Splits the input audio file into segments labeled as 'silence' and 'non_silence'
    using ffmpeg's silencedetect filter.
    
    Returns a list of tuples: (segment_file, segment_type, start_time, end_time)
    """
    # Run ffmpeg silencedetect to get silence intervals.
    silence_cmd = [
        'ffmpeg', '-i', input_file, '-af',
        f'silencedetect=noise={noise_threshold}dB:d={silence_duration}', '-f', 'null', '-'
    ]

    result = subprocess.run(silence_cmd, stderr=subprocess.PIPE, text=True, check=True, encoding='utf-8')
    
    # Parse the stderr output to get pairs of silence_start and silence_end.
    silence_intervals = []
    current_silence_start = None
    for line in result.stderr.splitlines():
        if 'silence_start' in line:
            match = re.search(r'silence_start: (\d+\.?\d*)', line)
            if match:
                current_silence_start = float(match.group(1))
        if 'silence_end' in line and current_silence_start is not None:
            match = re.search(r'silence_end: (\d+\.?\d*)', line)
            if match:
                silence_end = float(match.group(1))

                real_silence_start = current_silence_start + padding_sec
                if(real_silence_start < 0 ):
                    real_silence_start = 0
                real_silence_end = silence_end - padding_sec
                if(real_silence_end > total_duration):
                    real_silence_end = total_duration

                silence_intervals.append((real_silence_start, real_silence_end))
                current_silence_start = None

    # Create segments: non-silence segments are between silence intervals.
    segments = []  # Each element is (start_time, end_time, segment_type)
    current_time = 0.0
    log_message(f'BEGIN RAW silence_intervals')
    for (silence_start, silence_end) in silence_intervals:
        log_message(f'{silence_start}\t{silence_end}')
        adjust_silence_start = silence_start
        adjust_silence_end = silence_end
        #adjust silence_start and silence_end
        # 1. precision only to 1 decimal place
        # 2. adjust_silence_start will be greater or equal to silence_start
        # 3. adjust_silence_end will be less or equal to silence_end
        # adjust_silence_start = round(silence_start, 1)
        # adjust_silence_end = round(silence_end, 1)
        # if adjust_silence_start < silence_start:
        #     adjust_silence_start += 0.1
        # if adjust_silence_end > silence_end:
        #     adjust_silence_end -= 0.1
        # If there is audio before this silence, add it as a non-silence segment.
        if adjust_silence_start > current_time:
            segments.append((current_time, adjust_silence_start, 'non_silence'))
        # Add the silence segment.
        segments.append((adjust_silence_start, adjust_silence_end, 'silence'))
        current_time = adjust_silence_end
    log_message(f'END RAW silence_intervals')
    # If there is audio after the last silence, add it.
    if current_time < total_duration:
        segments.append((current_time, total_duration, 'non_silence'))

    # Extract each segment to its own file.
    extracted_segments = []
    log_message(f'BEGIN segments')
    for idx, (start, end, seg_type) in enumerate(segments):
        log_message(f'{start}\t{end}\t{seg_type}')
        duration = end - start
        if(duration < 0.05):
            safe_print(f"[WARNING] Skipping short segment of file {input_file}:\n\t {start}-{end} ({duration}s)")
            log_message(f"[WARNING] Skipping short segment of file {input_file}:\n\t {start}-{end} ({duration}s)")
            continue
        segment_filename = os.path.join(temp_dir, f"{idx:04d}_{seg_type}_{start}_{end}.wav")

        #if segment_filename already exists, skip it
        if continue_job and os.path.exists(segment_filename):
            extracted_segments.append((segment_filename, seg_type, start, end))
            safe_print(f"[WARNING] Skipping segment {segment_filename} because it already exists")
            continue
        ffmpeg_extract_cmd = [
            'ffmpeg', '-y', '-i', input_file, '-ss', str(start), '-t', str(duration),
            '-c', 'copy', segment_filename
        ]
        subprocess.run(ffmpeg_extract_cmd, check=True, encoding='utf-8', stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        extracted_segments.append((segment_filename, seg_type, start, end))
    log_message(f'END segments')
    
    return extracted_segments

def process_audio_files(input_dir, output_dir, target_voice_path, continue_job=False):
    supported_extensions = ('.wav', '.mp3')
    venv_python = sys.executable

    # Create a temporary directory to hold segments.
    temp_dir = os.path.join(output_dir, "temp_segments")
    os.makedirs(temp_dir, exist_ok=True)

    half_true = True  # Use half precision for faster processing if supported.

    # Automatically detect the best available device
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    print(f"Using device: {device}")
    model = ChatterboxVC.from_pretrained(device)

    # Process each file in the input directory.
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(supported_extensions):
            input_file = os.path.join(input_dir, filename)
            base_name, _ = os.path.splitext(filename)
            
            # Create a subdirectory for segments from this file.
            file_temp_dir = os.path.join(temp_dir, base_name)
            os.makedirs(file_temp_dir, exist_ok=True)
            
            # Split the file into silence and non-silence segments.
            segments = split_audio_into_segments(
                input_file, file_temp_dir, noise_threshold=-20, silence_duration=1.5, continue_job=continue_job, padding_sec=0.1
            )
            
            # Process each segment accordingly.
            # Batch process all non-silence segments at once using the new CLI
            env_python = sys.executable
            processed_dir = os.path.join(file_temp_dir, "processed_segments")
            os.makedirs(processed_dir, exist_ok=True)
            # Move all silence-only segments directly into processed_dir
            for filename in os.listdir(file_temp_dir):
                if '_silence' in filename and '_non_silence' not in filename:
                    src = os.path.join(file_temp_dir, filename)
                    dst = os.path.join(processed_dir, filename)
                    shutil.move(src, dst)
            
            print(f"Converting to target voice: {target_voice_path}")

            # model processing
            for file in os.listdir(file_temp_dir):
                if file.endswith(".wav"):
                    wav = model.generate(
                        audio=os.path.join(file_temp_dir, file),
                        target_voice_path=target_voice_path,
                    )
                    output_path = os.path.join(processed_dir, file)
                    ta.save(output_path, wav, model.sr)
                    #print(f"Saved converted audio to: {output_path}")

            #safe_print(f"renaming .wav.wav to .wav in {processed_dir}")
            # Fix double .wav extension from infer_cli_dir outputs
            for fname in os.listdir(processed_dir):
                if fname.endswith('.wav.wav'):
                    src = os.path.join(processed_dir, fname)
                    dst = os.path.join(processed_dir, fname[:-4])
                    shutil.move(src, dst)

            # Build final segment list, copying silence segments as-is
            final_segments = []
            for seg_file, seg_type, start, end in segments:
                basename = os.path.basename(seg_file)
                processed_seg = os.path.join(processed_dir, basename)
                if seg_type == 'silence':
                    continue
                final_segments.append((processed_seg, start))

            # Reorder segments by their original start times.
            final_segments_sorted = sorted(final_segments, key=lambda x: x[1])
            
            # Write a concat file for ffmpeg.
            concat_list = os.path.join(file_temp_dir, "concat.txt")
            with open(concat_list, 'w', encoding='utf-8') as f:
                for seg, _ in final_segments_sorted:
                    f.write(f"file '{seg}'\n")

            # get target_voice_path file name without path and extension
            target_voice_file_name = os.path.splitext(os.path.basename(target_voice_path))[0]
            final_output_path = os.path.join(output_dir, f"chatterbox-target-{target_voice_file_name}-_{base_name}.wav")

            all_segments = [seg for seg, _ in final_segments_sorted]

            batch_concat_python(all_segments, final_output=final_output_path)

    # remove temp_dir
    # shutil.rmtree(temp_dir)

def batch_concat_python(files, final_output):
    """
    Concatenate and mix segments precisely using pydub AudioSegment.
    Files should be named with start times encoded (e.g., _start_end.wav).
    """
    # Regex to extract start time (seconds)
    re_time = re.compile(r'_(\d+\.?\d*)_\d+\.?\d*\.wav$')
    segments = []  # (start_ms, AudioSegment)

    # Load segments and schedule overlays
    max_end = 0
    for f in files:
        m = re_time.search(os.path.basename(f))
        start_sec = float(m.group(1)) if m else 0.0
        audio = AudioSegment.from_file(f)
        start_ms = int(start_sec * 1000)
        end_ms = start_ms + len(audio)
        if end_ms > max_end:
            max_end = end_ms
        segments.append((start_ms, audio))

    # Create silent base track of required length
    mixed = AudioSegment.silent(duration=max_end)
    # Overlay each segment at correct position
    for start_ms, audio in segments:
        mixed = mixed.overlay(audio, position=start_ms)

    # Export mixed result
    mixed.export(final_output, format='wav')
    safe_print(f"[PYTHON CONCAT] Created {final_output} using pydub mix.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process audio files by splitting into silence and non-silence segments.')
    parser.add_argument('--input_dir', required=True, help="Input directory")
    parser.add_argument('--output_dir', required=True, help="Output directory")
    parser.add_argument('--target', help="Path to the target sound file")
    parser.add_argument('--log_file', default=LOG_FILE, help="Log")
    args = parser.parse_args()

    log_file = args.log_file

    # Record start time
    start_time = datetime.datetime.now()
    log_message(f"Processing started at {start_time}")
    safe_print(f"Processing started at {start_time}")

    process_audio_files(args.input_dir, args.output_dir, args.target, False)

    # Record end time
    end_time = datetime.datetime.now()
    log_message(f"Processing ended at {end_time}, duration: {end_time - start_time}")
    safe_print(f"Processing ended at {end_time}, duration: {end_time - start_time}")
