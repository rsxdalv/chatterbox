from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import librosa
import torch
import perth
import torch.nn.functional as F
from huggingface_hub import hf_hub_download
from safetensors.torch import load_file

from .models.t3 import T3
from .models.s3tokenizer import S3_SR, drop_invalid_tokens
from .models.s3gen import S3GEN_SR, S3Gen
from .models.tokenizers import EnTokenizer
from .models.voice_encoder import VoiceEncoder
from .models.t3.modules.cond_enc import T3Cond


REPO_ID = "ResembleAI/chatterbox"


def punc_norm(text: str) -> str:
    """
        Quick cleanup func for punctuation from LLMs or
        containing chars not seen often in the dataset
    """
    if len(text) == 0:
        return "You need to add some text for me to talk."

    # Capitalise first letter
    if text[0].islower():
        text = text[0].upper() + text[1:]

    # Remove multiple space chars
    text = " ".join(text.split())

    # Replace uncommon/llm punc
    punc_to_replace = [
        ("...", ", "),
        ("…", ", "),
        (":", ","),
        (" - ", ", "),
        (";", ", "),
        ("—", "-"),
        ("–", "-"),
        (" ,", ","),
        ("“", "\""),
        ("”", "\""),
        ("‘", "'"),
        ("’", "'"),
    ]
    for old_char_sequence, new_char in punc_to_replace:
        text = text.replace(old_char_sequence, new_char)

    # Add full stop if no ending punc
    text = text.rstrip(" ")
    sentence_enders = {".", "!", "?", "-", ","}
    if not any(text.endswith(p) for p in sentence_enders):
        text += "."

    return text


@dataclass
class Conditionals:
    """
    Conditionals for T3 and S3Gen
    - T3 conditionals:
        - speaker_emb
        - clap_emb
        - cond_prompt_speech_tokens
        - cond_prompt_speech_emb
        - emotion_adv
    - S3Gen conditionals:
        - prompt_token
        - prompt_token_len
        - prompt_feat
        - prompt_feat_len
        - embedding
    """
    t3: T3Cond
    gen: dict

    def to(self, device):
        self.t3 = self.t3.to(device=device)
        for k, v in self.gen.items():
            if torch.is_tensor(v):
                self.gen[k] = v.to(device=device)
        return self

    def save(self, fpath: Path):
        arg_dict = dict(
            t3=self.t3.__dict__,
            gen=self.gen
        )
        torch.save(arg_dict, fpath)

    @classmethod
    def load(cls, fpath, map_location="cpu"):
        if isinstance(map_location, str):
            map_location = torch.device(map_location)
        kwargs = torch.load(fpath, map_location=map_location, weights_only=True)
        return cls(T3Cond(**kwargs['t3']), kwargs['gen'])


class ChatterboxTTS:
    ENC_COND_LEN = 6 * S3_SR
    DEC_COND_LEN = 10 * S3GEN_SR

    def __init__(
        self,
        t3: T3,
        s3gen: S3Gen,
        ve: VoiceEncoder,
        tokenizer: EnTokenizer,
        device: str,
        conds: Conditionals = None,
    ):
        self.sr = S3GEN_SR  # sample rate of synthesized audio
        self.t3 = t3
        self.s3gen = s3gen
        self.ve = ve
        self.tokenizer = tokenizer
        self.device = device
        self.conds = conds
        self.watermarker = perth.PerthImplicitWatermarker()

    @classmethod
    def from_local(cls, ckpt_dir, device) -> 'ChatterboxTTS':
        ckpt_dir = Path(ckpt_dir)

        # Always load to CPU first for non-CUDA devices to handle CUDA-saved models
        if device in ["cpu", "mps"]:
            map_location = torch.device('cpu')
        else:
            map_location = None

        ve = VoiceEncoder()
        ve.load_state_dict(
            load_file(ckpt_dir / "ve.safetensors")
        )
        ve.to(device).eval()

        t3 = T3()
        t3_state = load_file(ckpt_dir / "t3_cfg.safetensors")
        if "model" in t3_state.keys():
            t3_state = t3_state["model"][0]
        t3.load_state_dict(t3_state)
        t3.to(device).eval()

        s3gen = S3Gen()
        s3gen.load_state_dict(
            load_file(ckpt_dir / "s3gen.safetensors"), strict=False
        )
        s3gen.to(device).eval()

        tokenizer = EnTokenizer(
            str(ckpt_dir / "tokenizer.json")
        )

        conds = None
        if (builtin_voice := ckpt_dir / "conds.pt").exists():
            conds = Conditionals.load(builtin_voice, map_location=map_location).to(device)

        return cls(t3, s3gen, ve, tokenizer, device, conds=conds)

    @classmethod
    def from_pretrained(cls, device) -> 'ChatterboxTTS':
        # Check if MPS is available on macOS
        if device == "mps" and not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("MPS not available because the current PyTorch install was not built with MPS enabled.")
            else:
                print("MPS not available because the current MacOS version is not 12.3+ and/or you do not have an MPS-enabled device on this machine.")
            device = "cpu"

        for fpath in ["ve.safetensors", "t3_cfg.safetensors", "s3gen.safetensors", "tokenizer.json", "conds.pt"]:
            local_path = hf_hub_download(repo_id=REPO_ID, filename=fpath)

        return cls.from_local(Path(local_path).parent, device)

    def prepare_conditionals(self, wav_fpath, exaggeration=0.5):
        ## Load reference wav
        s3gen_ref_wav, _sr = librosa.load(wav_fpath, sr=S3GEN_SR)

        ref_16k_wav = librosa.resample(s3gen_ref_wav, orig_sr=S3GEN_SR, target_sr=S3_SR)

        s3gen_ref_wav = s3gen_ref_wav[:self.DEC_COND_LEN]
        s3gen_ref_dict = self.s3gen.embed_ref(s3gen_ref_wav, S3GEN_SR, device=self.device)

        # Speech cond prompt tokens
        if plen := self.t3.hp.speech_cond_prompt_len:
            s3_tokzr = self.s3gen.tokenizer
            t3_cond_prompt_tokens, _ = s3_tokzr.forward([ref_16k_wav[:self.ENC_COND_LEN]], max_len=plen)
            t3_cond_prompt_tokens = torch.atleast_2d(t3_cond_prompt_tokens).to(self.device)

        # Voice-encoder speaker embedding
        ve_embed = torch.from_numpy(self.ve.embeds_from_wavs([ref_16k_wav], sample_rate=S3_SR))
        ve_embed = ve_embed.mean(axis=0, keepdim=True).to(self.device)

        t3_cond = T3Cond(
            speaker_emb=ve_embed,
            cond_prompt_speech_tokens=t3_cond_prompt_tokens,
            emotion_adv=exaggeration * torch.ones(1, 1, 1),
        ).to(device=self.device)
        self.conds = Conditionals(t3_cond, s3gen_ref_dict)

    def generate(
        self,
        text,
        audio_prompt_path=None,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
        # stream
        tokens_per_slice=1000,
        remove_milliseconds=45,
        remove_milliseconds_start=25,
        chunk_overlap_method: Literal["zero", "full"] = "zero",
    ):
        # Monkey patch the apply_rotary_pos_emb function to add cudagraph_mark_step_begin
        # from transformers.models.llama.modeling_llama import apply_rotary_pos_emb as original_apply_rotary_pos_emb
        
        # def patched_apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
        #     torch.compiler.cudagraph_mark_step_begin()
        #     return original_apply_rotary_pos_emb(q, k, cos, sin, position_ids, unsqueeze_dim)
        
        # import transformers.models.llama.modeling_llama
        # transformers.models.llama.modeling_llama.apply_rotary_pos_emb = patched_apply_rotary_pos_emb
        
        if audio_prompt_path:
            self.prepare_conditionals(audio_prompt_path, exaggeration=exaggeration)
        else:
            assert self.conds is not None, "Please `prepare_conditionals` first or specify `audio_prompt_path`"

        # Update exaggeration if needed
        if exaggeration != self.conds.t3.emotion_adv[0, 0, 0]:
            _cond: T3Cond = self.conds.t3
            self.conds.t3 = T3Cond(
                speaker_emb=_cond.speaker_emb,
                cond_prompt_speech_tokens=_cond.cond_prompt_speech_tokens,
                emotion_adv=exaggeration * torch.ones(1, 1, 1),
            ).to(device=self.device)

        # Norm and tokenize text
        text = punc_norm(text)
        text_tokens = self.tokenizer.text_to_tokens(text).to(self.device)

        if cfg_weight > 0.0:
            text_tokens = torch.cat([text_tokens, text_tokens], dim=0)  # Need two seqs for CFG

        sot = self.t3.hp.start_text_token
        eot = self.t3.hp.stop_text_token
        text_tokens = F.pad(text_tokens, (1, 0), value=sot)
        text_tokens = F.pad(text_tokens, (0, 1), value=eot)

        with torch.inference_mode():

            @torch.profiler.record_function("t3_infer")
            def _t3_infer():
                # with torch.profiler.record_function("t3_infer"):
                for token in self.t3.inference(
                    t3_cond=self.conds.t3,
                    text_tokens=text_tokens,
                    max_new_tokens=1000,  # TODO: use the value in config
                    temperature=temperature,
                    cfg_weight=cfg_weight,
                ):
                    torch.compiler.cudagraph_mark_step_begin()
                    yield token
            
            @torch.profiler.record_function("s3gen_infer")
            def speech_to_wav(speech_tokens, previous_length=0):
                # Extract only the conditional batch.
                speech_tokens = speech_tokens[0]

                # speech_tokens = speech_tokens[-tokens_per_slice * 2:]

                # TODO: output becomes 1D
                speech_tokens = drop_invalid_tokens(speech_tokens)
                
                speech_tokens = speech_tokens[speech_tokens < 6561]

                speech_tokens = speech_tokens.to(self.device)

                wav, _ = self.s3gen.inference(
                    speech_tokens=speech_tokens,
                    ref_dict=self.conds.gen,
                    no_trim=tokens_per_slice < 1000,
                )
                wav = wav.squeeze(0).detach().cpu().numpy()

                if chunk_overlap_method == "full":
                    wav = wav[previous_length:]
                
                watermarked_wav = self.watermarker.apply_watermark(wav, sample_rate=self.sr)
                if remove_milliseconds > 0:
                    watermarked_wav = watermarked_wav[:-int(self.sr * remove_milliseconds / 1000)]
                if remove_milliseconds_start > 0:
                    watermarked_wav = watermarked_wav[int(self.sr * remove_milliseconds_start / 1000):]

                return torch.from_numpy(watermarked_wav).unsqueeze(0), len(wav) + previous_length

                tokens_to_seconds = int(288960 * stride / 300)
                new_wav_stride = wav[-tokens_to_seconds:]
                if previous_wav_stride is not None:
                # if False:
                    import numpy as np
                    blended_tokens = min(tokens_to_seconds // 2, len(watermarked_wav), len(previous_wav_stride))

                    fade_in = np.linspace(0, 1, blended_tokens, endpoint=False)
                    fade_out = 1 - fade_in  # goes from 1 → 0

                    # Get segments to blend
                    prev_tail = previous_wav_stride[-blended_tokens:]
                    curr_head = watermarked_wav[:blended_tokens]

                    # Apply crossfade
                    crossfaded = prev_tail * fade_out + curr_head * fade_in

                    # Replace the start of current audio with the crossfade
                    watermarked_wav = np.concatenate([
                        crossfaded,
                        watermarked_wav[blended_tokens:]
                    ])
                watermarked_wav = watermarked_wav[:-tokens_to_seconds]
                print("watermarked_wav", watermarked_wav.shape)
                return torch.from_numpy(watermarked_wav).unsqueeze(0), new_wav_stride
            
            eos_token = torch.tensor([self.t3.hp.stop_text_token]).unsqueeze(0).to(self.device)

            def chunked():
                """Yield successive chunks of tokens from the inference generator."""
                token_stream = []
                for batch in _t3_infer():
                    token_stream.extend(batch.squeeze(0))
                    
                    while len(token_stream) >= tokens_per_slice:
                        yield token_stream[:tokens_per_slice]
                        token_stream = token_stream[tokens_per_slice:]
                
                # Yield any remaining tokens
                if token_stream:
                    yield token_stream

            def accumulating_chunks():
                """Yield accumulating slices of tokens, emitting every tokens_per_slice tokens."""
                accumulated = []
                
                for batch in _t3_infer():
                    accumulated.extend(batch.squeeze(0))
                    
                    # Yield whenever we have a multiple of tokens_per_slice tokens
                    if len(accumulated) % tokens_per_slice == 0 and len(accumulated) > 0:
                        yield accumulated.copy()
                
                # Yield any remaining tokens if not already yielded
                if accumulated and len(accumulated) % tokens_per_slice != 0:
                    yield accumulated.copy()


            previous_length = 0
            iterator = chunked() if chunk_overlap_method == "zero" else accumulating_chunks()
            for slice_tokens in iterator:
                tokens = torch.stack(slice_tokens).unsqueeze(0)
                tokens_with_eos = torch.cat([tokens, eos_token], dim=1)
                wav, previous_length = speech_to_wav(tokens_with_eos, previous_length)
                yield wav

