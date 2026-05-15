#!/usr/bin/env python3
from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import subprocess
import threading
import time
import wave
from collections import Counter
from pathlib import Path
from typing import Any, Literal, Optional

import numpy as np
import torch
from celery import Celery
from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel, Field

from chatterbox.mtl_tts import ChatterboxMultilingualTTS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("chatterbox_multilingual_api")


DEVICE = os.getenv("DEVICE", "cuda")
REQUIRE_CUDA = os.getenv("REQUIRE_CUDA", "1") == "1"
EXPECTED_GPU_NAME = os.getenv("EXPECTED_GPU_NAME", "").strip()

MAX_INPUT_CHARS = int(os.getenv("MAX_INPUT_CHARS", "12000"))
AUTO_CHUNK_ENABLED = os.getenv("AUTO_CHUNK_ENABLED", "1") == "1"
AUTO_CHUNK_HARD_LIMIT = int(os.getenv("AUTO_CHUNK_HARD_LIMIT", "580"))
MIN_CHUNK_CHARS = int(os.getenv("MIN_CHUNK_CHARS", "60"))
CHUNK_PAUSE_MS = int(os.getenv("CHUNK_PAUSE_MS", "140"))
DEFAULT_RESPONSE_FORMAT = os.getenv("DEFAULT_RESPONSE_FORMAT", "wav").strip().lower()

STARTUP_WARMUP = os.getenv("STARTUP_WARMUP", "0") == "1"
LAZY_LOAD_MODEL = os.getenv("LAZY_LOAD_MODEL", "1") == "1"
MODEL_IDLE_UNLOAD_SECONDS = int(os.getenv("MODEL_IDLE_UNLOAD_SECONDS", "300"))
MODEL_IDLE_CHECK_INTERVAL_SECONDS = int(os.getenv("MODEL_IDLE_CHECK_INTERVAL_SECONDS", "30"))
MIN_FREE_VRAM_MB = int(os.getenv("MIN_FREE_VRAM_MB", "4500"))
MODEL_LOAD_WAIT_TIMEOUT_SECONDS = int(os.getenv("MODEL_LOAD_WAIT_TIMEOUT_SECONDS", "60"))

ENABLE_CELERY = os.getenv("ENABLE_CELERY", "0") == "1"
CELERY_TASK_TIMEOUT_SECONDS = int(os.getenv("CELERY_TASK_TIMEOUT_SECONDS", "1800"))
CELERY_BROKER_URL = os.getenv("CELERY_BROKER_URL", "redis://127.0.0.1:6379/14")
CELERY_RESULT_BACKEND = os.getenv("CELERY_RESULT_BACKEND", CELERY_BROKER_URL)
CELERY_QUEUE = os.getenv("CELERY_QUEUE", "chatterbox_mtl_3060x2")
WORKER_GPU_INDICES = os.getenv("WORKER_GPU_INDICES", "2,3").strip()
WORKER_GPU_INDEX_LIST = [item.strip() for item in WORKER_GPU_INDICES.split(",") if item.strip()]
MAX_CELERY_PARALLEL_TASKS = int(
    os.getenv("MAX_CELERY_PARALLEL_TASKS", str(max(1, len(WORKER_GPU_INDEX_LIST))))
)
CELERY_CHUNK_DISPATCH_MODE = os.getenv("CELERY_CHUNK_DISPATCH_MODE", "dynamic_chunks").strip().lower()

SUPPORTED_LANGUAGES = ChatterboxMultilingualTTS.get_supported_languages()

DEFAULT_EXAGGERATION = float(os.getenv("DEFAULT_EXAGGERATION", "0.5"))
DEFAULT_CFG_WEIGHT = float(os.getenv("DEFAULT_CFG_WEIGHT", "0.3"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.8"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("DEFAULT_REPETITION_PENALTY", "2.0"))
DEFAULT_MIN_P = float(os.getenv("DEFAULT_MIN_P", "0.05"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "1.0"))


app = FastAPI(title="Real Chatterbox Multilingual", version="2.0.0")
celery_app = Celery("chatterbox_multilingual", broker=CELERY_BROKER_URL, backend=CELERY_RESULT_BACKEND)

model: Optional[ChatterboxMultilingualTTS] = None
model_loaded_at_seconds: Optional[float] = None
model_lock = threading.Lock()
device_configured = False
model_last_used_monotonic = 0.0
idle_monitor_started = False


class SpeechRequest(BaseModel):
    model: str = "chatterbox_multilingual"
    input: str = Field(min_length=1, max_length=MAX_INPUT_CHARS)
    voice: str = "default"
    language: str = "es"
    response_format: Literal["wav", "mp3"] = DEFAULT_RESPONSE_FORMAT
    exaggeration: float = DEFAULT_EXAGGERATION
    cfg_weight: float = DEFAULT_CFG_WEIGHT
    temperature: float = DEFAULT_TEMPERATURE
    repetition_penalty: float = DEFAULT_REPETITION_PENALTY
    min_p: float = DEFAULT_MIN_P
    top_p: float = DEFAULT_TOP_P


def configure_torch() -> str:
    global DEVICE
    global device_configured

    if device_configured:
        return DEVICE

    torch.set_grad_enabled(False)

    if DEVICE.startswith("cuda"):
        if not torch.cuda.is_available():
            if REQUIRE_CUDA:
                raise RuntimeError("DEVICE=cuda requested, but torch.cuda.is_available() is false.")
            DEVICE = "cpu"
        else:
            gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
            if EXPECTED_GPU_NAME and EXPECTED_GPU_NAME.lower() not in gpu_name.lower():
                raise RuntimeError(
                    f"Selected GPU '{gpu_name}' does not match EXPECTED_GPU_NAME='{EXPECTED_GPU_NAME}'."
                )
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True

    device_configured = True
    return DEVICE


def touch_model_usage() -> None:
    global model_last_used_monotonic
    model_last_used_monotonic = time.monotonic()


def unload_model_locked(reason: str) -> None:
    global model

    if model is None:
        return

    logger.info("Unloading multilingual model (%s).", reason)
    model = None
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    touch_model_usage()


def _idle_monitor_loop() -> None:
    while True:
        time.sleep(max(1, MODEL_IDLE_CHECK_INTERVAL_SECONDS))
        if MODEL_IDLE_UNLOAD_SECONDS <= 0:
            continue
        if model is None:
            continue
        idle_seconds = time.monotonic() - model_last_used_monotonic
        if idle_seconds < MODEL_IDLE_UNLOAD_SECONDS:
            continue
        with model_lock:
            if model is None:
                continue
            idle_seconds = time.monotonic() - model_last_used_monotonic
            if idle_seconds >= MODEL_IDLE_UNLOAD_SECONDS:
                unload_model_locked(f"idle for {idle_seconds:.1f}s")


def ensure_idle_monitor_started() -> None:
    global idle_monitor_started
    if idle_monitor_started:
        return
    idle_monitor_started = True
    thread = threading.Thread(target=_idle_monitor_loop, name="multilingual-model-idle-monitor", daemon=True)
    thread.start()


def wait_for_free_vram_if_needed() -> None:
    if not DEVICE.startswith("cuda") or MIN_FREE_VRAM_MB <= 0 or not torch.cuda.is_available():
        return
    deadline = time.monotonic() + max(1, MODEL_LOAD_WAIT_TIMEOUT_SECONDS)
    while True:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_mb = free_bytes / 1024**2
        total_mb = total_bytes / 1024**2
        if free_mb >= MIN_FREE_VRAM_MB:
            logger.info("VRAM gate passed: %.0f MiB free / %.0f MiB total.", free_mb, total_mb)
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Not enough free VRAM to load Chatterbox Multilingual: {free_mb:.0f} MiB free, "
                f"need at least {MIN_FREE_VRAM_MB} MiB."
            )
        logger.info("Waiting for free VRAM: %.0f MiB free, need %d MiB.", free_mb, MIN_FREE_VRAM_MB)
        time.sleep(5)


def get_model() -> ChatterboxMultilingualTTS:
    global model
    global model_loaded_at_seconds

    ensure_idle_monitor_started()
    configure_torch()
    touch_model_usage()

    with model_lock:
        if model is None:
            wait_for_free_vram_if_needed()
            logger.info("Loading Chatterbox Multilingual on %s...", DEVICE)
            started = time.perf_counter()
            model = ChatterboxMultilingualTTS.from_pretrained(device=DEVICE)
            model_loaded_at_seconds = time.perf_counter() - started
            logger.info("Multilingual model loaded in %.3fs", model_loaded_at_seconds)
        touch_model_usage()
        return model


def float_array_from_tensor(wav: torch.Tensor | np.ndarray) -> np.ndarray:
    if isinstance(wav, torch.Tensor):
        arr = wav.detach().cpu().float().numpy()
    else:
        arr = np.asarray(wav, dtype=np.float32)
    if arr.ndim > 1:
        arr = arr[0]
    return np.clip(arr.astype(np.float32, copy=False), -1.0, 1.0)


def pcm16_bytes_from_array(arr: np.ndarray) -> bytes:
    pcm = (np.clip(arr, -1.0, 1.0) * 32767.0).astype(np.int16)
    return pcm.tobytes()


def wav_bytes_from_pcm16_bytes(raw: bytes, sample_rate: int) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(raw)
    return buf.getvalue()


def mp3_bytes_from_pcm16_bytes(raw: bytes, sample_rate: int) -> bytes:
    cmd = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        "-f",
        "mp3",
        "-codec:a",
        "libmp3lame",
        "-b:a",
        "160k",
        "pipe:1",
    ]
    result = subprocess.run(cmd, input=raw, capture_output=True, check=True)
    return result.stdout


def encode_pcm16_bytes(raw: bytes, sample_rate: int, output_format: str) -> tuple[bytes, str, str]:
    if output_format == "wav":
        return wav_bytes_from_pcm16_bytes(raw, sample_rate), "audio/wav", "wav"
    if output_format == "mp3":
        return mp3_bytes_from_pcm16_bytes(raw, sample_rate), "audio/mpeg", "mp3"
    raise RuntimeError(f"Unsupported output format: {output_format}")


def split_oversized_unit(text: str, hard_limit: int) -> list[str]:
    text = text.strip()
    if len(text) <= hard_limit:
        return [text]

    clauses = [part.strip() for part in re.split(r"(?<=[,;:])\s+", text) if part.strip()]
    pieces: list[str] = []
    current = ""

    for clause in clauses:
        candidate = clause if not current else f"{current} {clause}"
        if len(candidate) <= hard_limit:
            current = candidate
            continue
        if current:
            pieces.append(current)
            current = clause
            continue

        words = clause.split()
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            if len(candidate) <= hard_limit:
                current = candidate
            else:
                if current:
                    pieces.append(current)
                current = word
        if current:
            pieces.append(current)
            current = ""

    if current:
        pieces.append(current)

    return pieces


def chunk_text(text: str) -> list[str]:
    cleaned = re.sub(r"[ \t]+", " ", text.strip())
    if not cleaned:
        return []
    if not AUTO_CHUNK_ENABLED:
        return [cleaned]

    raw_sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", cleaned) if s.strip()]
    if not raw_sentences:
        return [cleaned]

    units: list[str] = []
    for sentence in raw_sentences:
        if len(sentence) > AUTO_CHUNK_HARD_LIMIT:
            units.extend(split_oversized_unit(sentence, AUTO_CHUNK_HARD_LIMIT))
        else:
            units.append(sentence)

    chunks: list[str] = []
    current = ""
    for unit in units:
        if not current:
            current = unit
            continue
        merged = f"{current} {unit}"
        if len(merged) > AUTO_CHUNK_HARD_LIMIT:
            chunks.append(current)
            current = unit
        elif len(current) < MIN_CHUNK_CHARS:
            current = merged
        else:
            chunks.append(current)
            current = unit
    if current:
        chunks.append(current)

    if len(chunks) >= 2 and len(chunks[-1]) < MIN_CHUNK_CHARS:
        merged = f"{chunks[-2]} {chunks[-1]}"
        if len(merged) <= AUTO_CHUNK_HARD_LIMIT:
            chunks[-2] = merged
            chunks.pop()

    return chunks


def build_chunk_task_specs(chunks: list[str], base_payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    if not chunks:
        return []

    if CELERY_CHUNK_DISPATCH_MODE in {"dynamic", "dynamic_chunks", "individual", "per_chunk"}:
        return [
            ("chatterbox_multilingual.synthesize_chunk", {**base_payload, "text": chunk, "index": idx})
            for idx, chunk in enumerate(chunks)
        ]

    max_parallel = max(1, min(len(chunks), MAX_CELERY_PARALLEL_TASKS))
    batches: list[list[dict[str, Any]]] = [[] for _ in range(max_parallel)]
    batch_loads = [0] * max_parallel
    indexed_chunks = sorted(enumerate(chunks), key=lambda item: len(item[1]), reverse=True)

    for idx, chunk in indexed_chunks:
        batch_idx = min(range(max_parallel), key=lambda item: batch_loads[item])
        batches[batch_idx].append({"index": idx, "text": chunk})
        batch_loads[batch_idx] += len(chunk)

    specs: list[tuple[str, dict[str, Any]]] = []
    for batch in batches:
        if not batch:
            continue
        ordered = sorted(batch, key=lambda item: item["index"])
        if len(ordered) == 1:
            item = ordered[0]
            specs.append(("chatterbox_multilingual.synthesize_chunk", {**base_payload, "text": item["text"], "index": item["index"]}))
        else:
            specs.append(("chatterbox_multilingual.synthesize_chunk_batch", {**base_payload, "items": ordered}))
    return specs


def concatenate_pcm16_chunks(chunks: list[bytes], sample_rate: int) -> tuple[bytes, int]:
    if not chunks:
        return b"", 0
    pause_frames = max(0, int(sample_rate * (CHUNK_PAUSE_MS / 1000.0)))
    silence = b"\x00\x00" * pause_frames
    merged: list[bytes] = []
    total_frames = 0
    for idx, chunk in enumerate(chunks):
        if idx > 0 and pause_frames > 0:
            merged.append(silence)
            total_frames += pause_frames
        merged.append(chunk)
        total_frames += len(chunk) // 2
    return b"".join(merged), total_frames


def synthesize_chunk_array(
    *,
    text: str,
    language: str,
    exaggeration: float,
    cfg_weight: float,
    temperature: float,
    repetition_penalty: float,
    min_p: float,
    top_p: float,
) -> tuple[np.ndarray, int]:
    m = get_model()
    touch_model_usage()
    wav = m.generate(
        text,
        language_id=language,
        exaggeration=exaggeration,
        cfg_weight=cfg_weight,
        temperature=temperature,
        repetition_penalty=repetition_penalty,
        min_p=min_p,
        top_p=top_p,
    )
    return float_array_from_tensor(wav), int(m.sr)


def synthesize_single_chunk_payload(payload: dict[str, Any]) -> dict[str, Any]:
    arr, sample_rate = synthesize_chunk_array(
        text=payload["text"],
        language=payload["language"],
        exaggeration=float(payload["exaggeration"]),
        cfg_weight=float(payload["cfg_weight"]),
        temperature=float(payload["temperature"]),
        repetition_penalty=float(payload["repetition_penalty"]),
        min_p=float(payload["min_p"]),
        top_p=float(payload["top_p"]),
    )
    raw = pcm16_bytes_from_array(arr)
    return {
        "items": [
            {
                "index": int(payload.get("index", 0)),
                "sample_rate": sample_rate,
                "pcm16_base64": base64.b64encode(raw).decode("ascii"),
                "worker_id": f"pid:{os.getpid()}",
            }
        ]
    }


def synthesize_chunk_batch_payload(payload: dict[str, Any]) -> dict[str, Any]:
    items_out: list[dict[str, Any]] = []
    worker_id = f"pid:{os.getpid()}"
    for item in payload.get("items", []):
        arr, sample_rate = synthesize_chunk_array(
            text=item["text"],
            language=payload["language"],
            exaggeration=float(payload["exaggeration"]),
            cfg_weight=float(payload["cfg_weight"]),
            temperature=float(payload["temperature"]),
            repetition_penalty=float(payload["repetition_penalty"]),
            min_p=float(payload["min_p"]),
            top_p=float(payload["top_p"]),
        )
        raw = pcm16_bytes_from_array(arr)
        items_out.append(
            {
                "index": int(item["index"]),
                "sample_rate": sample_rate,
                "pcm16_base64": base64.b64encode(raw).decode("ascii"),
                "worker_id": worker_id,
            }
        )
    return {"items": items_out}


@celery_app.task(name="chatterbox_multilingual.synthesize_chunk")
def celery_synthesize_chunk(payload: dict[str, Any]) -> dict[str, Any]:
    return synthesize_single_chunk_payload(payload)


@celery_app.task(name="chatterbox_multilingual.synthesize_chunk_batch")
def celery_synthesize_chunk_batch(payload: dict[str, Any]) -> dict[str, Any]:
    return synthesize_chunk_batch_payload(payload)


def wait_for_celery_chunk_payloads(tasks: list[Any]) -> list[dict[str, Any]]:
    deadline = time.monotonic() + CELERY_TASK_TIMEOUT_SECONDS
    payloads_by_index: list[dict[str, Any] | None] = [None] * len(tasks)
    pending = list(enumerate(tasks))
    while pending:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise HTTPException(status_code=504, detail=f"Timed out waiting for {len(pending)}/{len(tasks)} chunk tasks.")

        still_pending: list[tuple[int, Any]] = []
        for idx, task in pending:
            if not task.ready():
                still_pending.append((idx, task))
                continue
            if task.failed():
                raise HTTPException(status_code=500, detail=f"Chunk {idx} synthesis failed: {task.result}")
            payload = task.result
            if not isinstance(payload, dict):
                raise HTTPException(status_code=500, detail=f"Chunk {idx} returned an unexpected payload type.")
            payloads_by_index[idx] = payload
            try:
                task.forget()
            except Exception:
                pass

        pending = still_pending
        if pending:
            time.sleep(0.05)
    return [payload for payload in payloads_by_index if payload is not None]


def decode_chunk_payload_items(payloads: list[dict[str, Any]]) -> list[tuple[int, bytes, int, str]]:
    items_out: list[tuple[int, bytes, int, str]] = []
    for payload in payloads:
        items = payload.get("items", [])
        if not isinstance(items, list):
            raise HTTPException(status_code=500, detail="Chunk task returned no items.")
        for item in items:
            items_out.append(
                (
                    int(item["index"]),
                    base64.b64decode(item["pcm16_base64"]),
                    int(item["sample_rate"]),
                    str(item.get("worker_id") or "unknown"),
                )
            )
    items_out.sort(key=lambda item: item[0])
    return items_out


def finalize_chunk_payloads(payloads: list[dict[str, Any]], *, chunk_count: int, output_format: str, started: float, language: str) -> tuple[bytes, dict[str, Any]]:
    chunk_items = decode_chunk_payload_items(payloads)
    if len(chunk_items) != chunk_count:
        raise HTTPException(status_code=500, detail=f"Expected {chunk_count} chunk results but received {len(chunk_items)}.")

    pcm_chunks: list[bytes] = []
    worker_ids: list[str] = []
    sample_rate: Optional[int] = None
    indices_seen: list[int] = []
    for index, raw, sr, worker_id in chunk_items:
        indices_seen.append(index)
        pcm_chunks.append(raw)
        worker_ids.append(worker_id)
        if sample_rate is None:
            sample_rate = sr
        elif sr != sample_rate:
            raise HTTPException(status_code=500, detail="Chunk sample rate mismatch.")
    if sample_rate is None:
        raise HTTPException(status_code=500, detail="No sample rate returned from chunk synthesis.")
    if indices_seen != list(range(chunk_count)):
        raise HTTPException(status_code=500, detail="Chunk results were incomplete or out of order.")

    stitched_pcm16, total_frames = concatenate_pcm16_chunks(pcm_chunks, sample_rate)
    audio_seconds = round(total_frames / sample_rate, 4)
    wall_seconds = round(time.monotonic() - started, 4)
    final_bytes, content_type, extension = encode_pcm16_bytes(stitched_pcm16, sample_rate, output_format)
    meta = {
        "sample_rate": sample_rate,
        "audio_seconds": audio_seconds,
        "wall_seconds": wall_seconds,
        "rtf": round(wall_seconds / audio_seconds, 4) if audio_seconds > 0 else None,
        "x_realtime": round(audio_seconds / wall_seconds, 4) if wall_seconds > 0 else None,
        "chunk_count": chunk_count,
        "chunk_task_count": len(payloads),
        "chunk_worker_counts": dict(sorted(Counter(worker_ids).items())),
        "content_type": content_type,
        "filename": f"speech.{extension}",
        "output_format": output_format,
        "language": language,
    }
    return final_bytes, meta


def response_with_audio(audio_bytes: bytes, meta: dict[str, Any]) -> Response:
    headers: dict[str, str] = {
        "Content-Disposition": f'attachment; filename="{meta["filename"]}"',
        "X-Sample-Rate": str(meta["sample_rate"]),
        "X-Wall-Seconds": str(meta["wall_seconds"]),
        "X-Audio-Seconds": str(meta["audio_seconds"]),
        "X-Chunk-Count": str(meta["chunk_count"]),
        "X-Chunk-Task-Count": str(meta["chunk_task_count"]),
        "X-Output-Format": str(meta["output_format"]),
        "X-Language": str(meta["language"]),
    }
    if meta.get("chunk_worker_counts"):
        headers["X-Chunk-Worker-Counts"] = json.dumps(meta["chunk_worker_counts"], sort_keys=True)
    if meta.get("rtf") is not None:
        headers["X-RTF"] = str(meta["rtf"])
    if meta.get("x_realtime") is not None:
        headers["X-Speed-X-Realtime"] = str(meta["x_realtime"])
    return Response(content=audio_bytes, media_type=meta["content_type"], headers=headers)


def runtime_status() -> dict[str, Any]:
    gpu_name = None
    if not ENABLE_CELERY and torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(torch.cuda.current_device())
    return {
        "ok": True,
        "backend_mode": "celery" if ENABLE_CELERY else "local",
        "engine": "ChatterboxMultilingualTTS",
        "model_loaded": model is not None,
        "model_load_seconds": model_loaded_at_seconds,
        "cuda_available": torch.cuda.is_available(),
        "gpu": gpu_name,
        "expected_gpu_name": EXPECTED_GPU_NAME or None,
        "supported_languages": sorted(SUPPORTED_LANGUAGES.keys()),
        "lazy_load_model": LAZY_LOAD_MODEL,
        "model_idle_unload_seconds": MODEL_IDLE_UNLOAD_SECONDS,
        "max_input_chars": MAX_INPUT_CHARS,
        "worker_gpu_indices": WORKER_GPU_INDICES or None,
        "worker_count": max(1, len(WORKER_GPU_INDEX_LIST)) if ENABLE_CELERY else None,
        "max_parallel_chunk_tasks": MAX_CELERY_PARALLEL_TASKS if ENABLE_CELERY else None,
        "celery_chunk_dispatch_mode": CELERY_CHUNK_DISPATCH_MODE if ENABLE_CELERY else None,
        "celery_queue": CELERY_QUEUE if ENABLE_CELERY else None,
    }


@app.on_event("startup")
def on_startup() -> None:
    ensure_idle_monitor_started()
    if STARTUP_WARMUP and not LAZY_LOAD_MODEL:
        get_model()


@app.get("/health")
def health() -> dict[str, Any]:
    return runtime_status()


@app.get("/status")
def status() -> dict[str, Any]:
    return runtime_status()


@app.get("/v1/tts/engines")
def engines() -> dict[str, list[dict[str, Any]]]:
    return {"engines": [{"name": "chatterbox_multilingual", "loaded": model is not None, "backend": "real-pytorch-cuda", "mock": False}]}


@app.post("/v1/audio/speech")
def speech(req: SpeechRequest) -> Response:
    language = req.language.lower().strip()
    if language not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"unsupported language: {req.language}")
    if len(req.input) > MAX_INPUT_CHARS:
        raise HTTPException(status_code=400, detail=f"input exceeds max_input_chars={MAX_INPUT_CHARS}")

    text = req.input.strip()
    output_format = req.response_format.strip().lower()
    if output_format not in {"wav", "mp3"}:
        raise HTTPException(status_code=400, detail=f"unsupported response_format: {req.response_format}")

    chunks = chunk_text(text)
    if not chunks:
        raise HTTPException(status_code=400, detail="No text to synthesize after cleaning.")

    if ENABLE_CELERY:
        from celery.result import AsyncResult

        base_payload = {
            "language": language,
            "exaggeration": req.exaggeration,
            "cfg_weight": req.cfg_weight,
            "temperature": req.temperature,
            "repetition_penalty": req.repetition_penalty,
            "min_p": req.min_p,
            "top_p": req.top_p,
        }
        task_specs = build_chunk_task_specs(chunks, base_payload)
        logger.info("Parallel dispatch: %d chunks across %d task(s) for %d chars (%s)", len(chunks), len(task_specs), len(text), language)
        started = time.monotonic()
        tasks: list[AsyncResult] = [
            celery_app.send_task(task_name, kwargs={"payload": task_payload}, queue=CELERY_QUEUE)
            for task_name, task_payload in task_specs
        ]
        payloads = wait_for_celery_chunk_payloads(tasks)
        audio_bytes, meta = finalize_chunk_payloads(payloads, chunk_count=len(chunks), output_format=output_format, started=started, language=language)
        return response_with_audio(audio_bytes, meta)

    started = time.monotonic()
    pcm_chunks: list[bytes] = []
    sample_rate: Optional[int] = None
    for chunk in chunks:
        arr, sr = synthesize_chunk_array(
            text=chunk,
            language=language,
            exaggeration=req.exaggeration,
            cfg_weight=req.cfg_weight,
            temperature=req.temperature,
            repetition_penalty=req.repetition_penalty,
            min_p=req.min_p,
            top_p=req.top_p,
        )
        pcm_chunks.append(pcm16_bytes_from_array(arr))
        if sample_rate is None:
            sample_rate = sr
    if sample_rate is None:
        raise HTTPException(status_code=500, detail="No sample rate returned from synthesis.")
    stitched_pcm16, total_frames = concatenate_pcm16_chunks(pcm_chunks, sample_rate)
    audio_seconds = round(total_frames / sample_rate, 4)
    wall_seconds = round(time.monotonic() - started, 4)
    final_bytes, content_type, extension = encode_pcm16_bytes(stitched_pcm16, sample_rate, output_format)
    meta = {
        "sample_rate": sample_rate,
        "audio_seconds": audio_seconds,
        "wall_seconds": wall_seconds,
        "rtf": round(wall_seconds / audio_seconds, 4) if audio_seconds > 0 else None,
        "x_realtime": round(audio_seconds / wall_seconds, 4) if wall_seconds > 0 else None,
        "chunk_count": len(chunks),
        "chunk_task_count": len(chunks),
        "chunk_worker_counts": {"local": len(chunks)},
        "content_type": content_type,
        "filename": f"speech.{extension}",
        "output_format": output_format,
        "language": language,
    }
    return response_with_audio(final_bytes, meta)
