#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
InfiniteVL Streaming Inference Demo
===================================

This script demonstrates how to perform real-time streaming video understanding 
using InfiniteVL. It showcases the model's ability to handle unlimited context 
via its linear attention mechanism and CUDA Graph acceleration.

Key Features:
1. Continuous State Update: Updates a fixed-size memory cache frame-by-frame.
2. CUDA Graph Acceleration: Captures the static computation graph for video 
   updates to achieve high FPS (e.g., ~24+ FPS on RTX 4090).
3. Branching QA: Allows asking questions at specific timestamps without 
   corrupting the main streaming state (by cloning the cache).

Usage:
    python demo_streaming_inference.py \
        --model_path /path/to/InfiniteVL \
        --video_path /path/to/video.mp4
"""

import argparse
import os
import time
import sys
import cv2
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

# ================= Configuration & Constants =================

# Default queries for demonstration: (Frame Index, Question)
DEFAULT_QUERIES = [
    (150, "Describe what is happening in the current scene."),
    (300, "Describe what is happening in the current scene."),
    (450, "Describe what is happening in the current scene."),
]

IMG_TOKENS_PER_FRAME = 256  # InfiniteVL specific: tokens per visual frame
FRAME_RESIZE = (448, 448)   # Standard resolution for efficient streaming
DTYPE = torch.bfloat16
DEVICE = "cuda"

# ================= Helper Functions =================

def extract_video_frames(video_path, resize_dim):
    """
    Generator that yields resized PIL Images from a video file.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        # Convert BGR (OpenCV) to RGB (PIL)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_pil = Image.fromarray(frame_rgb).resize(resize_dim, Image.Resampling.BICUBIC)
        yield frame_pil

    cap.release()

def build_image_inputs(processor, image_pil):
    """
    Constructs model inputs for a single image frame using the processor.
    """
    # Create a dummy template to extract vision features handled by the processor
    content = [{"type": "image", "image": image_pil}]
    messages = [{"role": "user", "content": content}]
    
    # We only want the vision part, but we use the chat template to ensure consistency
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False
    )
    
    batch = processor(text=[text], images=[image_pil], return_tensors="pt")
    return batch

def build_text_query_inputs(processor, question: str):
    """
    Constructs text-only inputs for Q&A.
    """
    content = [{"type": "text", "text": question}]
    messages = [{"role": "user", "content": content}]
    
    text = processor.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    
    batch = processor(text=[text], images=None, return_tensors="pt")
    return batch

def clone_inference_cache(model, src_cache):
    """
    Deep copies the inference cache state.
    
    Crucial for InfiniteVL:
    When a user asks a question, we must branch off the main streaming state.
    We clone the current streaming state (video history) into a new cache,
    perform the Q&A generation, and then discard the clone. The main stream
    continues unaffected.
    """
    dst_cache = model.allocate_inference_cache(batch_size=1)
    
    for dst_layer, src_layer in zip(dst_cache.layers, src_cache.layers):
        # 1. Handle Sliding Window Attention Layers
        # These layers have a circular buffer (_buf_keys, _buf_values)
        if getattr(src_layer, "is_sliding", False) and hasattr(src_layer, "_buf_keys"):
            dst_layer.size = src_layer.size
            dst_layer.cumulative_length = src_layer.cumulative_length

            if getattr(dst_layer, "capacity", 0) > 0 and src_layer.size > 0:
                L = src_layer.size
                # Copy the active valid data in the buffer
                dst_layer._buf_keys[:, :, :L, :].copy_(src_layer._buf_keys[:, :, :L, :])
                dst_layer._buf_values[:, :, :L, :].copy_(src_layer._buf_values[:, :, :L, :])
                
                # Update the view pointers
                dst_layer.keys = dst_layer._buf_keys[:, :, :L, :]
                dst_layer.values = dst_layer._buf_values[:, :, :L, :]
            elif getattr(dst_layer, "capacity", 0) > 0:
                # Empty buffer
                dst_layer.keys = dst_layer._buf_keys[:, :, :0, :]
                dst_layer.values = dst_layer._buf_values[:, :, :0, :]

        # 2. Handle Linear Attention (Gated DeltaNet) Layers
        # These layers maintain a fixed-size recurrent state
        elif hasattr(src_layer, "recurrent_state") and hasattr(src_layer, "conv_state_q"):
            # Copy convolution states
            if getattr(src_layer, "conv_state_q", None) is not None:
                dst_layer.conv_state_q.copy_(src_layer.conv_state_q)
                dst_layer.conv_state_k.copy_(src_layer.conv_state_k)
                dst_layer.conv_state_v.copy_(src_layer.conv_state_v)

            # Copy recurrent linear attention state
            if getattr(src_layer, "recurrent_state", None) is not None:
                dst_layer.recurrent_state.copy_(src_layer.recurrent_state)

            dst_layer.seq_len = src_layer.seq_len
            dst_layer.start = src_layer.start

    return dst_cache

# ================= Main Inference Logic =================

def main(args):
    # Optimizations for inference
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.benchmark = True
    
    device = torch.device(DEVICE)
    
    # 1. Initialize Query Schedule
    # Queries: List of (frame_index, question_string)
    queries = DEFAULT_QUERIES
    # Use a dictionary for O(1) lookup during the loop
    query_frames_map = {frame: q for frame, q in queries}
    processed_queries_count = 0
    
    print(f"Loading model from: {args.model_path}")
    
    # 2. Load Model & Processor
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=DTYPE,
        low_cpu_mem_usage=True,
        device_map=None, # We manually handle device placement for control
    ).to(device).eval()
    
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    
    # 3. Prepare First Frame (Template & Initialization)
    frame_generator = extract_video_frames(args.video_path, FRAME_RESIZE)
    try:
        first_frame = next(frame_generator)
    except StopIteration:
        raise RuntimeError("Video file is empty or unreadable.")

    # Generate template inputs from the first frame
    warm_inputs = build_image_inputs(processor, first_frame)
    warm_inputs = {k: v.to(device) for k, v in warm_inputs.items()}
    
    full_ids = warm_inputs["input_ids"]
    pixel_values_ref = warm_inputs["pixel_values"]
    grid_thw = warm_inputs["image_grid_thw"]

    # Extract Token IDs
    vstart_id = model.config.vision_start_token_id
    img_id = model.config.image_token_id
    
    # Locate vision tokens in the template
    vstart_pos = (full_ids[0] == vstart_id).nonzero(as_tuple=False)
    if vstart_pos.numel() == 0:
        raise RuntimeError("Could not find vision_start_token_id in template.")
    vstart_idx = vstart_pos[0].item()
    
    # Extract the sequence corresponding to visual tokens [1, 256]
    img_start = vstart_idx + 1
    image_span = full_ids[:, img_start : img_start + IMG_TOKENS_PER_FRAME]
    
    # Construct input_ids for the very first frame: <vstart> + [tokens]
    first_frame_input_ids = torch.cat(
        [full_ids[:, vstart_idx : vstart_idx + 1], image_span], dim=1
    ) # [1, 1 + 256]
    first_frame_len = first_frame_input_ids.shape[1]

    # Construct input_ids for subsequent streaming frames: [tokens] only
    stream_frame_input_ids = image_span.clone() # [1, 256]
    stream_frame_len = stream_frame_input_ids.shape[1]

    # 4. Initialize Visual Components & RoPE
    # This pre-computes buckets and buffers for sliding window attention
    model.visual.set_graph_bucket(warm_inputs["image_grid_thw"])
    model.visual.precompute_window_buffers()
    model.visual.precompute_full_cu_seqlens()

    # Calculate Base Rotary Embeddings
    pos_base_full, rope_deltas = model.model.get_rope_index(
        input_ids=first_frame_input_ids,
        image_grid_thw=grid_thw,
        video_grid_thw=None,
        attention_mask=None,
    )
    model.model.rope_deltas = rope_deltas # Store for internal usage
    
    pos_base_full = pos_base_full.to(device) # [3, 1, 1+256]
    pos_base_stream = pos_base_full[:, :, 1:].clone() # [3, 1, 256] for streaming frames
    
    # Tracking global position for streaming
    current_pos_max = pos_base_full.max().to(torch.long)
    
    # Time grid configuration for RoPE
    # InfiniteVL uses a time-aware position encoding
    second_per_grid_ts = float(getattr(model.config, "second_per_grid_ts", 1.0))
    tokens_per_second = 1.0
    if hasattr(model.config, "vision_config"):
        tokens_per_second = float(getattr(model.config.vision_config, "tokens_per_second", 1.0))
    tokens_per_grid = max(int(round(second_per_grid_ts * tokens_per_second)), 1)

    # 5. CUDA Graph Setup (Static Buffers)
    # We use CUDA Graphs to accelerate the repetitive "streaming frame update"
    cuda_graph = torch.cuda.CUDAGraph()
    graph_captured = False
    
    # Static buffers on GPU
    static_input_ids = stream_frame_input_ids.to(device).clone()
    static_pixel_values = torch.empty_like(pixel_values_ref)
    static_pos_ids = pos_base_stream.clone()
    static_cache_position = torch.empty(stream_frame_len, dtype=torch.long, device=device)

    # Performance metrics
    cum_tokens = 0 # Total visual tokens processed
    prefill_times = []

    print("=" * 60)
    print("Starting Streaming Inference")
    print(f"Video Path: {args.video_path}")
    print(f"Queries: {len(queries)} points defined.")
    print("=" * 60)

    # ------------------------------------------------------------------
    # Phase A: Warmup
    # Run dummy passes to compile kernels and settle the memory allocator
    # ------------------------------------------------------------------
    print("Warmup...")
    warmup_cache = model.allocate_inference_cache(batch_size=1)
    with torch.inference_mode():
        # Warmup First Frame
        warmup_cache_pos = torch.arange(0, first_frame_len, dtype=torch.long, device=device).view(1, -1)
        model(
            input_ids=first_frame_input_ids.to(device),
            position_ids=pos_base_full,
            pixel_values=pixel_values_ref,
            image_grid_thw=grid_thw,
            use_cache=True,
            past_key_values=warmup_cache,
            cache_position=warmup_cache_pos,
            return_dict=True,
        )
        torch.cuda.synchronize()
        
        # Warmup Streaming Frame
        dummy_cache_pos = torch.arange(0, stream_frame_len, dtype=torch.long, device=device)
        model(
            input_ids=stream_frame_input_ids.to(device),
            position_ids=pos_base_stream,
            pixel_values=torch.empty_like(pixel_values_ref),
            image_grid_thw=grid_thw,
            use_cache=True,
            past_key_values=warmup_cache,
            cache_position=dummy_cache_pos,
            return_dict=True,
        )
        torch.cuda.synchronize()
    del warmup_cache # Free warmup memory

    # ------------------------------------------------------------------
    # Phase B: Streaming Loop
    # ------------------------------------------------------------------
    # Allocate the MAIN streaming cache (State)
    stream_cache = model.allocate_inference_cache(batch_size=1)
    cum_tokens = 0
    current_pos_max = pos_base_full.max().to(torch.long)
    
    # --- Step 1: Process First Frame (Prefill) ---
    with torch.inference_mode():
        # Setup position IDs for t=0
        first_pos_ids = pos_base_full.clone()
        cache_position = torch.arange(0, first_frame_len, dtype=torch.long, device=device).view(1, -1)

        t0 = time.time()
        model(
            input_ids=first_frame_input_ids.to(device),
            position_ids=first_pos_ids,
            pixel_values=pixel_values_ref,
            image_grid_thw=grid_thw,
            use_cache=True,
            past_key_values=stream_cache,
            cache_position=cache_position,
            return_dict=True,
        )
        torch.cuda.synchronize()
        t1 = time.time()
        
        cum_tokens += first_frame_len
        prefill_times.append((t1 - t0) * 1000.0)
        print(f"[Frame 0] Initialized. Time: {(t1 - t0)*1000:.2f}ms")

    # --- Step 2: Loop remaining frames ---
    with torch.inference_mode():
        for local_idx, current_frame_pil in enumerate(frame_generator, start=1):
            frame_idx = local_idx
            
            # -------------------------------------------------------
            # Logic Branch: User Query (QA)
            # -------------------------------------------------------
            if frame_idx in query_frames_map:
                question_text = query_frames_map[frame_idx]
                
                print("\n" + "-" * 40)
                print(f"❓ Query at Frame {frame_idx}: {question_text}")
                
                # 1. Clone the State (Branching)
                # We clone stream_cache so Q&A tokens don't pollute the video history
                qa_cache = clone_inference_cache(model, stream_cache)
                qa_cum_tokens = cum_tokens
                qa_pos_max = current_pos_max

                # 2. Prepare Question Inputs
                query_batch = build_text_query_inputs(processor, question_text)
                q_input_ids = query_batch["input_ids"].to(device)
                
                # Prepend <vision_end> to close the visual stream
                vend_token = torch.full((1, 1), model.config.vision_end_token_id, dtype=q_input_ids.dtype, device=device)
                q_ids = torch.cat([vend_token, q_input_ids], dim=1)
                q_len = q_ids.shape[1]

                # Setup QA positions (Just after the latest visual token)
                q_cache_pos = torch.arange(qa_cum_tokens, qa_cum_tokens + q_len, dtype=torch.long, device=device).view(1, -1)
                
                start_pos = qa_pos_max + 1
                q_pos_ids_1d = start_pos + torch.arange(q_len, device=device, dtype=torch.long)
                q_pos_ids = q_pos_ids_1d.to(pos_base_full.dtype).view(1, 1, -1).expand(3, 1, -1)
                
                # Update local QA state pointers
                qa_pos_max = qa_pos_max + q_len
                
                # 3. Question Prefill
                q_out = model(
                    input_ids=q_ids,
                    past_key_values=qa_cache, # Using CLONED cache
                    use_cache=True,
                    cache_position=q_cache_pos,
                    position_ids=q_pos_ids,
                    return_dict=True,
                )
                qa_cum_tokens += q_len
                
                # 4. Answer Generation (Greedy Decoding)
                next_token = torch.argmax(q_out.logits[:, -1, :], dim=-1, keepdim=True)
                generated_ids = []
                
                for _ in range(args.max_new_tokens):
                    if next_token.item() == processor.tokenizer.eos_token_id:
                        break
                    generated_ids.append(next_token)
                    
                    # Update cache/pos for next token
                    step_cache_pos = torch.tensor([qa_cum_tokens], dtype=torch.long, device=device)
                    qa_pos_max = qa_pos_max + 1
                    step_pos_ids = torch.full((3, 1, 1), qa_pos_max.item(), dtype=pos_base_full.dtype, device=device)
                    
                    step_out = model(
                        input_ids=next_token,
                        past_key_values=qa_cache, # Keep using CLONED cache
                        use_cache=True,
                        cache_position=step_cache_pos,
                        position_ids=step_pos_ids,
                        return_dict=True,
                    )
                    qa_cum_tokens += 1
                    next_token = torch.argmax(step_out.logits[:, -1, :], dim=-1, keepdim=True)

                # Decode and Print
                if generated_ids:
                    ans_ids = torch.cat(generated_ids, dim=1)[0]
                    answer = processor.tokenizer.decode(ans_ids, skip_special_tokens=True)
                    print(f"🤖 Answer: {answer}")
                else:
                    print("🤖 Answer: (Empty)")
                
                print("-" * 40 + "\n")
                processed_queries_count += 1
                
                # Stop if all queries are done
                if processed_queries_count >= len(queries):
                    print("All queries processed. Stopping demo.")
                    break

            # -------------------------------------------------------
            # Logic Branch: Streaming Update (Background)
            # -------------------------------------------------------
            
            # Prepare data for current frame
            frame_time_sec = frame_idx / float(args.fps)
            grid_t = int(frame_time_sec / second_per_grid_ts)
            t_offset = grid_t * tokens_per_grid

            frame_batch = build_image_inputs(processor, current_frame_pil)
            frame_pixel_values = frame_batch["pixel_values"].to(device)

            # Update Static Buffers for CUDA Graph
            static_input_ids.copy_(stream_frame_input_ids.to(device))
            static_pixel_values.copy_(frame_pixel_values)
            
            # Update Position IDs (Time Aware)
            static_pos_ids.copy_(pos_base_stream)
            if t_offset != 0:
                p0_view = static_pos_ids.view(3, -1)[0]
                inc = torch.full((IMG_TOKENS_PER_FRAME,), t_offset, device=device, dtype=p0_view.dtype)
                p0_view.index_add_(0, torch.arange(IMG_TOKENS_PER_FRAME, device=device), inc)

            current_pos_max = torch.maximum(current_pos_max, static_pos_ids.max().to(torch.long))

            # Update Cache Position
            cache_pos_vals = torch.arange(cum_tokens, cum_tokens + stream_frame_len, dtype=torch.long, device=device)
            static_cache_position.copy_(cache_pos_vals)

            # Execute Model Update
            torch.cuda.synchronize()
            t_start = time.time()
            
            if not graph_captured:
                # First time: Capture the graph
                with torch.cuda.graph(cuda_graph):
                    _ = model(
                        input_ids=static_input_ids,
                        position_ids=static_pos_ids,
                        pixel_values=static_pixel_values,
                        image_grid_thw=grid_thw,
                        use_cache=True,
                        past_key_values=stream_cache, # Using MAIN stream cache
                        cache_position=static_cache_position,
                        return_dict=True,
                    )
                graph_captured = True
            else:
                # Subsequent times: Replay graph (Zero overhead)
                cuda_graph.replay()
                
            torch.cuda.synchronize()
            t_end = time.time()
            
            # Update State Trackers
            cum_tokens += stream_frame_len
            prefill_times.append((t_end - t_start) * 1000.0)

            # Logging
            if frame_idx % 100 == 0:
                fps = 1000.0 / prefill_times[-1]
                print(f"[Frame {frame_idx:4d}] Streaming Update: {prefill_times[-1]:.2f} ms ({fps:.1f} FPS) | Total Tokens: {cum_tokens}")

    # Summary
    if prefill_times:
        avg_time = sum(prefill_times) / len(prefill_times)
        print("\n" + "=" * 60)
        print(f"Performance Summary:")
        print(f"Average Latency per Frame: {avg_time:.3f} ms")
        print(f"Average Throughput: {1000/avg_time:.2f} FPS")
        print("=" * 60)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="InfiniteVL Streaming Inference Demo")
    parser.add_argument("--model_path", type=str, required=True, help="Path to InfiniteVL model weights")
    parser.add_argument("--video_path", type=str, required=True, help="Path to video file")
    parser.add_argument("--fps", type=int, default=30, help="Video FPS for time embedding")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Max tokens for QA generation")
    
    args = parser.parse_args()
    
    # Environment Setup
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")
    
    main(args)
