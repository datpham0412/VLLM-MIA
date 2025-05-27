#!/bin/bash
# Point Python at your local adapter code (no install needed)
export PYTHONPATH=/fred/oz402/tiend/VLLM-MIA/llama_adapter_v21:$PYTHONPATH
# Create llama_logs directory if it doesn't exist
mkdir -p llama_logs

# Step 1: Launch Controller (new port to avoid conflict with LLaVA)
echo "Launching LLaMA controller..."
python3 -m llava.serve.controller \
  --host 0.0.0.0 \
  --port 21101 \
  > llama_logs/controller.log 2>&1 &

sleep 2  # Wait for controller to start

# Step 2: Launch Meta-Llama Model Worker
echo "Launching Meta-Llama model worker..."
python3 -m llava.serve.model_worker \
  --model-path /fred/oz402/tiend/VLLM-MIA/target_models/Llama-2-13b-chat-hf \
  --model-name Llama-2-13b-chat-hf \
  --controller-address http://localhost:21101 \
  --port 41000 \
  > llama_logs/worker_meta.log 2>&1 &

# Step 3: Launch LLaMA-Adapter Worker
echo "Launching LLaMA-Adapter model worker..."
python3 -m llava.serve.model_worker \
  --model-path /fred/oz402/tiend/VLLM-MIA/llama_adapter_v21 \
  --model-base   /fred/oz402/tiend/VLLM-MIA/target_models/Llama-2-13b-chat-hf \
  --model-name BIAS-7B \
  --controller-address http://localhost:21101 \
  --port 41001 \
  > llama_logs/worker_adapter.log 2>&1 &

sleep 5  # Wait for both workers to register

# Step 4: Launch Gradio Web UI for LLaMA & Adapter
echo "Launching Gradio web server..."
python3 -m llava.serve.gradio_web_server \
  --controller-url http://localhost:21101 \
  --model-list-mode reload \
  --port 7870 \
  > llama_logs/gradio.log 2>&1 &

echo "✅ LLaMA services launched. Logs can be found in llama_logs directory:"
echo "   - Controller:        llama_logs/controller.log"
echo "   - Meta-Llama worker: llama_logs/worker_meta.log"
echo "   - Adapter worker:    llama_logs/worker_adapter.log"
echo "   - Gradio UI:         llama_logs/gradio.log"
