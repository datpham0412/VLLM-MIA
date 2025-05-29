#!/bin/bash

# Create logs directory if it doesn't exist
mkdir -p logs

# Step 1: Launch Controller
echo "Launching controller..."
python3 -m llava.serve.controller \
  --host 0.0.0.0 \
  --port 21001 \
  > logs/controller.log 2>&1 &


# Step 2: Launch Model Worker
echo "Launching model worker..."
python3 -m llava.serve.model_worker \
  --model-path /fred/oz402/aho/VLLM-MIA/target_models/llava-v1.5-7b \
  --model-name llava-v1.5-7b \
  --controller-address http://localhost:21001 \
  --port 40000 \
  > logs/model_worker.log 2>&1 &


# Step 3: Launch Gradio Web UI
echo "Launching gradio web server..."
python3 -m llava.serve.gradio_web_server \
  --controller-url http://localhost:21001 \
  --model-list-mode reload \
  --port 7860 \
  > logs/gradio_web_server.log 2>&1 &

echo "✅ All services launched. Logs can be found in logs directory"
# echo "✅ Controller and Model Worker launched. Logs: controller.log, model_worker.log"
