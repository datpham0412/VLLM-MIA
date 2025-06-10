# Step X: Launch MiniGPT-4 Model Worker
echo "Launching MiniGPT-4 model worker..."
export PYTHONPATH=/fred/oz402/nhnguyen/Model/MiniGPT-4:$PYTHONPATH
python3 /fred/oz402/nhnguyen/PJ/VLLM-MIA/minigpt_worker.py \
  --cfg-path /fred/oz402/nhnguyen/PJ/VLLM-MIA/MiniGPT-4/eval_configs/minigpt4_eval.yaml \
  --model-path /fred/oz402/nhnguyen/Model/vicuna_weights \
  --model-name minigpt-4 \
    --controller-address http://192.168.22.201:21101 \
  --port 41003 \
  > llama_logs/worker_minigpt.log 2>&1 &

