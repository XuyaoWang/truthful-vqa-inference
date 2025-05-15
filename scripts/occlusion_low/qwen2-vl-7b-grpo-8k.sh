export VLLM_USE_V1=0
# export CUDA_VISIBLE_DEVICES=4,5,6,7

# model
MODEL_PATH=lmms-lab/Qwen2-VL-7B-GRPO-8k
MODEL_NAME=qwen2-vl-7b-grpo-8k

# sampling
TEMPERATURE=0.0
TOP_P=1
REPETITION_PENALTY=1.0
MAX_TOKENS=4096

# server
SERVER_BACKEND=vllm
PORT=8020
HOST=localhost
PP_SIZE=1
TP_SIZE=4
GPU_UTILIZATION=0.9
LIMIT_MM_PER_PROMPT="image=1,video=0"
MAX_SEQ_LEN=32768
ENABLE_PREFIX_CACHING=true
DTYPE=bfloat16
DISABLE_LOG_STATS=true
DISABLE_LOG_REQUESTS=true
DISABLE_FASTAPI_DOCS=true
NUM_WORKERS=20

# benchmark
BENCHMARK=truthfulvqa-low
DATA_PATH=/aifs4su/yaodong/xuyao/workshop-truthful-vqa/TruthfulVQA
SPLIT=validation

# Set other default parameters
RESULTS_DIR="./results"
CACHE_DIR="./cache/${MODEL_NAME}"

python main.py \
  --model-path ${MODEL_PATH} \
  --model-name ${MODEL_NAME} \
  --server-backend ${SERVER_BACKEND} \
  --port ${PORT} \
  --host ${HOST} \
  --pipeline-parallel-size ${PP_SIZE} \
  --tensor-parallel-size ${TP_SIZE} \
  --gpu-memory-utilization ${GPU_UTILIZATION} \
  --limit-mm-per-prompt "${LIMIT_MM_PER_PROMPT}" \
  --max-seq-len ${MAX_SEQ_LEN} \
  --dtype ${DTYPE} \
  --benchmark ${BENCHMARK} \
  --data-path ${DATA_PATH} \
  --split ${SPLIT} \
  --num-workers ${NUM_WORKERS} \
  --temperature ${TEMPERATURE} \
  --top-p ${TOP_P} \
  --repetition-penalty ${REPETITION_PENALTY} \
  --max-tokens ${MAX_TOKENS} \
  --results-dir ${RESULTS_DIR} \
  --cache-dir ${CACHE_DIR} \
  $([ "$ENABLE_PREFIX_CACHING" = "true" ] && echo "--enable-prefix-caching" || echo "") \
  $([ "$DISABLE_LOG_STATS" = "true" ] && echo "--disable-log-stats" || echo "") \
  $([ "$DISABLE_LOG_REQUESTS" = "true" ] && echo "--disable-log-requests" || echo "") \
  $([ "$DISABLE_FASTAPI_DOCS" = "true" ] && echo "--disable-fastapi-docs" || echo "")