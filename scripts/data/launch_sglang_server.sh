#!/usr/bin/env bash
set -euo pipefail

# Requires SGLang to be installed separately (not in requirements.txt):
#   pip install "sglang[all]"
# See https://docs.sglang.ai/get_started/install.html for details.

model_path=${model_path:-Qwen/Qwen3-4B}
gpu_list=${gpu_list:-${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}}
IFS=',' read -r -a gpu_ids <<< "${gpu_list}"
num_workers=${num_workers:-${#gpu_ids[@]}}
start_port=${start_port:-30000}
start_nccl_port=${start_nccl_port:-31000}
host=${host:-0.0.0.0}
dtype=${dtype:-bfloat16}
mem_frac=${mem_frac:-0.9}
log_dir=${log_dir:-logs/sglang_qwen3_4b}
heartbeat_interval=${heartbeat_interval:-300}

if (( num_workers < 1 )); then
    echo "num_workers must be at least 1" >&2
    exit 1
fi

if (( num_workers > ${#gpu_ids[@]} )); then
    echo "num_workers (${num_workers}) cannot exceed available GPUs in gpu_list (${gpu_list})" >&2
    exit 1
fi

get_host_ip() {
    local host_ip=""

    if command -v hostname > /dev/null 2>&1; then
        host_ip=$(hostname -I 2> /dev/null | awk '{print $1}')
    fi

    if [[ -z "${host_ip}" ]] && command -v ip > /dev/null 2>&1; then
        host_ip=$(
            ip -4 route get 1.1.1.1 2> /dev/null | awk '
                /src/ {
                    for (i = 1; i <= NF; i++) {
                        if ($i == "src") {
                            print $(i + 1)
                            exit
                        }
                    }
                }
            '
        )
    fi

    if [[ -z "${host_ip}" ]]; then
        host_ip=127.0.0.1
    fi

    printf '%s\n' "${host_ip}"
}

mkdir -p "${log_dir}"

host_ip=$(get_host_ip)
pids=()
ports=()
nccl_ports=()
heartbeat_pid=""

print_heartbeat() {
    local timestamp alive_count status pid port nccl_port
    timestamp=$(date '+%Y-%m-%d %H:%M:%S')
    alive_count=0

    echo "[${timestamp}] heartbeat"
    for ((idx = 0; idx < ${#pids[@]}; idx++)); do
        pid=${pids[$idx]}
        port=${ports[$idx]}
        nccl_port=${nccl_ports[$idx]}
        status=dead
        if kill -0 "${pid}" > /dev/null 2>&1; then
            status=alive
            alive_count=$((alive_count + 1))
        fi
        echo "  worker_index=${idx} pid=${pid} port=${port} nccl_port=${nccl_port} status=${status}"
    done
    echo "  alive_workers=${alive_count}/${#pids[@]}"
}

heartbeat_loop() {
    while true; do
        sleep "${heartbeat_interval}"
        print_heartbeat
    done
}

cleanup() {
    if [[ -n "${heartbeat_pid}" ]] && kill -0 "${heartbeat_pid}" > /dev/null 2>&1; then
        kill "${heartbeat_pid}" > /dev/null 2>&1 || true
    fi
    for pid in "${pids[@]:-}"; do
        if kill -0 "${pid}" > /dev/null 2>&1; then
            kill "${pid}" > /dev/null 2>&1 || true
        fi
    done
    wait || true
}

trap cleanup INT TERM EXIT

for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    gpu_id=${gpu_ids[$worker_id]}
    gpu_log_id=${gpu_id//\//_}
    port=$((start_port + worker_id))
    nccl_port=$((start_nccl_port + worker_id))
    log_file=${log_dir}/worker_${host_ip}_gpu_${gpu_log_id}_port_${port}.log

    echo "Starting sglang worker ip=${host_ip} worker=${worker_id} gpu=${gpu_id} port=${port} nccl_port=${nccl_port} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu_id}" sglang serve \
        --model-path "${model_path}" \
        --host "${host}" \
        --port "${port}" \
        --nccl-port "${nccl_port}" \
        --dtype "${dtype}" \
        --mem-fraction-static "${mem_frac}" \
        "$@" > "${log_file}" 2>&1 &
    pids+=("$!")
    ports+=("${port}")
    nccl_ports+=("${nccl_port}")
done

echo "Workers launched:"
for ((worker_id = 0; worker_id < num_workers; worker_id++)); do
    gpu_id=${gpu_ids[$worker_id]}
    port=$((start_port + worker_id))
    nccl_port=$((start_nccl_port + worker_id))
    echo "  worker=${worker_id} gpu=${gpu_id} http://${host_ip}:${port} nccl_port=${nccl_port}"
done
echo "Heartbeat interval: ${heartbeat_interval}s"
print_heartbeat
heartbeat_loop &
heartbeat_pid=$!

wait "${pids[@]}"
