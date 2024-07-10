#!/bin/bash

folders=()
# 读取output文件夹下所有包含superni字串的文件夹
while IFS= read -r -d '' folder; do
    folders+=("$folder")
done < <(find /workspace/output -type d -name "*superni*" -print0)

# 每次读取的文件夹数量
batch_size=4

# 计算总共需要读取的次数
total_batches=$(((${#folders[@]} + batch_size - 1) / batch_size))

# 当前读取的起始索引
start_index=0

# 循环读取并打印文件夹
for ((batch = 1; batch <= total_batches; batch++)); do
    end_index=$((start_index + batch_size - 1))
    end_index=$((end_index < ${#folders[@]} ? end_index : ${#folders[@]} - 1))

    for ((i = start_index; i <= end_index; i++)); do
        echo $((i % batch_size)) ${folders[$i]}
        bash /workspace/eval_llama2_7b_superni_tydiqa.sh $((i % batch_size)) ${folders[$i]} &
    done
    wait
    sleep 1m
    start_index=$((end_index + 1))
done
