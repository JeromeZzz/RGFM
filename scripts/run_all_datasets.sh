#!/bin/bash

# 定义所有数据集列表
# 注意：rel-amazon 和 rel-stackex 训练时间非常长，请根据硬件情况决定是否保留
DATASETS=(
    "rel-f1"
    "rel-event"
    "rel-avito"
    "rel-trial"
    "rel-hm"
    #"rel-stackex"  # 数据量大，训练慢，默认注释
    #"rel-amazon"   # 数据量极大，默认注释
)

# 遍历每一个数据集
for DATASET in "${DATASETS[@]}"
do
    echo "========================================================"
    echo "Start training: $DATASET"
    echo "========================================================"

    # 获取该数据集下的第一个任务（这里简化处理，默认跑该数据集的第一个任务）
    # 如果你想跑特定任务，需要写更复杂的逻辑或为每个数据集指定任务
    # 这里我们假设你只想验证能不能跑通，所以用 rel-f1 的 driver-position 作为模板，
    # 实际应用中建议针对不同数据集指定不同的 task_name。
    
    # 更加通用的方式是：捕捉错误，继续下一个
    # 下面是一个通用的训练命令模板
    
    if [ "$DATASET" == "rel-f1" ]; then
        TASK="driver-position"
    elif [ "$DATASET" == "rel-event" ]; then
        TASK="user-attendance"
    elif [ "$DATASET" == "rel-avito" ]; then
        TASK="ad-ctr"
    elif [ "$DATASET" == "rel-trial" ]; then
        TASK="site-success"
    elif [ "$DATASET" == "rel-hm" ]; then
        TASK="user-churn"
    else
        echo "Unkown task, escape $DATASET"
        continue
    fi

    echo "   Task: $TASK"
    
    # 运行 Python 训练脚本
    # 这里的参数可以根据你的 config.yaml 进行调整
    python relbdata/train_on_relbench.py \
        --dataset_name "$DATASET" \
        --task_name "$TASK" \
        --num_epochs 10 \
        --batch_size 512

    if [ $? -eq 0 ]; then
        echo "Dataset $DATASET Train Finishede"
    else
        echo "Dataset $DATASET Train Failed"
    fi
    
    echo ""
done

echo "Finished"