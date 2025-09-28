"""
KumoRFM API服务器
提供REST API接口
"""

from flask import Flask, request, jsonify
from datetime import datetime
import torch
import logging
from pathlib import Path
import traceback

from config.model_config import KumoRFMConfig, TaskConfig
from models.kumorfm import KumoRFM
from inference.predictor import KumoRFMPredictor
from data.temporal_graph import TemporalHeterogeneousGraph
from data.database import MockDatabase
from data.graph_converter import MockGraphConverter

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 创建Flask应用
app = Flask(__name__)

# 全局变量存储模型
model = None
predictor = None
graph = None
database = None


def load_model(checkpoint_path: Optional[str] = None):
    """加载模型"""
    global model, predictor

    # 数据库模式（示例）
    database_schema = {
        'users': {'user_id': 'categorical', 'features': 'numerical'},
        'items': {'item_id': 'categorical', 'features': 'numerical'},
        'interactions': {'user_id': 'categorical', 'item_id': 'categorical',
                         'timestamp': 'time', 'value': 'numerical'}
    }

    if checkpoint_path and Path(checkpoint_path).exists():
        # 从检查点加载
        predictor = KumoRFMPredictor.from_checkpoint(
            checkpoint_path, database_schema
        )
        model = predictor.model
        logger.info(f"从 {checkpoint_path} 加载模型")
    else:
        # 创建新模型
        config = KumoRFMConfig()
        model = KumoRFM(config, database_schema)
        predictor = KumoRFMPredictor(model, config)
        logger.info("创建新模型")


def load_data():
    """加载数据（示例）"""
    global graph, database

    # 创建模拟数据
    database = MockDatabase()
    converter = MockGraphConverter()
    graph = converter.convert(database)

    logger.info("数据已加载")


@app.route('/health', methods=['GET'])
def health_check():
    """健康检查"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'data_loaded': graph is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    预测接口

    请求体示例:
    {
        "entity_type": "users",
        "entity_id": 123,
        "prediction_time": "2024-01-01T00:00:00",
        "task_type": "regression",
        "target_column": "value",
        "aggregation": "sum",
        "time_window_start": -7,
        "time_window_end": 0,
        "num_context": 10
    }
    """
    try:
        if not model or not graph:
            return jsonify({'error': 'Model or data not loaded'}), 503

        # 解析请求
        data = request.json

        # 构建任务配置
        task_config = TaskConfig(
            task_type=data.get('task_type', 'regression'),
            target_column=data.get('target_column'),
            aggregation=data.get('aggregation', 'mean'),
            time_window_start=data.get('time_window_start'),
            time_window_end=data.get('time_window_end')
        )

        # 解析实体和时间
        entity = (data['entity_type'], data['entity_id'])
        prediction_time = datetime.fromisoformat(data['prediction_time'])

        # 执行预测
        result = predictor.predict(
            graph,
            entity,
            prediction_time,
            task_config,
            num_context=data.get('num_context', 10),
            return_attention=data.get('return_attention', False)
        )

        return jsonify(result)

    except Exception as e:
        logger.error(f"预测错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    批量预测接口

    请求体示例:
    {
        "entities": [
            {"type": "users", "id": 123},
            {"type": "users", "id": 124}
        ],
        "prediction_time": "2024-01-01T00:00:00",
        "task_config": {...},
        "batch_size": 32
    }
    """
    try:
        if not model or not graph:
            return jsonify({'error': 'Model or data not loaded'}), 503

        data = request.json

        # 解析实体列表
        entities = [(e['type'], e['id']) for e in data['entities']]

        # 解析时间
        prediction_time = datetime.fromisoformat(data['prediction_time'])

        # 构建任务配置
        task_config_data = data.get('task_config', {})
        task_config = TaskConfig(**task_config_data)

        # 批量预测
        results = predictor.batch_predict(
            graph,
            entities,
            prediction_time,
            task_config,
            batch_size=data.get('batch_size', 32)
        )

        # 格式化结果
        formatted_results = []
        for i, result in enumerate(results):
            formatted_results.append({
                'entity': {'type': entities[i][0], 'id': entities[i][1]},
                'prediction': result
            })

        return jsonify({'results': formatted_results})

    except Exception as e:
        logger.error(f"批量预测错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/predict_pql', methods=['POST'])
def predict_pql():
    """
    PQL预测接口

    请求体示例:
    {
        "query": "PREDICT SUM(value, -7, 0) FOR user_id = 123",
        "prediction_time": "2024-01-01T00:00:00"
    }
    """
    try:
        if not model or not graph or not database:
            return jsonify({'error': 'Model, data or database not loaded'}), 503

        data = request.json

        # 解析时间
        prediction_time = None
        if 'prediction_time' in data:
            prediction_time = datetime.fromisoformat(data['prediction_time'])

        # 执行PQL预测
        results = predictor.predict_from_pql(
            database,
            graph,
            data['query'],
            prediction_time
        )

        return jsonify(results)

    except Exception as e:
        logger.error(f"PQL预测错误: {str(e)}\n{traceback.format_exc()}")
        return jsonify({'error': str(e)}), 500


@app.route('/model/info', methods=['GET'])
def model_info():
    """获取模型信息"""
    if not predictor:
        return jsonify({'error': 'Model not loaded'}), 503

    info = predictor.get_model_info()
    return jsonify(info)


@app.route('/model/reload', methods=['POST'])
def reload_model():
    """重新加载模型"""
    try:
        data = request.json or {}
        checkpoint_path = data.get('checkpoint_path')

        load_model(checkpoint_path)
        load_data()

        return jsonify({'message': 'Model reloaded successfully'})

    except Exception as e:
        logger.error(f"重新加载模型错误: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='KumoRFM API Server')
    parser.add_argument('--port', type=int, default=8000, help='Port to run on')
    parser.add_argument('--host', type=str, default='0.0.0.0', help='Host to run on')
    parser.add_argument('--checkpoint', type=str, help='Model checkpoint path')
    parser.add_argument('--debug', action='store_true', help='Run in debug mode')

    args = parser.parse_args()

    # 加载模型和数据
    logger.info("初始化模型和数据...")
    load_model(args.checkpoint)
    load_data()

    # 启动服务器
    logger.info(f"在 {args.host}:{args.port} 启动API服务器")
    app.run(host=args.host, port=args.port, debug=args.debug)

from typing import Optional