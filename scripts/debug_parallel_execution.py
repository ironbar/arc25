import argparse
import glob
from tqdm import tqdm

from arc25.parallel_code_execution import run_code_from_predictions
from arc25.metrics import aggregate_metrics, error_analysis
from arc25.utils import load_arc_dataset_with_solutions, load_json


def main(args=None):
    parser = argparse.ArgumentParser(description="Debug parallel execution")
    parser.add_argument("--dataset_path", type=str, 
                       default="/mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json",
                       help="Path to the dataset file")
    parser.add_argument("--prediction_path", type=str,
                       default="/mnt/hdd0/Kaggle/arc25/predictions/2025-08-28-base-model/evaluation/8preds_2025_08_31_09_47_48_predictions.json",
                       help="Path to the prediction file")
    parser.add_argument("--n_jobs", type=int, default=-1,
                       help="Number of parallel jobs")
    parser.add_argument("--timeout_duration", type=int, default=1,
                       help="Timeout duration in seconds")
    parser.add_argument("--batch_size", type=int, default=5000,
                       help="Batch size for processing")
    
    config = parser.parse_args(args)
    dataset = load_arc_dataset_with_solutions(config.dataset_path)

    predictions = load_all_predictions(config.prediction_path)
    n_preds = len(list(predictions.values())[0]['text_predictions'])
    print(f"Loaded {len(predictions)} tasks with {n_preds} predictions each.")

    tasks, task_ids, text_predictions, data_augmentation_params = [], [], [], []
    for task_id, task_preds in predictions.items():
        tasks.extend([dataset[task_id]] * len(task_preds['text_predictions']))
        task_ids.extend([task_id] * len(task_preds['text_predictions']))
        text_predictions.extend(task_preds['text_predictions'])
        data_augmentation_params.extend(task_preds['data_augmentation_params'])

    results = run_code_from_predictions(
        tasks, task_ids, text_predictions, data_augmentation_params,
        n_jobs=config.n_jobs, timeout_duration=config.timeout_duration, batch_size=config.batch_size)
    df = aggregate_metrics(results)
    error_analysis(results)
    print(df.iloc[-1:])


def load_all_predictions(path_pattern):
    filepaths = glob.glob(path_pattern)
    predictions = dict()
    for filepath in tqdm(filepaths, desc="Loading predictions", disable=len(filepaths)<=1):
        preds = load_json(filepath)
        for task_id, outputs in preds.items():
            if task_id not in predictions:
                predictions[task_id] = dict(text_predictions=[], data_augmentation_params=[])
            if isinstance(outputs, dict):
                predictions[task_id]['text_predictions'].extend(outputs['text_predictions'])
                data_augmentation_params = outputs.get('data_augmentation_params', None)
                if data_augmentation_params is not None and data_augmentation_params['color_map'] is not None:
                    data_augmentation_params['color_map'] = {int(k): int(v) for k, v in data_augmentation_params['color_map'].items()}
                predictions[task_id]['data_augmentation_params'].extend([data_augmentation_params]*len(outputs['text_predictions']))
            else:
                predictions[task_id]['text_predictions'].extend(outputs)
                predictions[task_id]['data_augmentation_params'].extend([None] * len(outputs))  # Assuming no params for old format
    return predictions


if __name__ == '__main__':
    main()
