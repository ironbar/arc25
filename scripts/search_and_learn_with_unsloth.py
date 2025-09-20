"""
Search and learn with unsloth
"""
import os
from arc25.utils import set_cuda_visible_devices_to_least_used_gpu_if_undefined
from arc25.logging import configure_logging, logging, log_execution_time

configure_logging()
set_cuda_visible_devices_to_least_used_gpu_if_undefined()

# Add VLLM specific environment variables to avoid common issues
os.environ['VLLM_USE_MODELSCOPE'] = 'False'
os.environ['VLLM_WORKER_MULTIPROC_METHOD'] = 'spawn'
os.environ['TOKENIZERS_PARALLELISM'] = 'false' # to avoid warnings, so far I haven't seen any slowdown

from unsloth import FastLanguageModel
from unsloth.chat_templates import train_on_responses_only

from dataclasses import dataclass
import time
import wandb
from datasets import Dataset
import tyro
from vllm import  SamplingParams
from trl import SFTConfig, SFTTrainer
from tqdm.auto import tqdm
from accelerate import Accelerator

from arc25.encoders import create_grid_encoder
from arc25.utils import load_arc_dataset_with_solutions, set_random_seed, write_json
from arc25.data_augmentation import apply_data_augmentation, get_random_data_augmentation_params
from arc25.prompting import create_prompt_from_task, pretty_print_prompt
from arc25.metrics import aggregate_metrics, error_analysis
from arc25.logging import log_execution_time, configure_logging
from arc25.parallel_code_execution import CodeRunner

logger = logging.getLogger(__name__)


@dataclass
class Config:
    # base model
    model_path: str = "/home/gbarbadillo/models/Llama-3.1-ARC-Potpourri-Induction-8B"
    load_in_4bit: bool = False
    max_seq_length: int = 9670 # 8635 + output tokens
    grid_encoder: str = 'ColorNameEncoder()'
    gpu_memory_utilization: float = 0.8 # best value for Kaggle L4 GPU
    # LoRA
    lora_r: int = 16
    use_rslora: bool = True
    # dataset
    dataset_path: str = "/mnt/hdd0/Kaggle/arc25/data/arc-prize-2024/arc-agi_evaluation_challenges.json"
    output_dir: str = "/mnt/hdd0/Kaggle/arc25/trainings/2025-09-06-debug-unsloth/first-steps"
    # search and learn hyperparameters
    use_data_augmentation: bool = True
    max_epochs: int = 0
    inference_batch_size: int = 8
    initial_predictions: int = 32
    predictions_per_epoch: int = 8
    training_batch_size: int = 1
    n_jobs: int = -1
    timeout_duration: int = 5 # seconds for code execution timeout
    # training hyperparameters
    learning_rate: float = 1e-5
    lr_scheduler_type: str = 'constant_with_warmup'
    train_max_seq_length: int = 9670
    # sampling hyperparameters
    temperature: float = 1.0
    top_p: float = 0.95
    max_output_tokens: int = 1024
    # other
    log_to_wandb: bool = True


def main():
    cfg = tyro.cli(Config, description="Search and learn with unsloth")
    accelerator = Accelerator() # seems to need to do this if I want to use logging
    configure_logging()
    if cfg.log_to_wandb:
        wandb.init(project=os.path.basename(os.path.dirname(cfg.output_dir)),
                   name=os.path.basename(cfg.output_dir), config=cfg, reinit=True,
                   dir=cfg.output_dir, save_code=True)
    t0 = time.time()
    logger.info(f'Running search and learn with config: {cfg}')

    dataset = load_arc_dataset_with_solutions(cfg.dataset_path)
    task_ids = list(dataset.keys())
    logger.info(f"Loaded {len(dataset)} tasks from {cfg.dataset_path}")

    llm, tokenizer = FastLanguageModel.from_pretrained(
        cfg.model_path, load_in_4bit=cfg.load_in_4bit, max_seq_length=cfg.max_seq_length,
        fast_inference=True, gpu_memory_utilization=cfg.gpu_memory_utilization)
    grid_encoder = create_grid_encoder(cfg.grid_encoder)
    code_runner = CodeRunner(n_jobs=cfg.n_jobs)

    results = search(dataset, task_ids, llm, tokenizer, grid_encoder, lora_request=None,
        inference_batch_size=cfg.inference_batch_size, n_predictions=cfg.initial_predictions,
        use_data_augmentation=cfg.use_data_augmentation, print_first_prompt=True,
        timeout_duration=cfg.timeout_duration, max_tokens=cfg.max_output_tokens,
        temperature=cfg.temperature, top_p=cfg.top_p, code_runner=code_runner)
    print(aggregate_metrics(results))

    model = create_peft_model(llm, lora_r=cfg.lora_r, use_rslora=cfg.use_rslora) # initialize peft model
    for task_id in tqdm(task_ids, desc="Tasks", unit="task"):
        if not cfg.max_epochs:
            continue
        print('\n'*2 + '='*80 + f'\nTask {task_id}\n' + '='*80)
        logger.info(f'Search and learn for task {task_id} ({task_ids.index(task_id)+1}/{len(task_ids)})')
        task = dataset[task_id]
        model = create_peft_model(llm, lora_r=cfg.lora_r, use_rslora=cfg.use_rslora, model=model) # reset the LoRA weights for each task
        lora_request = None
        task_results = results[task_id]
        for epoch in range(1, cfg.max_epochs + 1):
            logger.info(f'Prepare training data for task {task_id} epoch {epoch}/{cfg.max_epochs}')
            relabeled_tasks = create_hindsight_relabeled_tasks(task_results, task)
            training_prompts = create_training_prompts(relabeled_tasks, grid_encoder, tokenizer)
            lora_request = learn(
                training_prompts, model, tokenizer, cfg.output_dir, learning_rate=cfg.learning_rate,
                lr_scheduler_type=cfg.lr_scheduler_type, max_seq_length=cfg.train_max_seq_length,
                previous_lora_request=lora_request)

            logger.info(f'Searching solutions for epoch {epoch}')
            task_results = search(dataset, [task_id], llm, tokenizer, grid_encoder, lora_request,
                inference_batch_size=cfg.inference_batch_size, n_predictions=cfg.initial_predictions,
                use_data_augmentation=cfg.use_data_augmentation,
                timeout_duration=cfg.timeout_duration, max_tokens=cfg.max_output_tokens,
                temperature=cfg.temperature, top_p=cfg.top_p, code_runner=code_runner)
            print(aggregate_metrics(task_results).head(1).round(3))
            task_results = task_results[task_id]
            results[task_id].extend(task_results)
            # TODO: stop criteria
    # TODO: select best predictions and prepare submission
    error_analysis(results)
    save_results(results, cfg.output_dir, cfg.log_to_wandb)
    if cfg.log_to_wandb:
        wandb.log({"execution_time": time.time() - t0})
        log_metrics_evolution(results)


@log_execution_time
def create_peft_model(llm, lora_r, use_rslora, model=None):
    if model is not None: model.unload()
    model = FastLanguageModel.get_peft_model(
        llm,
        r = lora_r, # Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
        target_modules = ['k_proj', 'q_proj', 'v_proj', 'o_proj'], # TODO: learn more about which modules to choose
        lora_alpha = 64,
        lora_dropout = 0, # Supports any, but = 0 is optimized
        bias = "none",    # Supports any, but = "none" is optimized
        # [NEW] "unsloth" uses 30% less VRAM, fits 2x larger batch sizes!
        use_gradient_checkpointing = False, # True or "unsloth" for very long context # TODO: maybe I have to enable this
        use_rslora = use_rslora,  # We support rank stabilized LoRA
        loftq_config = None, # And LoftQ
    )
    return model


@log_execution_time
def search(dataset, task_ids, llm, tokenizer, grid_encoder, lora_request,
           inference_batch_size, n_predictions, use_data_augmentation, code_runner,
           print_first_prompt=False, timeout_duration=5,
           max_tokens=2048, temperature=1.0, top_p=0.95):
    set_random_seed(None)
    prompts, data_augmentation_params, inference_task_ids = [], [], []
    for task_id in task_ids:
        for _ in range(n_predictions // inference_batch_size):
            task = dataset[task_id]
            if use_data_augmentation:
                params = get_random_data_augmentation_params()
                task = apply_data_augmentation(task, **params)
            else:
                params = None
            data_augmentation_params.extend([params] * inference_batch_size)
            prompt = create_prompt_from_task(
                task, grid_encoder=grid_encoder, tokenizer=tokenizer, shuffle_train_samples=True)
            prompts.append(prompt)
            inference_task_ids.extend([task_id] * inference_batch_size)

    if print_first_prompt: pretty_print_prompt(prompts[0])

    sampling_params = SamplingParams(n=inference_batch_size, temperature=temperature,
                                     top_p=top_p, max_tokens=max_tokens)
    generations = llm.fast_generate(prompts, sampling_params, lora_request=lora_request)
    logger.info(f'Generated {len(generations)} generations with batch size {inference_batch_size}')
    text_predictions = []
    for generation in generations:
        for output in generation.outputs:
            text_predictions.append(output.text)

    results = code_runner.run(
        [dataset[task_id] for task_id in inference_task_ids], inference_task_ids,
        text_predictions, data_augmentation_params, timeout_duration=timeout_duration)
    return results


@log_execution_time
def learn(training_prompts, model, tokenizer, output_dir, learning_rate, lr_scheduler_type,
          max_seq_length, previous_lora_request):
    # train_dataset = Dataset.from_dict({'text': training_prompts})
    if not training_prompts:
        logger.warning('No training prompts provided, skipping learning step')
        return previous_lora_request
    training_tokens = tokenizer(training_prompts)
    train_dataset = Dataset.from_dict(training_tokens)
    logger.info(f'Training on {len(train_dataset)} samples')
    logger.info(f'Training sequence lengths: {[len(ids) for ids in training_tokens["input_ids"]]}')
    trainer = SFTTrainer(
        model = model,
        tokenizer = tokenizer,
        train_dataset = train_dataset,
        dataset_text_field = "input_ids",
        max_seq_length = max_seq_length,
        packing = False, # Can make training 5x faster for short sequences.
        args = SFTConfig(
            per_device_train_batch_size = 1,
            gradient_accumulation_steps = 1,
            warmup_ratio=0.1,
            num_train_epochs=1,
            save_strategy='no',
            learning_rate = learning_rate,
            logging_steps = 1,
            optim = "adamw_torch_fused",
            weight_decay = 0.01,
            lr_scheduler_type = lr_scheduler_type,
            # seed = 3407,
            output_dir = output_dir,
            report_to = "none", # Use this for WandB etc
            # added to fix this error: https://wandb.ai/guillermobarbadillo/2025-09-07-search-and-learn/runs/sqydbbim/logs
            dataloader_num_workers = 4,
            dataloader_persistent_workers = True,
        ),
    )
    logger.warning('Training on responses only, make sure the prompt templates are correct!. So far is only implemented for LLaMA style templates.')
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<|start_header_id|>user<|end_header_id|>\n\n",
        response_part = "<|start_header_id|>assistant<|end_header_id|>\n\n",
    )
    try:
        trainer_stats = trainer.train()
        logger.info(f'Trainer stats: {trainer_stats}')
        lora_filepath = os.path.join(output_dir, "LoRA")
        model.save_lora(lora_filepath)
        lora_request = model.load_lora(lora_filepath)
    except Exception as e:
        logger.error(f'Error during training: {e}', exc_info=True)
        logger.warning('Skipping learning step due to error')
        return previous_lora_request
    return lora_request


def save_results(results, output_dir, log_to_wandb):
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f'Saving results to {output_dir}')
    metrics = aggregate_metrics(results)
    metrics.to_csv(f'{output_dir}/metrics.csv', index_label='task_id')
    print(metrics.tail(1))
    if log_to_wandb:
        wandb.log({"task_metrics": wandb.Table(dataframe=metrics.tail(1))})
        metrics_summary = metrics.loc['MEAN'].to_dict()
        wandb.log(metrics_summary)
    # convert numpy arrays to lists for json serialization
    for task_id, task_results in results.items():
        for result in task_results:
            for key in ['input_grids', 'output_grids', 'test_output_grids']:
                if key in result:
                    result[key] = [grid.tolist() for grid in result[key]]
    write_json(results, f'{output_dir}/results.json.gz')


def log_metrics_evolution(results, step=8):
    for n_predictions in range(step, max(len(v) for v in results.values()) + 1, step):
        partial_results = {task_id: task_results[:n_predictions] for task_id, task_results in results.items()}
        partial_metrics = aggregate_metrics(partial_results)
        partial_metrics_summary = partial_metrics.loc['MEAN'].to_dict()
        partial_metrics_summary = {f"evolution/{k}": v for k, v in partial_metrics_summary.items()}
        wandb.log(partial_metrics_summary, step=n_predictions)

# hindsight relabeling functions, probably should be moved to module
def create_hindsight_relabeled_tasks(results, task):
    # TODO: strategies to avoid repetitions
    # sort the tasks, placing the best ones last
    sorted_results = sorted(results, key=lambda r: (r.get('train_correct_grids', -1), r.get('train_pixel_score', -1)), reverse=False)
    relabeled_tasks = []
    n_train = len(task['train'])
    for result in sorted_results:
        if 'output_grids' not in result:
            continue
        new_task = {
            'train': [{'input': input, 'output': output} for input, output in zip(result['input_grids'][:n_train], result['output_grids'][:n_train])],
            'test': [{'input': input, 'output': output} for input, output in zip(result['input_grids'][n_train:], result['output_grids'][n_train:])],
            'text_prediction': result['text_prediction'],
        }
        relabeled_tasks.append(new_task)
    return relabeled_tasks


def create_training_prompts(relabeled_tasks, grid_encoder, tokenizer):
    prompts = []
    for task in relabeled_tasks:
        prompt = create_prompt_from_task(
            task, grid_encoder=grid_encoder, tokenizer=tokenizer, shuffle_train_samples=True)
        prompt += task['text_prediction'] + tokenizer.eos_token
        prompts.append(prompt)
    return prompts


if __name__ == '__main__':
    main()
