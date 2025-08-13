import logging, os, numpy as np, torch

def init_logging(level=logging.INFO):
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    return logging.getLogger(__name__)

def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def build_wandb_config(trainer, config, tokenized_datasets=None, trial_number=None):
	"""
	Build a rich Weights & Biases configuration dictionary capturing
	model, training, data, and hardware metadata for the current run.

	Args:
		trainer: transformers.Trainer instance
		config: Config instance with training settings
		tokenized_datasets: optional DatasetDict to log sizes
		trial_number: optional Optuna trial number

	Returns:
		dict: configuration to pass to wandb.init(config=...)
	"""
	ta = getattr(trainer, 'args', None)
	model_cfg = getattr(getattr(trainer, 'model', None), 'config', None)

	data_info = None
	if tokenized_datasets is not None:
		data_info = {
			"seed": config.SEED,
			"val_size": config.VAL_SIZE,
			"train_len": len(tokenized_datasets["train"]) if "train" in tokenized_datasets else None,
			"val_len": len(tokenized_datasets["val"]) if "val" in tokenized_datasets else None,
			"test_len": len(tokenized_datasets["test"]) if "test" in tokenized_datasets else None,
		}

	training_info = None
	if ta is not None:
		training_info = {
			"learning_rate": config.LEARNING_RATE,
			"weight_decay": config.WEIGHT_DECAY,
			"epochs": config.NUM_EPOCHS,
			"lr_scheduler_type": config.LR_SCHEDULER_TYPE,
			"per_device_train_batch_size": ta.per_device_train_batch_size,
			"per_device_eval_batch_size": ta.per_device_eval_batch_size,
			"gradient_accumulation_steps": config.GRADIENT_ACCUMULATION_STEPS,
			"warmup_steps": config.WARMUP_STEPS,
			"early_stopping_patience": config.EARLY_STOPPING_PATIENCE,
			"eval_strategy": ta.eval_strategy,
			"save_strategy": ta.save_strategy,
			"metric_for_best_model": ta.metric_for_best_model,
			"greater_is_better": ta.greater_is_better,
		}

	model_info = {
		"name": config.MODEL_NAME,
		"num_labels": getattr(model_cfg, "num_labels", None),
		"hidden_dropout_prob": getattr(model_cfg, "hidden_dropout_prob", None),
		"attention_probs_dropout_prob": getattr(model_cfg, "attention_probs_dropout_prob", None),
		"classifier_dropout": getattr(model_cfg, "classifier_dropout", None),
		"max_length": getattr(config, "MAX_LENGTH", None),
		"hf_config": model_cfg.to_dict() if model_cfg is not None else None,
	}

	hardware_info = {
		"device": str(config.DEVICE),
		"num_gpus": int(config.NUM_GPUS),
		"effective_batch_size": int(config.BATCH_SIZE),
	}

	return {
		"trial_number": trial_number,
		"objective_metric": config.METRIC_FOR_BEST_MODEL,
		"model": model_info,
		"training": training_info,
		"data": data_info,
		"hardware": hardware_info,
	}
