from multiprocessing import cpu_count

SEED = 42
# RESULT_FILE = 'result.tsv'
# RESULT_IMAGE = 'result.jpg'
SUBMISSION_FILE = 'predictions.txt'
MODEL_TYPE = 'xlmroberta'
MODEL_NAME = 'xlm-roberta-large'

siamese_config = {
    'output_dir': '.temp/outputs/model/',
    'best_model_dir': '.temp/outputs/best_model/',
    'cache_dir': '.cache/',

    'fp16': False,
    'fp16_opt_level': 'O1',
    'max_seq_length': 128,
    'train_batch_size': 8,
    'gradient_accumulation_steps': 1,
    'eval_batch_size': 8,
    'num_train_epochs': 10,
    'weight_decay': 0,
    'learning_rate': 2e-5,
    'adam_epsilon': 1e-8,
    'warmup_ratio': 0.1,
    'warmup_steps': 0,
    'max_grad_norm': 1.0,
    'do_lower_case': False,

    'logging_steps': 900,
    'save_steps': 900,
    'no_cache': False,
    'no_save': False,
    'save_recent_only': True,
    'save_model_every_epoch': False,
    'n_fold': 1,
    'evaluate_during_training': True,
    'evaluate_during_training_silent': True,
    'evaluate_during_training_steps': 900,
    'evaluate_during_training_verbose': True,
    'use_cached_eval_features': False,
    'save_best_model': True,
    'save_eval_checkpoints': True,
    'tensorboard_dir': None,
    'save_optimizer_and_scheduler': True,

    'regression': True,

    'overwrite_output_dir': False,
    'reprocess_input_data': True,

    'process_count': cpu_count() - 2 if cpu_count() > 2 else 1,
    'n_gpu': 1,
    'use_multiprocessing': True,
    'silent': False,

    'wandb_project': None,
    'wandb_kwargs': {},

    'use_early_stopping': True,
    'early_stopping_patience': 10,
    'early_stopping_delta': 0,
    'early_stopping_metric': 'eval_loss',
    'early_stopping_metric_minimize': True,
    'early_stopping_consider_epochs': False,

    'manual_seed': SEED,

    'encoding': None,
}
