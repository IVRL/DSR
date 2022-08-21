import os
import mlflow
import tempfile

from src import train
from .options import args
from .helpers import report_model
from .trainer import Trainer



if args.report_to_mlflow:
    mlflow.set_experiment(args.arch)
    name = ''
    if not args.validation_only:
        name += 'train:' + '+'.join(dataset.name for dataset in args.dataset_train) + '-'
    name += 'valid:' + '+'.join(dataset.name for dataset in args.dataset_val)
    run = mlflow.start_run(run_name=name)
    mlflow.log_params(vars(args))

trainer = Trainer()
if args.validation_only or args.images:
    if args.load_pretrained is None \
            and args.load_checkpoint is None \
            and not args.download_pretrained \
            and args.load_from_mlflow is None \
            and not args.arch in ('nearest', 'linear', 'bilinear', 'bicubic', 'trilinear'):
        raise ValueError("For validation, please use --load-pretrained CHECKPOINT or --download-pretrained")
    if args.images:
        trainer.run_model()
    else:
        trainer.validation()
else:
    report_model(trainer.model)
    if args.save_checkpoint:
        if args.meta:
            trainer.train_meta()
        else:
            trainer.train()
    else:
        with tempfile.TemporaryDirectory(dir=os.getcwd()) as temp_dir:
            
            args.save_checkpoint = os.path.join(temp_dir, 'model', 'model.pth')
            os.makedirs(os.path.dirname(args.save_checkpoint), exist_ok=True)
            if args.meta:
                trainer.train_meta()
            else:
                trainer.train()
            if args.report_to_mlflow:
                mlflow.log_artifact(os.path.dirname(args.save_checkpoint))

if args.report_to_mlflow:
    mlflow.end_run()
