### BERT Training

This script uses PyTorch Lightning to fine tune a pretrained BERT model. Suggested usage is to update `train_bert.bash` and run with 

``` bash
bash train_bert.bash
```

Note: this script expects you to have WandB configured. If you do not want to use WandB, you can edit out the wandB parts and use another logger (e.g. tensorboard).
