from absl import app
from absl import flags

import pandas as pd

from pathlib import Path
import numpy as np

from tqdm import tqdm

from transformers import (
    BertForSequenceClassification,
    DistilBertForSequenceClassification,
)
from transformers import AutoTokenizer

from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch

import lightning as L
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import confusion_matrix, accuracy_score

import time

import wandb

wandb.login()

flags.DEFINE_string("model_size", "mini", "size of BERT model to load")
flags.DEFINE_string("device", "cpu", "cpu | cuda | mps")
flags.DEFINE_integer("batch_size", 16, "batch size for model training")
flags.DEFINE_float("learning_rate", 1e-5, "learning rate for model training")
flags.DEFINE_integer("max_epochs", 10, "maximum training epochs")
flags.DEFINE_string("project_name", "my_project", "wandb project name")
flags.DEFINE_string("output_dir", "output", "output data location")
flags.DEFINE_string("text_field", "my_text", "input text field")
flags.DEFINE_string("target_field", "my_target", "target field (categorical)")
flags.DEFINE_integer("num_targets", 1, "number of categories in target")
flags.DEFINE_string(
    "train_data_path", "data/train_data.csv", "input train data location"
)
flags.DEFINE_string(
    "valid_data_path", "data/valid_data.csv", "input valid data location"
)
flags.DEFINE_string("test_data_path", "data/test_data.csv", "input test data location")
flags.DEFINE_float("dropout", 0.0, "classifier dropout")
flags.DEFINE_float("weight_decay", 0.005, "weight decay on optimizer")


MODEL_DICT = {
    "tiny": "google/bert_uncased_L-2_H-128_A-2",
    "mini": "google/bert_uncased_L-4_H-256_A-4",
    "small": "google/bert_uncased_L-4_H-512_A-8",
    "medium": "google/bert_uncased_L-8_H-512_A-8",
    "distilbert": "distilbert/distilbert-base-uncased",
}
MODEL_MAX_CONTEXT_LENGTH = 512
FLAGS = flags.FLAGS

rng = np.random.default_rng(512)


class TextDataForBert(Dataset):
    def __init__(self, df, targets, tokenizer, max_sequence_length):
        self.df = df
        self.targets = targets
        self.tokenizer = tokenizer
        self.max_sequence_length = max_sequence_length

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        t = self.tokenizer(
            self.df[idx],
            max_length=self.max_sequence_length,
            padding="max_length",
            return_tensors="pt",
            truncation=True,
        )
        return {
            "target": torch.tensor(self.targets[idx], dtype=torch.int64),
            "input_ids": t["input_ids"],
            "attention_mask": t["attention_mask"],
        }


class LitBertModel(L.LightningModule):
    def __init__(self, model, learning_rate, weight_decay, loss_fn, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters()

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.loss_fn = loss_fn

    def training_step(self, batch, batch_idx):
        preds = self.model(
            batch["input_ids"].squeeze(1).to(self.device),
            attention_mask=batch["attention_mask"].squeeze(1).to(self.device),
        ).logits
        labels = batch["target"].to(self.device)

        # compute loss
        loss = self.loss_fn(preds, labels)

        # Logging to WandB or TensorBoard
        self.log("train_loss", loss)
        return loss

    def on_validation_epoch_end(self):
        sch = self.lr_schedulers()

        if isinstance(sch, torch.optim.lr_scheduler.ReduceLROnPlateau):
            sch.step(self.trainer.callback_metrics["valid_loss"])

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            preds = self.model(
                batch["input_ids"].squeeze(1).to(self.device),
                attention_mask=batch["attention_mask"].squeeze(1).to(self.device),
            ).logits
            labels = batch["target"].to(self.device)
            loss = self.loss_fn(preds, labels)

            p2 = [np.argmax(pr.detach().cpu().numpy()) for pr in preds]
            acc = accuracy_score(labels.detach().cpu().numpy(), p2)

        self.log("valid_loss", loss)
        self.log("valid_acc", acc)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min")
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
            "monitor": "valid_loss",
        }

    def forward(self, x, attention_mask, *args, **kwargs):
        return self.model(x, attention_mask=attention_mask, **kwargs)


def get_loader(data_path, tokenizer, configs):
    bs = configs["batch_size"]
    max_length = configs["max_sequence_length"]
    text_field = configs["text_field"]
    target_field = configs["target"]

    data = pd.read_csv(data_path)

    loader = DataLoader(
        TextDataForBert(
            data[text_field].values,
            data[target_field].values,
            tokenizer=tokenizer,
            max_sequence_length=max_length,
        ),
        batch_size=bs,
        shuffle=True,
    )

    return loader


def get_logger(model_size, num_targets, project_name, output_dir, configs):
    ts = time.time()
    run_name = f"bert-{model_size}-top{num_targets}-{ts:.0f}"
    wandb_logger = WandbLogger(
        log_model="all", project=project_name, name=run_name, save_dir=output_dir
    )
    wandb_logger.experiment.config.update(
        {
            "num_target": num_targets,
            "batch_size": configs["batch_size"],
            "model_name": configs["model_name"],
            "text_field": configs["text_field"],
            "target": configs["target"],
            "dropout": configs["dropout"],
        }
    )

    return wandb_logger, run_name


def train_model(
    model,
    wandb_logger,
    configs,
    train_loader,
    valid_loader,
    device,
    weight=None,
    weight_decay=0.005,
):
    loss_fn = nn.CrossEntropyLoss(weight=weight)

    lit_rnn_model = LitBertModel(
        model=model,
        loss_fn=loss_fn,
        learning_rate=configs["lr"],
        weight_decay=weight_decay,
    ).to(device)

    checkpoint_callback = ModelCheckpoint(monitor="valid_acc", save_top_k=2, mode="max")
    early_stop_callback = EarlyStopping(
        monitor="valid_loss", min_delta=0.00, patience=5, verbose=False, mode="min"
    )

    trainer = L.Trainer(
        limit_train_batches=1000,
        max_epochs=configs["max_epochs"],
        callbacks=[early_stop_callback, checkpoint_callback],
        enable_progress_bar=True,
        logger=wandb_logger,
    )
    trainer.fit(lit_rnn_model, train_loader, valid_loader)

    return lit_rnn_model


def evaluate_model(model, test_loader, device):
    ## Multiclass
    ylabel = []
    ypred = []
    for batch in tqdm(test_loader):
        with torch.no_grad():
            preds = model(
                batch["input_ids"].squeeze(1).to(device),
                attention_mask=batch["attention_mask"].squeeze(1).to(device),
            ).logits
            ypred += [np.argmax(pr.detach().cpu().numpy()) for pr in preds]
            ylabel += list(batch["target"].detach().cpu().numpy())

    return ylabel, ypred


def main(_):
    model_name = MODEL_DICT[FLAGS.model_size]

    num_targets = FLAGS.num_targets

    configs = {
        "batch_size": FLAGS.batch_size,
        "lr": FLAGS.learning_rate,
        "max_sequence_length": MODEL_MAX_CONTEXT_LENGTH,
        "max_epochs": FLAGS.max_epochs,
        "text_field": FLAGS.text_field,
        "model_name": model_name,
        "target": FLAGS.target_field,
        "dropout": FLAGS.dropout,
        "num_targets": num_targets,
    }

    if FLAGS.model_size.lower() == "distilbert":
        model = DistilBertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_targets, seq_classif_dropout=FLAGS.dropout or 0.2
        ).to(FLAGS.device)
    else:
        model = BertForSequenceClassification.from_pretrained(
            model_name, num_labels=num_targets, classifier_dropout=FLAGS.dropout or None
        ).to(FLAGS.device)

    tokenizer = AutoTokenizer.from_pretrained(model_name)

    logger, run_name = get_logger(
        FLAGS.model_size, num_targets, FLAGS.project_name, FLAGS.output_dir, configs
    )

    print("training: ", model_name, max_length, FLAGS.project_name, run_name)
    train_loader = get_loader(
        FLAGS.train_data_path, tokenizer, configs, downsample=False
    )
    valid_loader = get_loader(
        FLAGS.valid_data_path, tokenizer, configs, target_mask=True
    )
    test_loader = get_loader(FLAGS.test_data_path, tokenizer, configs, target_mask=True)
    train_model(
        model,
        logger,
        configs,
        train_loader,
        valid_loader,
        FLAGS.device,
        weight=None,
        weight_decay=FLAGS.weight_decay,
    )

    print("downloading model")
    checkpoint_reference = f"{FLAGS.project_name}/model-{logger.version}:best"
    artifact_path = logger.download_artifact(
        checkpoint_reference, artifact_type="model"
    )

    print("loading model")
    best_model = LitBertModel.load_from_checkpoint(Path(artifact_path) / "model.ckpt")

    # disable randomness, dropout, etc...
    best_model.to(FLAGS.device)
    best_model.eval()

    ylabel, ypred = evaluate_model(best_model, test_loader, FLAGS.device)

    print(f"Accuracy score: {accuracy_score(ylabel, ypred):.2f}")
    print(confusion_matrix(ylabel, ypred))


if __name__ == "__main__":
    app.run(main)
