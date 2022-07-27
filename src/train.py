from functools import partial
import torch
import pytorch_lightning as pl
import torch.nn.functional as F
from tokenizer import get_tokenizer, get_power_ten_combinations_splits, get_numeric_addition_string, get_language_addition_string
from dataset import ArithmeticDataset, pad_collate
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch.utils.data import DataLoader
from torchmetrics import Accuracy
from transformers import GPT2LMHeadModel, GPT2Config


def roll_columns(mat, column_shifts):
    n_rows, n_cols = mat.shape
    arange1 = torch.arange(n_cols, device=mat.device).view((1, n_cols)).repeat((n_rows, 1))
    arange2 = (arange1 - column_shifts[:, None]) % n_cols
    return torch.gather(mat, 1, arange2)


def load_model_for_transfer_learning(checkpoint_path: str, model_hparams, freeze_layers: bool):
    # Load checkpoint
    state_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))["state_dict"]

    # Rename parameters
    for key in list(state_dict.keys()):
        state_dict[key.strip("model.")] = state_dict.pop(key)

    # Remove embedding layer weights from pre-trained checkpoint
    del state_dict["transformer.wte.weight"]
    del state_dict["transformer.wpe.weight"]

    # Instantiate GPT2LMHeadModel from checkpointed weights
    config = GPT2Config(
        vocab_size=model_hparams["vocab_size"],
        n_positions=model_hparams["n_positions"],
        n_embd=model_hparams["n_embd"],
        n_layer=model_hparams["n_layer"],
        n_inner=model_hparams["n_inner"],
        n_head=model_hparams["n_head"]
    )
    gpt_model = GPT2LMHeadModel.from_pretrained(None, config=config, state_dict=state_dict)

    # Optionally freeze checkpointed weights
    for name, param in gpt_model.named_parameters():
        if name in state_dict:
            param.requires_grad = not freeze_layers

    # Instantiate model
    model = AutoRegressiveTransformer(model_hparams, gpt_model=gpt_model)

    return model


class AutoRegressiveTransformer(pl.LightningModule):

    def __init__(self, hparams, gpt_model=None):
        super().__init__()

        self.lr = hparams["learning_rate"]
        self.pad_token_idx = hparams["pad_token_idx"]
        self.generate_after_token = hparams["generate_after_token"]
        self.mask_data_type_specific_tokens = hparams.get("mask_data_type_specific_tokens", False)
        self.bad_words_ids_per_data_type = hparams.get("bad_words_ids_per_data_type", []) # List[List[int]] - [data_type, token_id]

        if gpt_model is not None:
            self.model = gpt_model
        else:
            self.model = GPT2LMHeadModel(
                GPT2Config(
                    vocab_size=hparams["vocab_size"],
                    n_positions=hparams["n_positions"],
                    n_embd=hparams["n_embd"],
                    n_layer=hparams["n_layer"],
                    n_inner=hparams["n_inner"],
                    n_head=hparams["n_head"]
                )
            )

        self.train_comp_accuracy = Accuracy()
        self.val_comp_accuracy = Accuracy()

        self.save_hyperparameters()

    def forward(self, batch):
        return self.model(**batch)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
        
    def training_step(self, batch, batch_idx):
        data_types = batch.pop("data_types")

        output = self(batch)
        loss = output.loss
        
        self.log("loss", loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        generate_after_token_idxs = (batch["input_ids"] == self.generate_after_token).long().argmax(dim=1)

        # Align on generate_after_token for batch generation
        input_ids = F.pad(
            batch["input_ids"], 
            (0, generate_after_token_idxs.max() - generate_after_token_idxs.min(), 0, 0), 
            value=self.pad_token_idx
        )
        input_ids = roll_columns(input_ids, generate_after_token_idxs.max() - generate_after_token_idxs)
        attention_mask =  (input_ids != self.pad_token_idx).int().to(input_ids.device)

        if self.mask_data_type_specific_tokens:
            assert torch.all(batch["data_types"][0] == batch["data_types"]), "During validation all examples in batch must be of same data_type"
            bad_words_ids = [ [token_id] for token_id in self.bad_words_ids_per_data_type[batch["data_types"][0]] ]
        else:
            bad_words_ids = None
        gen_out = self.model.generate(
            input_ids[:, :generate_after_token_idxs.max()+1], 
            attention_mask=attention_mask[:, :generate_after_token_idxs.max()+1],
            max_length=input_ids.shape[1], 
            pad_token_id=self.pad_token_idx,
            bad_words_ids=bad_words_ids
        )

        is_prediction_correct = ((gen_out == input_ids) | (input_ids == self.pad_token_idx)).all(dim=1)
        metric = self.train_comp_accuracy if dataloader_idx == 0 else self.val_comp_accuracy
        metric(is_prediction_correct, torch.ones_like(is_prediction_correct, dtype=torch.bool))

    def validation_epoch_end(self, outputs):
        self.log("train_comp_accuracy", self.train_comp_accuracy, prog_bar=True)
        self.log("val_comp_accuracy", self.val_comp_accuracy, prog_bar=True)


def main(config, callbacks=[]):

    # Data init
    tokenizer = get_tokenizer(type=config["type"], magnitude=config["magnitude"])
    train_numbers, test_numbers = get_power_ten_combinations_splits(config["magnitude"], train_ratio=config["train_set_ratio"])
    to_string_fn = get_numeric_addition_string if config["type"] == "math" else get_language_addition_string
    train_ds = ArithmeticDataset(list(train_numbers), tokenizer, to_string_fn)
    test_ds = ArithmeticDataset(list(test_numbers), tokenizer, to_string_fn)

    collate_fn = partial(pad_collate, pad_token_idx=tokenizer.pad_token_idx)
    train_dl = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, collate_fn=collate_fn, num_workers=4)
    test_dl = DataLoader(test_ds, batch_size=config.get("val_batch_size", config["batch_size"]), collate_fn=collate_fn, num_workers=4)

    model_hparams = {
        "learning_rate": config["learning_rate"],
        "pad_token_idx": tokenizer.pad_token_idx,
        "generate_after_token": tokenizer.token2idx["="] if config["type"] == "math" else tokenizer.token2idx["to"],
        "vocab_size": len(tokenizer.vocab),
        "n_positions": 32,
        "n_embd": config["n_embd"],
        "n_layer": config["n_layer"],
        "n_inner": config["n_inner"],
        "n_head": config["n_head"]
    }

    # Optionally load checkpoint for transfer learning
    if config.get("load_checkpoint", None):
        model = load_model_for_transfer_learning(config["load_checkpoint"], model_hparams, config["freeze_pretrained_layers"])
    else:
        model = AutoRegressiveTransformer(model_hparams)

    callbacks += [EarlyStopping(monitor="val_comp_accuracy", mode="max", stopping_threshold=1.0-1e-6, patience=50)]

    if config["experiment"]:
        logger = TensorBoardLogger(save_dir="../data", name=config["experiment"], version=f"v_{config['version']}")
        callbacks.append(ModelCheckpoint(monitor="val_comp_accuracy", mode="max", save_weights_only=True))
        enable_checkpointing = True
    else:
        logger = False
        enable_checkpointing = False

    trainer = pl.Trainer(
        default_root_dir="../data", 
        logger=logger, 
        max_steps=10000,
        callbacks=callbacks,
        check_val_every_n_epoch=config["validate_every"],
        checkpoint_callback=enable_checkpointing, # TODO: Change to enable_checkpointing later when upgrading pl
        gpus=None
    )
    trainer.fit(
        model, 
        train_dl, 
        [train_dl, test_dl]
    )


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    
    # Data args
    parser.add_argument("type", choices=["math", "language"])
    parser.add_argument("--magnitude", type=int, required=True)
    parser.add_argument("--train-set-ratio", type=float, default=0.8)

    # Training args
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--version", default=None, help="Experiment version (optional)")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--val-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=0.0005)
    parser.add_argument("--validate-every", default=1, type=int)
    parser.add_argument("--freeze-pretrained-layers", action="store_true")
    
    # Model args
    parser.add_argument("--load-checkpoint", default=None)
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-inner", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=8)

    args = parser.parse_args()

    main(vars(args))