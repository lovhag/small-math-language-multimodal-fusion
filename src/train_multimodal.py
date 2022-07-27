from functools import partial
import pytorch_lightning as pl
import itertools
import torch
from torch.utils.data import DataLoader
from dataset import ArithmeticDatasetWithHiddenReps, pad_collate_with_hidden_reps
from tokenizer import ArithmeticLanguageTokenizer, ArithmeticTokenizer, get_power_ten_combinations_splits, get_language_addition_string
from train import AutoRegressiveTransformer
from gpt2_single_stream import GPT2ConfigLikeVisualBERT, GPT2LMHeadModelLikeVisualBERT, AutoRegressiveTransformerLikeVisualBERT
from gpt2_dual_stream import GPT2ConfigLikeLXMERT, GPT2LMHeadModelLikeLXMERT, AutoRegressiveTransformerLikeLXMERT
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from num2words import num2words
from embeddings import EMBEDDING_TYPES

STREAM_MODELS = {"single": AutoRegressiveTransformerLikeVisualBERT, "dual": AutoRegressiveTransformerLikeLXMERT}

def get_math_model_representation(*n, math_model, math_tokenizer):
    number_part = f"{'+'.join(str(n_i) for n_i in n)}="

    outputs = math_model.model.transformer(torch.tensor(math_tokenizer.encode(number_part, add_eos_token=False))[None, :])
    return outputs.last_hidden_state.squeeze().detach().cpu()

def main(config):

    magnitude = config["magnitude"]

    math_tokenizer = ArithmeticTokenizer()
    language_tokenizer = ArithmeticLanguageTokenizer(magnitude, include_numbers=True)

    # Get only some number combinations for language dataset
    language_train_numbers, language_test_numbers = get_power_ten_combinations_splits(
        magnitude, 
        train_ratio=config["language_train_set_ratio"]
    )

    assert config["math_model_checkpoint_file"] is not None, "A math model checkpoint file must be provided if math model representations are to be used"
    
    math_model = AutoRegressiveTransformer.load_from_checkpoint(config["math_model_checkpoint_file"])
    math_model.eval()
    
    # Get all number combinations for numeric dataset
    numeric_numbers, _ = get_power_ten_combinations_splits(magnitude, train_ratio=1.0)

    # Get hidden math representations
    print("Generating math model representations...")
    math_representations = {val: get_math_model_representation(*val, math_model=math_model, math_tokenizer=math_tokenizer) for val in numeric_numbers}

    # Datasets for training
    train_ds = ArithmeticDatasetWithHiddenReps(list(language_train_numbers), language_tokenizer, get_language_addition_string, math_representations, data_type=1)
    collate_fn = partial(pad_collate_with_hidden_reps, pad_token_idx=language_tokenizer.pad_token_idx)
    train_dl = DataLoader(
        train_ds, 
        batch_size=config["batch_size"], 
        shuffle=True,
        collate_fn=collate_fn, 
        num_workers=4
    )

    # Datasets for validation
    language_train_ds_for_validation = ArithmeticDatasetWithHiddenReps(list(language_train_numbers), language_tokenizer, get_language_addition_string, math_representations, data_type=1)
    language_test_ds_for_validation = ArithmeticDatasetWithHiddenReps(list(language_test_numbers), language_tokenizer, get_language_addition_string, math_representations, data_type=1)
    language_train_dl_for_validation = DataLoader(
        language_train_ds_for_validation, 
        batch_size=config.get("val_batch_size", config["batch_size"]), 
        collate_fn=collate_fn, 
        num_workers=4
    )
    language_test_dl_for_validation = DataLoader(
        language_test_ds_for_validation, 
        batch_size=config.get("val_batch_size", config["batch_size"]), 
        collate_fn=collate_fn, 
        num_workers=4
    )

    assert config["stream_type"] in STREAM_MODELS

    if config["stream_type"] == "single":
        assert config["visual_embedding_type"] in EMBEDDING_TYPES, "The desired embeddings must be implemented."

        model_hparams = {
            "learning_rate": config["learning_rate"],
            "pad_token_idx": language_tokenizer.pad_token_idx,
            "generate_after_token": language_tokenizer.token2idx["to"],
            "vocab_size": len(language_tokenizer.vocab),
            "n_positions": 64,
            "n_embd": config["n_embd"],
            "n_layer": config["n_layer"],
            "n_inner": config["n_inner"],
            "n_head": config["n_head"],
            "visual_embedding_dim": math_model.model.transformer.embed_dim, # this is the same as the hidden size of the transformer
            "visual_embedding_type": config["visual_embedding_type"],
        }
        model = STREAM_MODELS[config["stream_type"]](model_hparams)
    else:
        model_hparams = {
            "learning_rate": config["learning_rate"],
            "pad_token_idx": language_tokenizer.pad_token_idx,
            "generate_after_token": language_tokenizer.token2idx["to"],
            "vocab_size": len(language_tokenizer.vocab),
            "n_positions": 64,
            "n_embd": config["n_embd"],
            "n_layer": config["n_layer"],
            "n_inner": config["n_inner"],
            "n_head": config["n_head"],
            "visual_embedding_dim": math_model.model.transformer.embed_dim, # this is the same as the hidden size of the transformer
            "l_layers": config["l_layers"],
            "x_layers": config["x_layers"],
            "r_layers": config["r_layers"]
        }
        model = STREAM_MODELS[config["stream_type"]](model_hparams)

    callbacks = [EarlyStopping(monitor="val_comp_accuracy", mode="max", stopping_threshold=1.0-1e-6, patience=30)]

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
        max_epochs=2000,
        callbacks=callbacks,
        check_val_every_n_epoch=config["validate_every"],
        checkpoint_callback=enable_checkpointing, # TODO: Change to enable_checkpointing later when upgrading pl
        gpus=None
    )
    trainer.fit(
        model, 
        train_dl, 
        [language_train_dl_for_validation, language_test_dl_for_validation], #[numeric_dl_for_validation, numeric_dl_for_validation], 
    )



if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()

    # Data args
    parser.add_argument("--magnitude", type=int, required=True)
    parser.add_argument("--language-train-set-ratio", type=float, default=0.1)

    # Setup args
    parser.add_argument("--math-model-checkpoint-file", type=str, default=None)

    # Training args
    parser.add_argument("--experiment", default=None)
    parser.add_argument("--version", default=None, help="Experiment version (optional)")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--val-batch-size", type=int, default=1024)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--validate-every", default=1, type=int)
    
    # Model args
    parser.add_argument("--n-embd", type=int, default=64)
    parser.add_argument("--n-layer", type=int, default=4)
    parser.add_argument("--n-inner", type=int, default=256)
    parser.add_argument("--n-head", type=int, default=8)
    parser.add_argument("--visual-embedding-type", type=str, default="visualbert")
    parser.add_argument("--stream-type", type=str, default="single", help="Can be single- or dual-stream")
    parser.add_argument("--l-layers", type=int, default=2, help="For the dual-stream implementation")
    parser.add_argument("--x-layers", type=int, default=1, help="For the dual-stream implementation")
    parser.add_argument("--r-layers", type=int, default=1, help="For the dual-stream implementation")

    args = parser.parse_args()

    main(vars(args))
