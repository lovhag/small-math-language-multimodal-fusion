# EXPERIMENTS

All bash commands assume that you are standing in the `src` folder.

___

## Train a pure math model

Train the autoregressive math model on strings like `1+2=3`. We want it to generate good representations of math strings.

### Train a good math representations model

We train this model on 99% of the train data. It attains a validation accuracy close to 1.0 after training.

```bash
python train.py math --experiment trained_math_model --batch-size 32 --magnitude 2 --validate-every 1 --train-set-ratio 0.99 --version 1
```

Weights for a previously trained such model can be found under `data/model_weights/trained_math_model`.

### Train a less good math representations model

We train this model on 5% of the train data. It attains a validation accuracy of 0.67 after training.

```bash
python train.py math --experiment less_trained_math_model --batch-size 32 --magnitude 2 --validate-every 5 --train-set-ratio 0.05 --version 1
```

Weights for a previously trained such model can be found under `data/model_weights/less_trained_math_model`.
___

## Train a pure language model (final_exp=2)

Denoted `GPT2text` in the article.

Train the autoregressive language model on strings like `one plus two is equal to three`. We can use this as a baseline.

```bash
python train.py language --experiment gpt2text --batch-size 32 --magnitude 2 --validate-every 10 --train-set-ratio 0.01 --version 0
python train.py language --experiment gpt2text --batch-size 32 --magnitude 2 --validate-every 5 --train-set-ratio 0.05 --version 1
python train.py language --experiment gpt2text --batch-size 32 --magnitude 2 --validate-every 5 --train-set-ratio 0.1 --version 2
python train.py language --experiment gpt2text --batch-size 32 --magnitude 2 --validate-every 5 --train-set-ratio 0.2 --version 3
```

Results for a previously trained such model can be found under `data/final_exp=2`. The folder names are in the format of `v_i_j` where `i` denotes the experiment version number corresponding to `--version` from the run command and `j` denotes the run number of a total of 5 repeated training runs with the exact same settings (except for different random seeds for training).

___

## Experiments on math-and-language models

___


### VisualBERT embeddings and Single-stream (final_exp=10)

Denoted `VisualBERT` in the article.

Model config:
* Model has 4 transformer layers in total. 
* Number of model parameters: 218K

```bash
python train_multimodal.py --experiment visualbert --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.01 --version 0
python train_multimodal.py --experiment visualbert --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.05 --version 1
python train_multimodal.py --experiment visualbert --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.1 --version 2
python train_multimodal.py --experiment visualbert --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.2 --version 3
```

Results for a previously trained such model can be found under `data/final_exp=10`.

___

### UNITER embeddings and Single-stream (final_exp=11)

Denoted `UNITER` in the article.

Same as the previous experiment except for that we use a more UNITER like embedding (from Volta).

Model config:
* Model has 4 transformer layers in total. 
* Number of model parameters: 211K

```bash
python train_multimodal.py --experiment uniter --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.01 --visual-embedding-type uniter --version 0
python train_multimodal.py --experiment uniter --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.05 --visual-embedding-type uniter --version 1
python train_multimodal.py --experiment uniter --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.1 --visual-embedding-type uniter --version 2
python train_multimodal.py --experiment uniter --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.2 --visual-embedding-type uniter --version 3
```

Results for a previously trained such model can be found under `data/final_exp=11`.

___

### LXMERT embeddings and Dual-stream (final_exp=15)

Denoted `LXMERTd` in the article.

Run a dual-stream GPT2 model, inspired from LXMERT. LXMERT version given by `gpt2_dual_stream_with_visn.py`.

Current model config: 
* 3 language layers, 2 relational layers and 2 cross-layers (7 layers in total)
* 495K parameters in total (VisualBERT: 211K)
* Compare with LXMERT (228M) and VisualBERT (112M)
    * LXMERT is approx 2.04 times larger
    * Current sandbox case: LXMERT 2.3 times larger

```bash
python train_multimodal.py --experiment lxmertd --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.01 --stream-type dual --l-layers 3 --r-layers 2 --x-layers 2 --version 0
python train_multimodal.py --experiment lxmertd --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.05 --stream-type dual --l-layers 3 --r-layers 2 --x-layers 2 --version 1
python train_multimodal.py --experiment lxmertd --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.1 --stream-type dual --l-layers 3 --r-layers 2 --x-layers 2 --version 2
python train_multimodal.py --experiment lxmertd --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.2 --stream-type dual --l-layers 3 --r-layers 2 --x-layers 2 --version 3
```

Results for a previously trained such model can be found under `data/final_exp=15`.

___

### LXMERT embeddings and Single-stream (final_exp=16)

Denoted `LXMERTs` in the article.

Same as the VisualBERT and UNITER embedding experiments except for that we use a more LXMERT like embedding (from LXMERT).

Model config:
* Model has 4 transformer layers in total. 
* Number of model parameters: 211K

```bash
python train_multimodal.py --experiment lxmerts --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.01 --visual-embedding-type lxmert --version 0
python train_multimodal.py --experiment lxmerts --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.05 --visual-embedding-type lxmert --version 1
python train_multimodal.py --experiment lxmerts --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.1 --visual-embedding-type lxmert --version 2
python train_multimodal.py --experiment lxmerts --math-model-checkpoint-file "../data/model_weights/trained_math_model/epoch=13-step=4339.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.2 --visual-embedding-type lxmert --version 3
```

Results for a previously trained such model can be found under `data/final_exp=16`.

___

### LXMERT embeddings and Single stream and Less trained math representations (final_exp=18)

Denoted `LXMERTb` in the article.

Same as the previous experiment except for that we use bad multimodal representations from a less trained math model. The math model has only been trained on 5% of the math data (500 samples) and reached a validation accuracy of 0.67, to be compared with 1.0 for the better trained math model.

Model config:
* Model has 4 transformer layers in total. 
* Number of model parameters: 211K

```bash
python train_multimodal.py --experiment lxmertb --math-model-checkpoint-file "../data/model_weights/less_trained_math_model/epoch=239-step=3839.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.01 --visual-embedding-type lxmert --version 0
python train_multimodal.py --experiment lxmertb --math-model-checkpoint-file "../data/model_weights/less_trained_math_model/epoch=239-step=3839.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.05 --visual-embedding-type lxmert --version 1
python train_multimodal.py --experiment lxmertb --math-model-checkpoint-file "../data/model_weights/less_trained_math_model/epoch=239-step=3839.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.1 --visual-embedding-type lxmert --version 2
python train_multimodal.py --experiment lxmertb --math-model-checkpoint-file "../data/model_weights/less_trained_math_model/epoch=239-step=3839.ckpt" --batch-size 1024 --magnitude 2 --validate-every 5 --language-train-set-ratio 0.2 --visual-embedding-type lxmert --version 3
```

Results for a previously trained such model can be found under `data/final_exp=18`.