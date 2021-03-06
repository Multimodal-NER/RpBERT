# RpBERT
RpBERT: A Text-image Relationship Propagation Based BERT Model  for Multimodal NER

## Environment
### Python packages
>- python==3.7
>- torch==1.2.0
>- gensim==3.8.0 
>- numpy==1.18.3
>- torchcrf==1.0.4
>- pytorch-pretrained-bert==0.6.2
**(pip install -r requirements.txt)**

### Configuration
All configuration are listed in /cfgs/. Please verify parameters before running the codes.

### Data

```
datasets/
├── fudan/
│   ├──ner_img/
│   ├── train
│   ├── dev
│   └── test
├── snap/
│   ├──ner_img/
│   ├── train
│   ├── dev
│   └── test
└── bloomberg/
    ├──rel_img/
    └──textimage-data-image
```
####The MNER datasets format is as follows:

```
IMGID:94770
RT	O
@ShervinSinatra	O
:	O
Amsterdam	B-ORG
Savage	I-ORG
AF	I-ORG
.	O
http://t.co/btkv2ddE58	O

IMGID:1330809
RT	O
@noisywoman	O
:	O
@davrosz	O
@SamAntixMusic	O
This	O
Peter	B-PER
Dutton	I-PER
?	O
http://t.co/yjPOz2aLEW	O
```

### Pre-trained Models
```
pretrained/
├── embeddings/
│   ├──en-fasttext-crawl-300d-1M
│   ├──common_characters_large
├── bert-base-uncased/
│   ├── vocab.txt
│   ├── bert_config.json
│   └── pytorch_model.bin
├── bert-large-uncased/
│   ├── vocab.txt
│   ├── bert_config.json
│   └── pytorch_model.bin
└── resnet/
    └──resnet152-b121ed2d.pth
```
## Usage
### Training
>- python main.py 

### Testing
>- python test.py