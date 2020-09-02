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

### Pre-trained Models
```
pretrained_model/
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