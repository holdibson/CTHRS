# CTHRS
Unsupervised Contrastive Transformer Hashing for Multimodal Remote Sensing Retrieval


### Recommendation
We recommend the following operating environment:
- Python == 3.8.10
- Pytorch == 2.3.0+cu118
- Torchvision == 0.18.0+cu118
- And other packages

### Datasets
We release the three experimental datasets as follows:
- RSICD, [[Baidu Pan](https://pan.baidu.com/s/1lFlQ1AtyDiLKXKrQvDHhJg), password: smk3]
- RSITMD, [[Baidu Pan](https://pan.baidu.com/s/1jambmgVdiIeRk8RInzcItw), password: aqr3]
- UCM, [[Baidu Pan](https://pan.baidu.com/s/1LNgWvPfz8zsZGlT4EsjTiA), password: xmt7]

Please download this three datasets, and put them in the `./data/` folder.


### Demo 
Taking RSICD as an example, our model can be trained and verified by the following command:
```bash
./test-rsicd.sh
```
