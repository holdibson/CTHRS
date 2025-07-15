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
- RSICD, [[Baidu Pan](https://pan.baidu.com/s/1Hm-BFv0epUpJhHJMPkyKsA), password: g8wv]
- RSITMD, [[Baidu Pan](https://pan.baidu.com/s/1QnjYIp-TD5ucmWgrvJFQwg), password: 0lzq]
- UCM, [[Baidu Pan](https://pan.baidu.com/s/1fc2h_ow9RajV1oUdo6c1sw), password: s49e]

Please download this three datasets, and put them in the `./data/` folder.


### Demo 
Taking RSICD as an example, our model can be trained and verified by the following command:
```bash
./test-rsicd.sh
```
