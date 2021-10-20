# Triplet Attention: Rethinking the similarity in Transformers

[![Hits](https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fzhouhaoyi%2FTripletAttention&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false)](https://hits.seeyoufarm.com) This is the pytorch implementation of Triplet Attention in the KDD'21 paper: [Triplet Attention: Rethinking the similarity in Transformers](https://dl.acm.org/doi/abs/10.1145/3447548.3467241).


## Requirements
+ Python 3.6
+ numpy==1.17.3
+ scipy==1.1.0
+ pandas==0.25.1
+ torch==1.2.0
+ tqdm==4.36.1
+ matplotlib==3.1.1
+ tokenizers==0.10.3
+ ...

Dependencies can be installed using the following command:

```
pip install -r requirements.txt
```

## Usage
We implement BERT-A<sup>3</sup> and DistilBERT-A<sup>3</sup> in `huggingface transformers`, you can use BERT-A<sup>3</sup> or DistilBERT-A<sup>3</sup> model like BERT or DistilBERT model in `huggingface transformers`.

build BERT-A<sup>3</sup>

```
from transformers import BertTokenizer, BertModel, BertConfig

config = BertConfig.from_pretrained('bert-base-uncased')

config.group_size = 2 # number of triplet attention head
config.cross_type = 0 # cross product type (0: L cross product with permutation, 1: L*L cross product)
config.agg_type = 0 # aggregation type when using L*L cross product
config.absolute_flag = 0 # whether to use absolute value of triplet attention (1: use abs)
config.random_flag = 0 # permutation type (0: multi permutation, 1:only random permutation)
config.permute_type = '1,2,3,4,5' # permutaion type groups
config.permute_back = 0 # whether to do permutation inverse (0: do permutation inverse)
config.Tlayers = '0,1,2,8,9,10' # layers which use triplet attention heads
config.key2_flag = 0 # whether to use key2 linear layer to get key_triplet2 (0: use key2 layer)
config.head_choice = 12 # whether to choose triplet attention heads randomly (0: choose the last 3*group_size heads as triplet attention heads, 1: randomly choose 3*group_size heads as triplet attention heads)

bert_A3 = BertModel.from_pretrained('bert-base-uncased', config=config)
```

use BERT-A<sup>3</sup>

```
from transformers import BertTokenizer, BertModel, BertConfig

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

config = BertConfig.from_pretrained('bert-base-uncased')
bert_A3 = BertModel.from_pretrained('bert-base-uncased', config=config)

inputs = tokenizer('hello world',return_tensors="pt")
outputs = bert_A3(**inputs)
```

You can refer to `/transformers/models/bert/modeling_bert.py` to get more details.


## Train Commands
Commands for training and testing the model BERT-A<sup>3</sup> on GLUE task (rte):

```
python run_glue.py --model_name_or_path bert-base-uncased --task_name rte --do_train --do_eval --do_predict --max_seq_length 128 --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --cross_type 0 --agg_type 0 --tlayers '0,1,2,3' --learning_rate 3e-5 --num_train_epochs 6 --key2_flag 0 --random_flag 0 --absolute_flag 0 --permute_back 0 --permute_type '0,1,3,5,6' --head_choice 0 --group_size 1 --overwrite_output_dir --output_dir ./run/
```
