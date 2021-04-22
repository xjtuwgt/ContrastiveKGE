pytorch-metric-learning 0.9.90
pytorch 1.6.0+
dgl 0.6.0

a) For wordnet:
### Step 1: Graph contrastive learning to learn both entity embedding and relation embedding
---- Key parameter setting (need to tune) --> refer to configs/train.wn18rr.json
python3 ckge_train.py --config_file configs/train.wn18rr.json --run_type graph_contrastive_train
### Step 2: Inference the entity embedding and relation embedding and save as numpy data
---- Note: need the same config setting, however, for sub-graph sample (all neighorhood information is used)
---- The saved entity embedding and relation embedding are 'entity.npy' and 'relation.npy'
python3 ckge_train.py --config_file configs/train.wn18rr.json --run_type graph_contrastive_train
### Step 3: initialize the KGE model by the inferred entity and relation embeddings for KGE training
---- The parameter in 'init_checkpoint' stores the folder name of 'entity.npy' and 'relation.npy'
python3 ckge_train.py --config_file configs/train.wn18rr.json --run_type KGE_train

b) For FreedBase:
python3 ckge_train.py --config_file configs/train.fb15k237.json --run_type graph_contrastive_train
python3 ckge_train.py --config_file configs/train.fb15k237.json --run_type graph_contrastive_train
python3 ckge_train.py --config_file configs/train.fb15k237.json --run_type KGE_train