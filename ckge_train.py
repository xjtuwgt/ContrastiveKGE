from traincodes.run_cke import train_run, infer_run
from traincodes.run_kge import train_kge_run
from kgeutils.ioutils import ArgParser
import argparse
import sys

if __name__ == '__main__':
    parser = ArgParser()
    args_config_provided = parser.parse_args(sys.argv[1:])
    args = parser.parse_args()

    run_type = args.run_type
    ### Step 1: Graph contrastive learning to learn both entity embedding and relation embedding
    ### Key parameter setting (need to tune) --> refer to configs/train.wn18rr.json
    if run_type == 'graph_contrastive_train':
        train_run()
    ### Step 2: Inference the entity embedding and relation embedding and save as numpy data
    ### Note: need the same config setting, however, for sub-graph sample (all neighorhood information is used)
    ### The saved entity embedding and relation embedding are 'entity.npy' and 'relation.npy'
    elif run_type == 'graph_contrastive_infer':
        infer_run()
    ### Step 3: initialize the KGE model by the inferred entity and relation embeddings for KGE training
    ### The parameter in 'init_checkpoint' stores the folder name of 'entity.npy' and 'relation.npy'
    elif run_type == 'KGE_train':
        train_kge_run()
    else:
        raise 'Run type {} is not supported'.format(run_type)