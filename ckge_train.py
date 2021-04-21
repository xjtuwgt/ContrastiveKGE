from traincodes.run_cke import train_run, infer_run
from traincodes.run_kge import train_kge_run
from kgeutils.ioutils import ArgParser
import sys

if __name__ == '__main__':
    parser = ArgParser()
    args_config_provided = parser.parse_args(sys.argv[1:])
    args = parser.parse_args()

    run_type = args.run_type
    if run_type == 'graph_contrastive_train':
        train_run()
    elif run_type == 'graph_contrastive_infer':
        infer_run()
    elif run_type == 'KGE_train':
        train_kge_run()
    else:
        raise 'Run type {} is not supported'.format(run_type)