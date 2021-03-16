import os
from numpy import random
import numpy as np
import shutil
import json

def remove_all_files(dirpath):
    for filename in os.listdir(dirpath):
        filepath = os.path.join(dirpath, filename)
        try:
            shutil.rmtree(filepath)
        except OSError:
            os.remove(filepath)

def single_task_trial(search_space: dict, rand_seed=42):
    parameter_dict = {}
    for key, value in search_space.items():
        parameter_dict[key] = rand_search_parameter(value)
    parameter_dict['seed'] = rand_seed
    exp_name = 'train.bs' + str(parameter_dict['graph_batch_size']) + \
               '.lr' + str(parameter_dict['learning_rate']) + '.data' +str(parameter_dict['dataset']) + '.seed' +str(rand_seed)
    parameter_dict['exp_name'] = exp_name
    return parameter_dict

def rand_search_parameter(space: dict):
    para_type = space['type']
    if para_type == 'fixed':
        return space['value']
    if para_type == 'choice':
        candidates = space['values']
        value = random.choice(candidates, 1).tolist()[0]
        return value
    if para_type == 'range':
        log_scale = space.get('log_scale', False)
        low, high = space['bounds']
        if log_scale:
            value = random.uniform(low=np.log(low), high=np.log(high), size=1)[0]
            value = np.exp(value)
        else:
            value = random.uniform(low=low, high=high, size=1)[0]
        return value
    else:
        raise ValueError('Training batch mode %s not supported' % para_type)

def HypeParameterSpace():
    learning_rate = {'name': 'learning_rate', 'type': 'choice', 'values': [1e-6, 2e-6, 5e-6]}
    graph_batch_size = {'name': 'graph_batch_size', 'type': 'choice', 'values': [8, 16, 32]}
    gradient_accumulation_steps = {'name': 'gradient_accumulation_steps', 'type': 'choice', 'values': [1]}
    attn_drop = {'name': 'bi_attn_drop', 'type': 'choice', 'values': [0.3]}
    feat_drop = {'name': 'trans_drop', 'type': 'choice', 'values': [0.3]}
    hop_num = {'name': 'hop_num', 'type': 'choice', 'values': [2]}
    num_train_epochs = {'name': 'num_train_epochs', 'type': 'choice', 'values': [200]}
    fanouts = {'name': 'fanouts', 'type': 'choice', 'values': ['15,10']} # 300
    ent_dim = {'name': 'ent_dim', 'type': 'choice', 'values': [256]} # 300
    rel_dim = {'name': 'rel_dim', 'type': 'choice', 'values': [256]}
    graph_hid_dim = {'name': 'graph_hid_dim', 'type': 'choice', 'values': [256]}
    head_num = {'name': 'head_num', 'type': 'choice', 'values': [8]}
    layers = {'name': 'layers', 'type': 'choice', 'values': [2]}
    dataset = {'name': 'dataset', 'type': 'choice', 'values': ['wn18rr']}
    #++++++++++++++++++++++++++++++++++
    search_space = [learning_rate, graph_batch_size, gradient_accumulation_steps, attn_drop,
                    num_train_epochs, feat_drop, hop_num, fanouts, ent_dim, rel_dim, graph_hid_dim,
                    head_num, layers, dataset]
    search_space = dict((x['name'], x) for x in search_space)
    return search_space

def generate_random_search_bash(task_num, seed=42):
    relative_path = '../'
    json_file_path = 'configs/kge/'
    job_path = 'kge_jobs/'
    #================================================
    bash_save_path = relative_path + json_file_path
    jobs_path = relative_path + job_path
    if os.path.exists(jobs_path):
        remove_all_files(jobs_path)
    if os.path.exists(bash_save_path):
        remove_all_files(bash_save_path)
    if bash_save_path and not os.path.exists(bash_save_path):
        os.makedirs(bash_save_path)
    if jobs_path and not os.path.exists(jobs_path):
        os.makedirs(jobs_path)
    ##################################################
    search_space = HypeParameterSpace()
    for i in range(task_num):
        rand_hype_dict = single_task_trial(search_space, seed+i)
        config_json_file_name = 'train.data.' + rand_hype_dict['dataset'] \
                                +'.lr.'+ str(rand_hype_dict['learning_rate']) + 'bs.' + rand_hype_dict['graph_batch_size']\
                                + str(rand_hype_dict['seed']) + '.json'
        with open(os.path.join(bash_save_path, config_json_file_name), 'w') as fp:
            json.dump(rand_hype_dict, fp)
        print('{}\n{}'.format(rand_hype_dict, config_json_file_name))
        with open(jobs_path + 'jdhgn_' + config_json_file_name +'.sh', 'w') as rsh_i:
            command_i = "CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.launch --nproc_per_node=4 jdtrain.py --config_file " + \
                        json_file_path + config_json_file_name
            rsh_i.write(command_i)
    print('{} jobs have been generated'.format(task_num))

if __name__ == '__main__':
    generate_random_search_bash(task_num=5, seed=42)