import subprocess
from io import StringIO
import torch
import pandas as pd
### pip install pandas==1.1.5

def get_single_free_gpu():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory_free'] = gpu_df['memory.free'].apply(lambda x: float(x.rstrip(' [MiB]')))
    gpu_df['memory_used'] = gpu_df['memory.used'].apply(lambda x: float(x.rstrip(' [MiB]')))
    idx = gpu_df['memory_free'].argmax()
    used_memory = gpu_df.iloc[idx]['memory_used']
    print('Returning GPU{} with {} free MiB'.format(idx, gpu_df.iloc[idx]['memory.free']))
    return idx, used_memory

def single_free_cuda():
    free_gpu_id, used_memory = get_single_free_gpu()
    device = torch.device('cuda:'+str(free_gpu_id))
    torch.cuda.set_device(device=device)
    return [free_gpu_id], used_memory

def get_multi_free_gpus():
    gpu_stats = subprocess.check_output(["nvidia-smi", "--format=csv", "--query-gpu=memory.used,memory.free"])
    gpu_df = pd.read_csv(StringIO(gpu_stats.decode("utf-8")),
                         names=['memory.used', 'memory.free'],
                         skiprows=1)
    print('GPU usage:\n{}'.format(gpu_df))
    gpu_df['memory_free'] = gpu_df['memory.free'].apply(lambda x: float(x.rstrip(' [MiB]')))
    gpu_df['memory_used'] = gpu_df['memory.used'].apply(lambda x: float(x.rstrip(' [MiB]')))
    idx_ = gpu_df['memory_free'].argmax()
    used_memory = gpu_df.iloc[idx_]['memory_used']
    free_idxs = []
    for idx, row in gpu_df.iterrows():
        if row['memory_used'] <= used_memory:
            free_idxs.append(idx)
    print('Returning GPU {} with smaller than {} free MiB'.format(free_idxs, gpu_df.iloc[idx_]['memory.free']))
    return free_idxs, used_memory

def multi_free_cuda():
    free_gpu_ids, used_memory = get_multi_free_gpus()
    return free_gpu_ids, used_memory

def gpu_setting(num_gpu=1):
    if num_gpu > 1:
        return multi_free_cuda()
    else:
        return single_free_cuda()