import os
import deeplake as dl
import numpy as np
from termcolor import colored
import psutil
import time

# construct dataset 
# RESULTS_DATASET_PATH = '/mnt/weka/deeplake/deeplake-ds-200k'
# RESULTS_DATASET_PATH = '/dev/kas_temp/deeplake-ds-25k'
RESULTS_DATASET_PATH = '/fsx/kas_temp/deeplake-ds-200k'

os.remove(os.path.join(RESULTS_DATASET_PATH, 'dataset_lock.lock')) if os.path.exists(os.path.join(RESULTS_DATASET_PATH, 'dataset_lock.lock')) else None

# Populate dataset, in parallel, using all but 1 CPU core. 
@dl.compute
def populate_ds_with_zeros(sample_in, sample_out, min_val, max_val, arr_shape, dtype=np.float32):
  # caption = sample_in.caption.numpy()
  sample_out.context_vector.append( np.array(np.random.uniform(min_val,max_val,arr_shape), dtype=dtype) )
  return sample_out

# experiment settings
min = np.finfo(np.float32).min
max = np.finfo(np.float32).max
arr_shape = (1024,1024)
dtype = np.float32
dataset_size = [None] * 200_000

fifty_gb = int(20 * 1e9)
output_ds = dl.load(RESULTS_DATASET_PATH, read_only=False, memory_cache_size=fifty_gb)

start_time = time.monotonic()
populate_ds_with_zeros(min_val=min, max_val=max, arr_shape=arr_shape, dtype=dtype).eval(dataset_size, output_ds, scheduler="ray", num_workers=psutil.cpu_count()-1, skip_ok=True)
print(f"⏰ Populate Runtime: {(time.monotonic() - start_time):.3f} seconds")
print(f"Dataset size: {len(dataset_size)}")


print(output_ds.summary())

print(colored('Done!', 'green'))