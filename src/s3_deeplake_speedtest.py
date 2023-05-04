import os
import time
import psutil
import deeplake as dl
import numpy as np
import boto3
from termcolor import colored

### 1. SETUP BUCKET 
creds = {'aws_access_key_id': #'', 
         'aws_secret_access_key': #''}

                      # s3: bucket/folder
RESULTS_DATASET_PATH = "s3://performant-dataloader/deeplake_threaded9"

# Create the connection to the source data
s3 = boto3.resource('s3', 
                    aws_access_key_id = creds['aws_access_key_id'], 
                    aws_secret_access_key = creds['aws_secret_access_key'])

s3_bucket = s3.Bucket(RESULTS_DATASET_PATH)

fifty_gb = int(220 * 1e9)
ds = dl.empty(RESULTS_DATASET_PATH, creds = creds, overwrite = True, memory_cache_size=fifty_gb)
print(colored(f"👉 Created output database at {RESULTS_DATASET_PATH}", "cyan", attrs=["reverse", "bold"]))

### 2. Initialize tensor as a column in database
with ds:
  ds.create_tensor("context_vector", htype="generic", dtype=np.float32, sample_compression=None, chunk_compression=None) #) "lz4" compression
  ds.flush()


### 3. Parallel speed test
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


start_time = time.monotonic()
# populate_ds_with_zeros(min_val=min, max_val=max, arr_shape=arr_shape, dtype=dtype).eval(dataset_size, ds, scheduler="ray", num_workers=psutil.cpu_count(), skip_ok=True)
populate_ds_with_zeros(min_val=min, max_val=max, arr_shape=arr_shape, dtype=dtype).eval(dataset_size, ds, scheduler='processed', num_workers=psutil.cpu_count()*2, skip_ok=True)
print(f"⏰ Populate Runtime: {(time.monotonic() - start_time):.3f} seconds")
print(f"Dataset size: {len(dataset_size)}")


print(ds.summary())

print(colored('Done!', 'green'))
