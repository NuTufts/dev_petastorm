import os,sys
import torch
from petastorm import make_reader, TransformSpec
from petastorm.pytorch import DataLoader

torch.manual_seed(1)

dataset_folder = 'file:///tmp/test_flash_dataset'
#dataset_list = ['file:///tmp/test_flash_dataset/dlmerged_mcc9_v13_bnbnue_corsika.root']
#dataset_list += ['file:///tmp/test_flash_dataset/dlmerged_mcc9_v13_bnbnue_corsika_v2.root']
#dataset_folder += ['file:///tmp/test_flash_dataset/dlmerged_mcc9_v13_bnbnue_corsika_v2.root']
#dataset_folder = 'file:///tmp/test_flash_dataset/dlmerged_mcc9_v13_bnbnue_corsika.root'

#pout = os.popen('find /tmp/test_flash_dataset/ | grep parquet | grep -v parquet.crc')
#out = pout.readlines()
#dataset_list = []
#for l in out:
#    dataset_list.append('file://'+l.strip())
#print(dataset_list)


def _transform_row( row ):
    #print(row)
    result = {"array_flashpe":row["array_flashpe"],
              "event":row["event"]}
    return result

transform = TransformSpec(_transform_row, removed_fields=['sourcefile','run','subrun','trackindex'])
#transform = TransformSpec(_transform_row, removed_fields=[])
#reader = make_reader( dataset_folder, num_epochs=1, transform_spec=transform, seed=1, shuffle_rows=False )
#for row in reader:
#    print(row)

with DataLoader( make_reader(dataset_folder, num_epochs=1, transform_spec=transform, seed=1, shuffle_rows=False ),
#with DataLoader( make_reader(*dataset_list, num_epochs=2, transform_spec=transform, seed=1, shuffle_rows=False ),
                 batch_size=1 ) as loader:

    for batch_idx, row in enumerate(loader):
        print("BATCH[",batch_idx,"] ==================")
        print(" event: ",row['event'])

