import os, json

DATA_ROOT = os.path.join(os.sep, 'datasets', 'bdd100k')

train_meta_data_loc = os.path.join(DATA_ROOT, 'labels', 'bdd100k_labels_images_train.json')
val_meta_data_loc = os.path.join(DATA_ROOT, 'labels', 'bdd100k_labels_images_val.json')
train_seg_data_loc = os.path.join(DATA_ROOT, 'seg', 'images', 'train')
val_seg_data_loc = os.path.join(DATA_ROOT, 'seg', 'images', 'val')

train_files = [f for f in os.listdir(train_seg_data_loc) if os.path.isfile(os.path.join(train_seg_data_loc, f))]
val_files = [f for f in os.listdir(val_seg_data_loc) if os.path.isfile(os.path.join(val_seg_data_loc, f))]

all_files = train_files + val_files

with open(train_meta_data_loc, 'r') as f:
    train_meta_data = json.load(f)
with open(val_meta_data_loc, 'r') as f:
    val_meta_data = json.load(f)

train_meta_data_files = [str(inst['name']) for inst in train_meta_data]
val_meta_data_files = [str(inst['name']) for inst in val_meta_data]
all_meta_data_files = train_meta_data_files + val_meta_data_files

train_train_intersection = set(train_meta_data_files).intersection(train_files)
train_val_intersection = set(train_meta_data_files).intersection(val_files)
val_val_intersection = set(val_meta_data_files).intersection(val_files)
val_train_intersection = set(val_meta_data_files).intersection(train_files)
all_intersection = set(all_meta_data_files).intersection(all_files)

print("Number train files", len(train_files))
print("Number val files", len(val_files))
print("Train/train metadata intersection", len(train_train_intersection))
print("Train/val metadata intersection", len(train_val_intersection))
print("val/train metadata intersection", len(val_val_intersection))
print("val/train metadata intersection", len(val_train_intersection))
print()
print("Total number of files w/ metadata", len(all_intersection))

