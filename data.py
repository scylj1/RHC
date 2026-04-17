from datasets import load_from_disk
ds = load_from_disk("data/train_us")
print(ds)
labels = ds['main_ipcr_label']
print(labels[0])
claims = ds['claims']
print(claims[0])
