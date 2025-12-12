import kagglehub

# Download latest version
path = kagglehub.dataset_download("ultralytics/coco128")
print(path)  # Output: /path/to/downloaded/dataset