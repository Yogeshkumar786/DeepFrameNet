import os
import json

def generate_metadata(dataset_root="videos"):
    metadata = []
    for class_name in ["real", "fake"]:
        class_dir = os.path.join(dataset_root, class_name)
        for video in os.listdir(class_dir):
            metadata.append({
                "file": f"{class_name}/{video}",
                "n_fakes": 1 if class_name == "fake" else 0,
                "split": "train"
            })
    with open("metadata.json", "w") as f:
        json.dump(metadata, f)

if __name__ == "__main__":
    generate_metadata()
