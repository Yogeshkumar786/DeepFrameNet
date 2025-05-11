import os
import json
import random

def generate_metadata(input_dir='videos', output_path='LAV-DF/metadata.json'):
    data = []
    for label_type in ['real', 'fake']:
        folder = os.path.join(input_dir, label_type)
        for filename in os.listdir(folder):
            if filename.endswith('.mp4'):
                entry = {
                    'file': os.path.join(label_type, filename),
                    'split': 'train',  # You can change this later if needed
                    'n_fakes': 0 if label_type == 'real' else 1
                }
                data.append(entry)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"Metadata saved to {output_path} with {len(data)} entries.")

if __name__ == '__main__':
    generate_metadata()
