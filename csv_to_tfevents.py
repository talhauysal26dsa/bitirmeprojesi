import csv
import os
from torch.utils.tensorboard import SummaryWriter

def convert_csv_to_tfevents(csv_path, log_dir):
    if not os.path.exists(csv_path):
        print(f"Error: {csv_path} not found.")
        return

    # Use the existing log directory or create a new one 'runs/tensorboard_logs' if preferred
    # Here we write to the same directory so tensorboard --logdir runs/detect picks it up
    writer = SummaryWriter(log_dir=log_dir)

    print(f"Reading {csv_path}...")
    with open(csv_path, 'r') as f:
        reader = csv.DictReader(f)
        # Clean field names (strip whitespace)
        reader.fieldnames = [name.strip() for name in reader.fieldnames]
        
        count = 0
        for row in reader:
            try:
                epoch = int(row['epoch'].strip())
                # Log all other columns as scalars
                for key, value in row.items():
                    if key == 'epoch':
                        continue
                    try:
                        val = float(value.strip())
                        writer.add_scalar(key, val, epoch)
                    except ValueError:
                        pass # Ignore non-numeric
                count += 1
            except ValueError:
                continue # Skip bad rows
                
    writer.close()
    print(f"Successfully converted {count} epochs to TensorBoard events in {log_dir}")

if __name__ == "__main__":
    csv_file = "/home/ilaha/bitirmeprojesi/runs/detect/train/results.csv"
    log_dir = "/home/ilaha/bitirmeprojesi/runs/detect/train" 
    convert_csv_to_tfevents(csv_file, log_dir)
