import os
import glob
import json
from tqdm import tqdm
from icecream import ic
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.ipc as ipc


# input_dir = "../data/data_cache/allenai___olmo-mix-1124/wiki/0.0.0/99ee6aaace88779d1ef099d36251b91101c1679b"
# output_dir = "../data/wiki"
# os.makedirs(output_dir, exist_ok=True)
# arrow_files = sorted(glob.glob(os.path.join(input_dir, "*.arrow")))
# ic(f"Found {len(arrow_files)} arrow files")

def convert_arrow_to_jsonl(arrow_path: str, jsonl_path: str):
    """
    Convert a single Arrow file to JSONL format.

    Args:
        arrow_path (str): Path to the input Arrow file.
        jsonl_path (str): Path to the output JSONL file.
    """
    try:
        # Try as IPC file (RecordBatchFile)
        with ipc.open_file(arrow_path) as reader:
            table = reader.read_all()
    except Exception as e1:
        ic(f"ipc.open_file failed: {e1}, trying stream reader...")
        try:
            # Try IPC stream format (RecordBatchStream)
            with open(arrow_path, 'rb') as f:
                reader = ipc.RecordBatchStreamReader(f)
                table = reader.read_all()
        except Exception as e2:
            ic(f"StreamReader also failed: {e2}, trying dataset reader...")
            try:
                # Fall back to pyarrow dataset API (for folder or file dataset)
                dataset = ds.dataset(arrow_path, format="arrow")
                table = dataset.to_table()
            except Exception as e3:
                ic(f"Failed loading with dataset API: {e3}")
                return

    df = table.to_pandas()

    with open(jsonl_path, 'w', encoding='utf-8') as jsonl_out:
        for text in tqdm(df['text'], desc=f"Writing {os.path.basename(jsonl_path)}", leave=False):
            jsonl_out.write(json.dumps({'text': text}, ensure_ascii=False) + '\n')
    ic("Conversion complete.")

