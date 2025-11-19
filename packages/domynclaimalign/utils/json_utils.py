import os
import json

class JsonlStore:
    def __init__(self, base_dir):
        self.base = base_dir
        self._offsets = {}  # rel_path -> list[int] offsets

    def _build_offsets(self, rel_path):
        abspath = os.path.join(self.base, rel_path)
        offs = []
        # Build offsets in binary mode for exact byte counts
        with open(abspath, "rb") as f:
            off = 0
            offs.append(off)
            for line in f:
                off += len(line)
                offs.append(off)
        self._offsets[rel_path] = offs

    def get_line_json(self, rel_path, linenum):
        if rel_path not in self._offsets:
            # Lazily build offsets on first use per file
            self._build_offsets(rel_path)
        offs = self._offsets[rel_path]
        if linenum < 0 or linenum >= len(offs) - 1:
            return None
        abspath = os.path.join(self.base, rel_path)
        with open(abspath, "rb") as f:
            start = offs[linenum]
            end = offs[linenum + 1]
            f.seek(start)
            raw = f.read(end - start)
        try:
            return json.loads(raw.decode("utf-8"))
        except Exception:
            return None


def get_jsonl_store(data_base_dir):
    return JsonlStore(data_base_dir)