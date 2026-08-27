import json

nb_path = r"D:\M3-TTS\notebooks\phase1_dataset_setup.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell index for "Validate Column Mapping" (cell 8)
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "available_cols = list(df.columns)" in src:
        nb["cells"][i]["source"] = [
            "if df is None:\n",
            "    print('ERROR: df not loaded. You must run Cells 4, 5, 6, 7 in order first.')\n",
            "    print('Cell 4 = Unzip, Cell 5 = Detect Paths, Cell 6 = Install, Cell 7 = Load TSV')\n",
            "else:\n",
            "    available_cols = list(df.columns)\n",
            "    print(f'Available columns: {available_cols}')\n",
            "\n",
            "    missing = []\n",
            "    for col_name, col_val in [('audio', AUDIO_COLUMN), ('text', TEXT_COLUMN), ('speaker', SPEAKER_COLUMN)]:\n",
            "        if col_val in available_cols:\n",
            "            print(f'  [OK] {col_name} -> {col_val}')\n",
            "        else:\n",
            "            print(f'  [MISSING] {col_name} column {col_val} not found!')\n",
            "            missing.append(col_val)\n",
            "\n",
            "    if missing:\n",
            "        print(f'WARNING: Missing columns: {missing}')\n",
            "    else:\n",
            "        print('All column mappings valid.')",
        ]
        print(f"Fixed cell {i}")
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done.")
