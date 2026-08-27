import json

nb_path = r"D:\M3-TTS\notebooks\phase1_dataset_setup.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

# Find cell for "Load and Inspect TSV" (cell 7)
for i, cell in enumerate(nb["cells"]):
    src = "".join(cell.get("source", []))
    if "Load and Inspect TSV" in src and cell["cell_type"] == "markdown":
        # Replace the NEXT cell (the code cell)
        nb["cells"][i+1]["source"] = [
            "# ====== ROBUST LOAD CELL ======\n",
            "# This cell handles: zip extraction, path detection, TSV loading\n",
            "\n",
            "import zipfile, glob\n",
            "\n",
            "# Step 1: Extract zip if needed\n",
            "if not os.path.exists(EXTRACT_DIR) or not os.listdir(EXTRACT_DIR):\n",
            "    print('Searching for zip in Google Drive...')\n",
            "    candidates = glob.glob('/content/drive/MyDrive/*.zip')\n",
            "    print(f'  Found zips: {candidates}')\n",
            "    zip_file = None\n",
            "    for z in candidates:\n",
            "        if 'khmer' in z.lower() or 'tts' in z.lower():\n",
            "            zip_file = z\n",
            "            break\n",
            "    if not zip_file and candidates:\n",
            "        zip_file = candidates[0]\n",
            "    if zip_file:\n",
            "        print(f'  Extracting: {zip_file}')\n",
            "        os.makedirs(EXTRACT_DIR, exist_ok=True)\n",
            "        with zipfile.ZipFile(zip_file, 'r') as z:\n",
            "            z.extractall(EXTRACT_DIR)\n",
            "        print(f'  Done. Contents: {os.listdir(EXTRACT_DIR)}')\n",
            "    else:\n",
            "        print('  No zip found. Listing MyDrive:')\n",
            "        for item in sorted(os.listdir('/content/drive/MyDrive')):\n",
            "            print(f'    {item}')\n",
            "\n",
            "# Step 2: Find TSV and audio dir\n",
            "if os.path.exists(EXTRACT_DIR) and os.listdir(EXTRACT_DIR):\n",
            "    tsv_files = []\n",
            "    audio_dirs = []\n",
            "    for dirpath, dirnames, filenames in os.walk(EXTRACT_DIR):\n",
            "        for f in filenames:\n",
            "            if f.lower().endswith('.tsv'):\n",
            "                tsv_files.append(os.path.join(dirpath, f))\n",
            "        wav_count = sum(1 for f in filenames if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')))\n",
            "        if wav_count > 0:\n",
            "            audio_dirs.append((dirpath, wav_count))\n",
            "\n",
            "    # Pick TSV\n",
            "    METADATA_TSV = None\n",
            "    for t in tsv_files:\n",
            "        if 'metadata' in os.path.basename(t).lower():\n",
            "            METADATA_TSV = t\n",
            "            break\n",
            "    if not METADATA_TSV and tsv_files:\n",
            "        METADATA_TSV = tsv_files[0]\n",
            "\n",
            "    # Pick audio dir\n",
            "    AUDIO_DIR = max(audio_dirs, key=lambda x: x[1])[0] if audio_dirs else None\n",
            "    DATASET_DIR = os.path.dirname(METADATA_TSV) if METADATA_TSV else None\n",
            "\n",
            "    if METADATA_TSV:\n",
            "        print(f'TSV:  {METADATA_TSV}')\n",
            "    if AUDIO_DIR:\n",
            "        wav_count = sum(1 for f in os.listdir(AUDIO_DIR) if f.lower().endswith(('.wav', '.mp3', '.flac', '.ogg')))\n",
            "        print(f'Audio: {AUDIO_DIR} ({wav_count} files)')\n",
            "\n",
            "    # Step 3: Load TSV\n",
            "    if METADATA_TSV and os.path.exists(METADATA_TSV):\n",
            "        df = pd.read_csv(METADATA_TSV, sep='\\t', dtype=str)\n",
            "        print(f'\\nColumns: {list(df.columns)}')\n",
            "        print(f'Rows: {len(df)}')\n",
            "        display(df.head())\n",
            "        print(f'\\nData types:')\n",
            "        print(df.dtypes)\n",
            "        print(f'\\nMissing values:')\n",
            "        print(df.isnull().sum())\n",
            "    else:\n",
            "        print('ERROR: No TSV found.')\n",
            "        for root, dirs, files in os.walk(EXTRACT_DIR):\n",
            "            print(f'  {root}: {files[:5]}')\n",
            "else:\n",
            "    print('ERROR: Could not extract zip. Check that khmer_tts_data.zip is in MyDrive.')",
        ]
        print(f"Replaced cell {i+1} (Load TSV)")
        break

with open(nb_path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("Done.")
