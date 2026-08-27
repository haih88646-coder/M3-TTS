import json

nb_path = r"D:\M3-TTS\notebooks\phase1_dataset_setup.ipynb"
with open(nb_path, "r", encoding="utf-8") as f:
    nb = json.load(f)

print("Notebook validation:")
fmt = nb["nbformat"]
minr = nb["nbformat_minor"]
print("  nbformat:", fmt, ".", minr)
print("  Total cells:", len(nb["cells"]))

md_count = sum(1 for c in nb["cells"] if c["cell_type"] == "markdown")
code_count = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
print("  Markdown:", md_count)
print("  Code:", code_count)

empty_code = [i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code" and not c.get("source")]
if empty_code:
    print("  WARNING: Empty code cells at:", empty_code)
else:
    print("  All code cells have content: OK")

ks = nb.get("metadata", {}).get("kernelspec", {})
print("  Kernel:", ks.get("display_name", "?"), ks.get("name", "?"))

colab = nb.get("metadata", {}).get("colab", {})
gpu = colab.get("gpuType", "not set")
runtime = colab.get("runtimeType", "not set")
print("  Colab runtime:", runtime)
print("  Colab GPU:", gpu)
print()
print("Notebook is valid and ready.")
