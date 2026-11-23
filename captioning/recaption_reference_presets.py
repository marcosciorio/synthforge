"""Regenerate the algorithmic captions of the reference presets.

Reuses the stored AudioCommons features and corpus statistics.
"""

import glob, json, os, importlib.util
QUI = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("cap", os.path.join(QUI, "captioner.py"))
cap = importlib.util.module_from_spec(spec); spec.loader.exec_module(cap)
NIP = os.path.join(QUI, "..", "data", "Dataset_TAL_48kHz", "not_in_pool")
stats = json.load(open(os.path.join(NIP, "_corpus_stats.json")))
cambiati, altri = [], 0
for f in sorted(glob.glob(os.path.join(NIP, "captions", "*.json"))):
    c = json.load(open(f)); stem = os.path.basename(f)[:-5]
    p = {k: float(v) for k, v in json.load(open(os.path.join(NIP, "json", stem + ".json")))["parameters"].items()}
    feats = c["features"]
    ch = {
        "category": cap.extract_category(stem),
        "brightness": cap.pick_brightness(feats, stats),
        "loudness": cap.pick_loudness(feats, p, stats),
        "weight": cap.pick_weight(feats, p, stats),
        "texture": cap.pick_texture(feats, stats),
        "timeShape": cap.time_shape_from_params(p),
        "space": cap.space_from_params(p),
    }
    old = c["choices"]
    altri += any(old.get(k) != ch.get(k) for k in ch if k != "timeShape")
    if old.get("timeShape") != ch["timeShape"]:
        cambiati.append({"name": stem, "vecchio": old.get("timeShape"), "nuovo": ch["timeShape"],
                         "vecchia_caption": c["caption"]})
    c["choices"] = ch
    c["caption"] = cap.compose_sentence(ch)
    if c.get("captions"):
        c["captions"][0] = c["caption"]
    json.dump(c, open(f, "w"), indent=2, ensure_ascii=False)
for x in cambiati:
    x["nuova_caption"] = json.load(open(os.path.join(NIP, "captions", x["name"] + ".json")))["caption"]
json.dump(cambiati, open(os.path.join(NIP, "_timeshape_changed.json"), "w"), indent=2, ensure_ascii=False)
print(f"timeShape changed: {len(cambiati)}/517; other dimensions changed: {altri}")
