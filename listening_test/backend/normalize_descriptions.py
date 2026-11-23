"""Convert the guided descriptions collected by the website into the common
caption schema used by the captioner.
"""

import glob
import json
import os

QUI = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(QUI, "results_server", "results")
OUT = os.path.join(QUI, "results_server", "unified_descriptions.json")

VOC = {
    "category": ["lead", "bass", "fx", "pad", "drum", "keyboard", "arp", "chord", "pluck"],
    "brightness": ["bright", "dark", "warm", "mellow", "sharp", "muffled"],
    "texture": ["smooth", "rough", "clean", "harsh", "buzzy", "metallic", "hollow", "lush"],
    "loudness": ["soft", "gentle", "loud", "powerful", "aggressive"],
    "weight": ["deep", "full", "thin", "light", "heavy"],
    "timeShape": ["short", "long", "slow", "fading", "pulsing", "evolving"],
}
CATEGORIA = {"key": "keyboard", "percussion": "drum"}


def componi(c):
    cat = c.get("category") or "synthesizer"
    desc = [c[k] for k in ("brightness", "texture", "loudness", "weight") if c.get(k)]
    art = lambda s: "an" if s[:1].lower() in "aeiou" else "a"
    sogg = desc[0] if len(desc) == 1 else (", ".join(desc[:-1]) + " and " + desc[-1] if desc else "")
    s = f"The sound of {art(sogg)} {sogg} {cat}" if sogg else f"The sound of {art(cat)} {cat}"
    return s + (f", {c['timeShape']}" if c.get("timeShape") else "")


def main():
    out = []
    for f in sorted(glob.glob(os.path.join(SRC, "*", "*", "description.json"))):
        d = json.load(open(f))
        g = dict(d.get("guided_choices") or {})
        versione = "v1" if ({"pitch", "comparison"} & set(g)) else ("v2" if "weight" in g else "?")
        u, note = {}, []
        for k, v in g.items():
            if k == "category":
                v2 = CATEGORIA.get(v, v)
                if v2 in VOC["category"]:
                    u["category"] = v2
                    if v2 != v:
                        note.append(f"category {v}->{v2}")
                else:
                    note.append(f"category '{v}' has no equivalent: omitted")
            elif k == "pitch":
                if v in VOC["weight"] and "weight" not in g:
                    u["weight"] = v
                    note.append(f"pitch {v}->weight")
                else:
                    note.append(f"pitch '{v}' scartato")
            elif k in VOC:
                if v in VOC[k]:
                    u[k] = v
                else:
                    note.append(f"{k} '{v}' fuori vocabolario: omesso")
            else:
                note.append(f"{k} scartato")
        out.append({
            "sample_id": d["sample_id"], "versione_sito": versione,
            "esperienza": d.get("describer_expertise"), "esperienza_label": d.get("describer_expertise_label"),
            "libera_originale": d.get("free_description"),
            "guidata_originale": d.get("guided_description"), "scelte_originali": g,
            "scelte_unificate": u, "guidata_unificata": componi(u), "note_conversione": note,
            "submitted_at": d.get("submitted_at"),
        })
    json.dump(out, open(OUT, "w"), indent=2, ensure_ascii=False)
    print(f"{len(out)} descriptions written to {OUT}")


if __name__ == "__main__":
    main()
