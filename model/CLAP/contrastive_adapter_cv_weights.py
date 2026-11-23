"""Contrastive adapter, phase 1b: weight of the InfoNCE term,
with the same folds and training procedure as phase 1.
"""

import os

import contrastive_adapter_cv as base

base.VARIANTI = {f"misto_p{p:g}".replace(".", ""): dict(alpha_cos=1.0, alpha_nce=p, tau=0.07)
                 for p in (0.05, 0.1, 0.2, 0.3, 0.5)}
base.OUT = os.path.join(base.QUI, "contrastive_adapter_cv_weights.json")

if __name__ == "__main__":
    base.main()
