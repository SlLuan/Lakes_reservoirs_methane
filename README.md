This project employs machine learning (XGBoost) algorithms combined with physicochemical parameters of water bodies to predict methane (CH₄) emission fluxes 
from lakes and reservoirs across China, assess feature importance, and perform regional‑scale spatial analysis.
The codebase covers the complete workflow, including data modeling, feature optimization, zonal statistics, and provincial emission estimates.
├── model.py                # Core XGBoost training, ensemble prediction and visualisation
├── best_combination.py     # Optimal feature subset selection (exhaustive/heuristic search)
├── 5_zones.py              # Zonal modelling and comparative analysis across five major lake regions (e.g., Tibetan Plateau, Eastern Plain, etc.)
├── province.py             # Provincial‑scale CH₄ emission aggregation and spatial mapping
├── Emission_area.py        # Area‑weighted total emission calculation based on water surface area
├── Flux_area.py            # Spatialisation of methane flux densities over different regions/watersheds
