🔹 System & Host Identifiers

- sy_name → System name (can include star + planet system, e.g. “Kepler-10”).

- hostname → Host star name (same system may have multiple planets).

- tic_id → TESS Input Catalog ID.

🔹 Star & Planet Counts

- sy_snum → Number of stars in the system (single, binary, multiple).

- sy_pnum → Number of detected planets.

🔹 Stellar Properties

- st_spectype → Spectral type (O, B, A, F, G, K, M → hot → cool).

- st_teff → Effective temperature (K).

- st_rad → Stellar radius (in Solar radii).

- st_mass → Stellar mass (in Solar masses).

- st_met → Stellar metallicity (dex, relative to Sun; higher means more metals → more likely planets).

- st_logg → Surface gravity (log10 cm/s²).

Each of these also has error estimates:

- st_tefferr1, st_tefferr2 → upper/lower uncertainties.

Same for radius, mass, metallicity, logg.

🔹 System Position

- ra, dec → Right Ascension & Declination (sky coordinates, in degrees).

- rastr, decstr → Same, but in sexagesimal (hh:mm:ss, ±dd:mm:ss).

🔹 Distance & Brightness

- sy_dist → Distance to system (in parsecs).

- sy_vmag → Visual magnitude (Johnson filter).

- sy_kmag → Infrared magnitude (2MASS Ks filter).

- sy_gaiamag → Gaia magnitude.

