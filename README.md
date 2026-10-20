# Mie scattering homework

Open `HW5_Mie.ipynb` in VS Code or Jupyter and run its cells from top to bottom. Each result appears in its own section. The notebook has no precomputed outputs. Alternatively, run `python mie_homework.py` from this extracted folder. Install dependencies with `python -m pip install -r requirements.txt`. The script writes figures, spectra, numerical comparisons, and a brief results summary to `results/`. Use `python mie_homework.py --show` to open plot windows. For a faster run without field maps, use `python mie_homework.py --no-fields`.

The Ag optical-constant input is `data.d`, beside the script. Its three columns are wavelength in µm, refractive index `n`, and extinction coefficient `kappa` (Johnson–Christy data). You may replace it with another file in the same format; the program sorts wavelengths before interpolation and never extrapolates. Aluminum optical data are in `data/Al_Hagemann_1974.csv`. Use `--no-experimental` if either optical-data comparison is not needed.

The assigned Drude runs use relative magnetic permeability 1 and radii 1, 5, 20, 50, and 100 nm. The source is divided into short numbered sections. The plotted N=5 spectra are accompanied by convergence comparisons with stabilized higher-order Mie sums; inspect them for the large aluminum spheres.
