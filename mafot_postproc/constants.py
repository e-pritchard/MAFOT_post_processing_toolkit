"""
Physical constants and pipeline-wide defaults.

Keep this module import-only — no functions, no state.  Every other
module in the package can import from here.
"""

# --- Physical constants (SI) ---
QE = 1.602176634e-19       # Coulomb; also J/eV
MP = 1.67262192369e-27     # kg, proton mass
ME = 9.1093837e-31         # kg, electron mass

# --- Sheath transmission coefficients (Stangeby, "Plasma Boundary" §25) ---
ION_TRANSMISSION_COEFF = 2.5      # gamma_i, standard sheath value
ELECTRON_TRANSMISSION_COEFF = 7.0  # gamma_e, includes pre-sheath potential

# --- Connection-length cutoff for the "orbit reached the wall" mask ---
LC_MIN_KM_DEFAULT = 0.075

# --- Deuterium as default ion ---
DEFAULT_ION_MASS_KG = 2.0 * MP
DEFAULT_ELECTRON_MASS_KG = ME
DEFAULT_Z_ION = 1

# --- Miscellaneous ---
DEFAULT_FILE_TAG = "unnamed_pipeline"
