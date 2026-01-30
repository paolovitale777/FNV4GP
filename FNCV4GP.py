import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import queue

import pandas as pd
import numpy as np

# =============================
# ML imports for Nested CV (SVR)
# =============================
from sklearn.svm import SVR
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from scipy.stats import pearsonr
from itertools import product
from sklearn.linear_model import Ridge
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import BayesianRidge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import SGDRegressor
from sklearn.neighbors import KNeighborsRegressor

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic, ExpSineSquared, ConstantKernel, DotProduct, Matern

from sklearn.cross_decomposition import PLSRegression

from sklearn.tree import DecisionTreeRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.ensemble import RandomForestRegressor

from sklearn.ensemble import VotingRegressor

from sklearn.neural_network import MLPRegressor

import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import time




# -----------------------------
# Constants / helpers
# -----------------------------
MISSING_TOKENS = {"", "NA", "NaN", "nan", "N/A", ".", "-", "N"}

# IUPAC heterozygous codes (biallelic ambiguity)
IUPAC_HET = {
    "R": ("A", "G"),
    "Y": ("C", "T"),
    "S": ("G", "C"),
    "W": ("A", "T"),
    "K": ("G", "T"),
    "M": ("A", "C"),
}
VALID_BASES = {"A", "C", "G", "T"}


def detect_sep(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return "\t" if ext in [".tsv", ".txt", ".hmp", ".hapmap"] else ","


def normalize_call(x):
    """Return normalized genotype call string or np.nan."""
    if x is None:
        return np.nan
    s = str(x).strip()
    if s in MISSING_TOKENS:
        return np.nan
    return s.upper()


# -----------------------------
# Readers
# -----------------------------
def read_numeric_marker_matrix(path: str) -> pd.DataFrame:
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str)

    if df.shape[1] < 2:
        raise ValueError("File must have at least 2 columns (ID + markers).")

    gid_col = df.columns[0]
    markers = df.columns[1:]

    df[markers] = df[markers].applymap(
        lambda x: np.nan if (x is None or str(x).strip() in MISSING_TOKENS) else str(x).strip()
    )

    for c in markers:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df[gid_col] = df[gid_col].astype(str)
    return df


def hapmap_to_numeric_matrix(path: str, major_hom=2, minor_hom=0, het_code=1) -> pd.DataFrame:
    sep = detect_sep(path)
    df = pd.read_csv(path, sep=sep, dtype=str)

    if df.shape[1] <= 11:
        raise ValueError("This file does not look like HapMap (needs > 11 columns).")

    meta = df.iloc[:, :11].copy()
    geno = df.iloc[:, 11:].copy()  # taxa columns

    marker_id_col = meta.columns[0]
    marker_ids = meta[marker_id_col].astype(str).tolist()

    def build_fallback_ids():
        chrom_col = None
        pos_col = None
        for c in meta.columns:
            cl = c.lower()
            if chrom_col is None and ("chrom" in cl or cl == "chr"):
                chrom_col = c
            if pos_col is None and (cl == "pos" or "position" in cl):
                pos_col = c
        if chrom_col is not None and pos_col is not None:
            return (meta[chrom_col].astype(str) + "_" + meta[pos_col].astype(str)).tolist()
        return [f"M{i+1}" for i in range(meta.shape[0])]

    if len(set(marker_ids)) != len(marker_ids) or any(m.strip() == "" for m in marker_ids):
        marker_ids = build_fallback_ids()

    geno = geno.applymap(normalize_call)

    taxa = list(geno.columns)
    n_markers = geno.shape[0]
    out = pd.DataFrame(index=taxa, columns=marker_ids, dtype=float)

    for i in range(n_markers):
        calls = geno.iloc[i, :]

        allele_counts = {b: 0 for b in VALID_BASES}
        for v in calls.values:
            if pd.isna(v):
                continue
            if v in VALID_BASES:
                allele_counts[v] += 2
            elif v in IUPAC_HET:
                a1, a2 = IUPAC_HET[v]
                allele_counts[a1] += 1
                allele_counts[a2] += 1

        if sum(allele_counts.values()) == 0:
            continue

        major_allele = max(allele_counts.items(), key=lambda kv: kv[1])[0]
        marker_name = marker_ids[i]

        for taxon, v in calls.items():
            if pd.isna(v):
                out.at[taxon, marker_name] = np.nan
            elif v in VALID_BASES:
                out.at[taxon, marker_name] = major_hom if v == major_allele else minor_hom
            elif v in IUPAC_HET:
                out.at[taxon, marker_name] = het_code
            else:
                out.at[taxon, marker_name] = np.nan

    return out.reset_index().rename(columns={"index": "Taxon"})


# -----------------------------
# Marker statistics / filtering
# -----------------------------
def maf_of_marker_numeric(x: pd.Series) -> float:
    x = x.dropna()
    if x.empty:
        return np.nan
    if np.nanstd(x.values) == 0:
        return 0.0
    p = np.nanmean(x.values) / 2.0
    p = min(max(p, 0.0), 1.0)
    return min(p, 1.0 - p)


def filter_markers(df: pd.DataFrame, maf_thr: float, max_missing_mrk: float, max_het_mrk: float):
    gid_col = df.columns[0]
    markers = df.columns[1:]
    X = df[markers]

    missing_rate = X.isna().mean(axis=0)

    non_missing = X.notna().sum(axis=0).replace(0, np.nan)
    het_count = (X == 1).sum(axis=0)
    het_rate = (het_count / non_missing).astype(float)

    maf = X.apply(maf_of_marker_numeric, axis=0)

    keep = (missing_rate <= max_missing_mrk) & (maf >= maf_thr) & (het_rate <= max_het_mrk)

    summary = {
        "n_markers_before": int(len(markers)),
        "n_markers_after": int(keep.sum()),
        "removed_total": int((~keep).sum()),
        "removed_missing": int((missing_rate > max_missing_mrk).sum()),
        "removed_maf": int((maf < maf_thr).sum()),
        "removed_het": int((het_rate > max_het_mrk).sum()),
    }

    out_df = pd.concat([df[[gid_col]], X.loc[:, keep]], axis=1)

    details = pd.DataFrame({
        "marker": markers,
        "missing_rate": missing_rate.values,
        "het_rate": het_rate.values,
        "maf": maf.values,
        "keep": keep.values
    })

    return out_df, summary, details


def filter_genotypes(df: pd.DataFrame, max_missing_ind: float, max_het_ind: float):
    id_col = df.columns[0]
    markers = df.columns[1:]
    X = df[markers]

    missing_rate = X.isna().mean(axis=1)

    non_missing = X.notna().sum(axis=1).replace(0, np.nan)
    het_count = (X == 1).sum(axis=1)
    het_rate = (het_count / non_missing).astype(float)

    keep = (missing_rate <= max_missing_ind) & (het_rate <= max_het_ind)

    out_df = df.loc[keep].reset_index(drop=True)

    summary = {
        "n_genotypes_before": int(df.shape[0]),
        "n_genotypes_after": int(out_df.shape[0]),
        "removed_genotypes_total": int((~keep).sum()),
        "removed_genotypes_missing": int((missing_rate > max_missing_ind).sum()),
        "removed_genotypes_het": int((het_rate > max_het_ind).sum()),
    }

    details = pd.DataFrame({
        "genotype": df[id_col].astype(str).values,
        "missing_rate": missing_rate.values,
        "het_rate": het_rate.values,
        "keep": keep.values
    })

    return out_df, summary, details


# -----------------------------
# Imputation
# -----------------------------
def impute_matrix(df: pd.DataFrame, method: str) -> pd.DataFrame:
    if method == "None":
        return df

    out = df.copy()
    markers = out.columns[1:]
    X = out[markers]

    if method == "Mean":
        means = X.mean(axis=0, skipna=True)
        out[markers] = X.fillna(means)
        return out

    if method == "Major allele":
        modes = {}
        for m in markers:
            s = X[m].dropna()
            if s.empty:
                modes[m] = np.nan
            else:
                modes[m] = float(s.value_counts().idxmax())
        out[markers] = X.fillna(pd.Series(modes))
        return out

    raise ValueError(f"Unknown imputation method: {method}")


# -----------------------------
# Main GUI
# -----------------------------
class MarkerFilterApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("FNCV4GP")
        self.geometry("1120x780")

        # --- Scrollable container (Canvas + inner frame) ---
        self._canvas = tk.Canvas(self, highlightthickness=0)
        self._vscroll = ttk.Scrollbar(self, orient="vertical", command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vscroll.set)

        self._vscroll.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)

        # Inner frame that holds ALL widgets (including pages)
        self._main = ttk.Frame(self._canvas)
        self._main_window = self._canvas.create_window((0, 0), window=self._main, anchor="nw")

        # Keep scrollregion updated
        self._main.bind("<Configure>", self._on_frame_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)

        # Mouse wheel scrolling
        self._canvas.bind_all("<MouseWheel>", self._on_mousewheel)      # Windows/macOS
        self._canvas.bind_all("<Button-4>", self._on_mousewheel_linux)  # Linux up
        self._canvas.bind_all("<Button-5>", self._on_mousewheel_linux)  # Linux down

        # --- variables / state ---
        self.input_path = tk.StringVar(value="")
        self.output_path = tk.StringVar(value="")
        self.input_type = tk.StringVar(value="Numeric matrix")

        # Marker-level thresholds
        self.maf_thr = tk.DoubleVar(value=0.05)
        self.max_missing_marker = tk.DoubleVar(value=0.50)
        self.max_het_marker = tk.DoubleVar(value=1.00)

        # Genotype-level thresholds (applied AFTER marker filtering)
        self.max_missing_genotype = tk.DoubleVar(value=1.00)
        self.max_het_genotype = tk.DoubleVar(value=1.00)

        # Imputation
        self.imputation_method = tk.StringVar(value="None")

        # Outputs
        self.df = None
        self.marker_details = None
        self.genotype_details = None

        # Threading
        self._worker_thread = None
        self._result_q = queue.Queue()

        # Build UI (pages)
        self._build_ui()


    # -----------------------------
    # Scroll handling
    # -----------------------------
    def _on_frame_configure(self, event):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, event):
        # Make inner frame same width as canvas (prevents weird right alignment)
        self._canvas.itemconfig(self._main_window, width=event.width)
        self._canvas.xview_moveto(0)

    def _on_mousewheel(self, event):
        if event.delta:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux(self, event):
        if event.num == 4:
            self._canvas.yview_scroll(-3, "units")
        elif event.num == 5:
            self._canvas.yview_scroll(3, "units")

    # -----------------------------
    # Page navigation
    # -----------------------------
    def _show_frame(self, frame: ttk.Frame):
        frame.tkraise()
        self._canvas.yview_moveto(0)
        self.update_idletasks()
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def show_home(self):
        self._show_frame(self.home_frame)

    def show_filtering(self):
        self._show_frame(self.filter_frame)

    def show_cv(self):
        self._show_frame(self.cv_frame)

    def show_cv_ridge(self):
        self._show_frame(self.cv_frame_ridge)

    def show_cv_kernelridge(self):
        self._show_frame(self.cv_frame_kernelridge)

    def show_cv_lasso(self):
        self._show_frame(self.cv_frame_lasso)

    def show_cv_elasticnet(self):
        self._show_frame(self.cv_frame_elasticnet)

    def show_cv_bayesianridge(self):
        self._show_frame(self.cv_frame_bayesianridge)

    def show_cv_SGDRegressor(self):
        self._show_frame(self.cv_frame_SGDRegressor)

    def show_cv_KNeighborsRegressor(self):
        self._show_frame(self.cv_frame_KNeighborsRegressor)

    def show_cv_GaussianProcessRegressor(self):
        self._show_frame(self.cv_frame_GaussianProcessRegressor)

    def show_cv_PLSRegression(self):
        self._show_frame(self.cv_frame_PLSRegression)

    
    def show_cv_DecisionTreeRegressor(self):
        self._show_frame(self.cv_frame_DecisionTreeRegressor)

    def show_cv_GradientBoostingRegressor(self):
        self._show_frame(self.cv_frame_GradientBoostingRegressor)

    def show_cv_RandomForestRegressor(self):
        self._show_frame(self.cv_frame_RandomForestRegressor)

    def show_cv_VotingRegressor(self):
        self._show_frame(self.cv_frame_VotingRegressor)


    def show_cv_MLPRegressor(self):
        self._show_frame(self.cv_frame_MLPRegressor)

    def show_summary(self):
        self._show_frame(self.summary_frame)






    # -----------------------------
    # UI build
    # -----------------------------
    def _build_ui(self):


        # Top navigation bar
        nav = ttk.Frame(self._main)
        nav.pack(fill="x", padx=10, pady=(10, 0))

        ttk.Button(nav, text="Home", command=self.show_home).pack(side="left", padx=(0, 6))

        ttk.Separator(self._main, orient="horizontal").pack(fill="x", padx=10, pady=(6, 10))

        # Pages container
        pages = ttk.Frame(self._main)
        pages.pack(fill="both", expand=True)

        pages.grid_rowconfigure(0, weight=1)
        pages.grid_columnconfigure(0, weight=1)



        self.home_frame = ttk.Frame(pages)
        self.filter_frame = ttk.Frame(pages)
        self.cv_frame = ttk.Frame(pages)
        self.cv_frame_ridge = ttk.Frame(pages)
        self.cv_frame_kernelridge = ttk.Frame(pages)
        self.cv_frame_lasso = ttk.Frame(pages)
        self.cv_frame_elasticnet = ttk.Frame(pages)
        self.cv_frame_bayesianridge = ttk.Frame(pages)

        self.cv_frame_SGDRegressor = ttk.Frame(pages)
        self.cv_frame_KNeighborsRegressor = ttk.Frame(pages)
        self.cv_frame_GaussianProcessRegressor = ttk.Frame(pages)
        self.cv_frame_PLSRegression = ttk.Frame(pages)
        self.cv_frame_DecisionTreeRegressor = ttk.Frame(pages)
        self.cv_frame_GradientBoostingRegressor = ttk.Frame(pages)
        self.cv_frame_RandomForestRegressor = ttk.Frame(pages)
        self.cv_frame_VotingRegressor = ttk.Frame(pages)
        self.cv_frame_MLPRegressor = ttk.Frame(pages)
        # --- add this near your other frames ---
        self.summary_frame = ttk.Frame(pages)
        
        for f in (self.home_frame, self.filter_frame, self.cv_frame, self.cv_frame_ridge,self.cv_frame_kernelridge, self.cv_frame_lasso, self.cv_frame_elasticnet, self.cv_frame_bayesianridge, self.cv_frame_SGDRegressor, self.cv_frame_KNeighborsRegressor, self.cv_frame_GaussianProcessRegressor,
                  self.cv_frame_PLSRegression, self.cv_frame_DecisionTreeRegressor, self.cv_frame_GradientBoostingRegressor, self.cv_frame_RandomForestRegressor,
                  self.cv_frame_VotingRegressor, self.cv_frame_MLPRegressor, self.summary_frame):
            f.grid(row=0, column=0, sticky="nsew")

        # Build each page
        self._build_home_page(self.home_frame)
        self._build_filtering_page(self.filter_frame)
        self._build_cv_page(self.cv_frame)
        self._build_cv_page_ridge(self.cv_frame_ridge) 
        self._build_cv_page_kernelridge(self.cv_frame_kernelridge) 
        self._build_cv_page_lasso(self.cv_frame_lasso) 
        self._build_cv_page_elasticnet(self.cv_frame_elasticnet) 
        self._build_cv_page_bayesianridge(self.cv_frame_bayesianridge)
        self._build_cv_page_SGDRegressor(self.cv_frame_SGDRegressor)
        self._build_cv_page_KNeighborsRegressor(self.cv_frame_KNeighborsRegressor)
        self._build_cv_page_GaussianProcessRegressor(self.cv_frame_GaussianProcessRegressor)
        self._build_cv_page_PLSRegression(self.cv_frame_PLSRegression)
        self._build_cv_page_DecisionTreeRegressor(self.cv_frame_DecisionTreeRegressor)
        self._build_cv_page_GradientBoostingRegressor(self.cv_frame_GradientBoostingRegressor)
        self._build_cv_page_RandomForestRegressor(self.cv_frame_RandomForestRegressor)
        self._build_cv_page_VotingRegressor(self.cv_frame_VotingRegressor)
        self._build_cv_page_MLPRegressor(self.cv_frame_MLPRegressor)
        self._build_summary_page(self.summary_frame)

        # Start on HOME (clean page)
        self.show_home()

    def _build_home_page(self, parent):
        pad = {"padx": 20, "pady": 20}

        home_box = ttk.Frame(parent)
        home_box.pack(expand=True, anchor="n")


        # Title

          # Title
        ttk.Label(
            home_box,
            text="FNCV4GP",
            font=("Segoe UI", 50, "bold")
        ).pack(pady=(0, 5))


        ttk.Label(
            home_box,
            text="Fast Nested Cross-Validation for Genomic Prediction",
            font=("Segoe UI", 30, "bold")
        ).pack(pady=(0, 5))


        ttk.Label(
            home_box,
            text="Welcome",
            font=("Segoe UI", 30, "bold")
        ).pack(pady=(0, 10))

        ttk.Label(
            home_box,
            text="Choose an analysis module",
            font=("Segoe UI", 20)
        ).pack(pady=(0, 30))



        style = ttk.Style()
        style.configure(
            "Home.TButton",
            font=("Segoe UI", 20)
        )



        ttk.Label(
            home_box,
            text="------------------------------------------------------------------------------",
            font=("Segoe UI", 20)
        ).pack(pady=(0, 30))




        ttk.Label(
            home_box,
            text="Filtering",
            font=("Segoe UI", 20, "bold")
        ).pack(pady=(0, 30))


    # --- Marker filtering button ---
        ttk.Button(
            home_box,
            text="Marker filtering",
            width=60,
            style="Home.TButton",
            command=self.show_filtering
        ).pack(pady=10)

    



        ttk.Label(
            home_box,
            text="------------------------------------------------------------------------------",
            font=("Segoe UI", 20)
        ).pack(pady=(0, 30))



        ttk.Label(
            home_box,
            text="Nested CV",
            font=("Segoe UI", 20, "bold")
        ).pack(pady=(0, 30))



        ttk.Button(
            home_box,
            text="Ridge Regression",
            style="Home.TButton",
            width=60,
            command=self.show_cv_ridge
        ).pack(pady=10)

    
        ttk.Button(
            home_box,
            text="Bayesian Ridge Regression",
            style="Home.TButton",
            width=60,
            command=self.show_cv_bayesianridge
        ).pack(pady=10)
 
    
             
        ttk.Button(
            home_box,
            text="Kernel Ridge Regression",
            style="Home.TButton",
            width=60,
            command=self.show_cv_kernelridge
        ).pack(pady=10)
 

        ttk.Button(
            home_box,
            text="LASSO",
            style="Home.TButton",
            width=60,
            command=self.show_cv_lasso
        ).pack(pady=10)

    
    
    
        # --- Nested CV button (THIS is what you asked about) ---
        ttk.Button(
            home_box,
            text="Support Vector Regression",
            style="Home.TButton",
            width=60,
            command=self.show_cv
        ).pack(pady=10)


        ttk.Button(
            home_box,
            text="Elastic Net",
            style="Home.TButton",
            width=60,
            command=self.show_cv_elasticnet
        ).pack(pady=10)



        ttk.Button(
            home_box,
            text="Stochastic Gradient Descent",
            style="Home.TButton",
            width=60,
            command=self.show_cv_SGDRegressor
        ).pack(pady=10)


        ttk.Button(
            home_box,
            text="Nearest Neighbors",
            style="Home.TButton",
            width=60,
            command=self.show_cv_KNeighborsRegressor
        ).pack(pady=10)

        ttk.Button(
            home_box,
            text="Gaussian Process",
            style="Home.TButton",
            width=60,
            command=self.show_cv_GaussianProcessRegressor
        ).pack(pady=10)


        ttk.Button(
            home_box,
            text="Partial Least Squares",
            style="Home.TButton",
            width=60,
            command=self.show_cv_PLSRegression
        ).pack(pady=10)

        ttk.Button(
            home_box,
            text="Decision Tree",
            style="Home.TButton",
            width=60,
            command=self.show_cv_DecisionTreeRegressor
        ).pack(pady=10)


        ttk.Button(
            home_box,
            text="Gradient Boosting",
            style="Home.TButton",
            width=60,
            command=self.show_cv_GradientBoostingRegressor
        ).pack(pady=10)

        ttk.Button(
            home_box,
            text="Random Forest",
            style="Home.TButton",
            width=60,
            command=self.show_cv_RandomForestRegressor
        ).pack(pady=10)

        ttk.Button(
            home_box,
            text="Multi-Layer Perception",
            style="Home.TButton",
            width=60,
            command=self.show_cv_MLPRegressor
        ).pack(pady=10)



        ttk.Button(
            home_box,
            text="Voting Regressor",
            style="Home.TButton",
            width=60,
            command=self.show_cv_VotingRegressor
        ).pack(pady=10)



        ttk.Label(
            home_box,
            text="------------------------------------------------------------------------------",
            font=("Segoe UI", 20)
        ).pack(pady=(0, 30))



        ttk.Label(
            home_box,
            text="Summary",
            font=("Segoe UI", 20, "bold")
        ).pack(pady=(0, 30))



        ttk.Button(
            home_box,
            text="Predictability Test",
            width=60,
            style="Home.TButton",
            command=self.show_summary
        ).pack(pady=10)

        ttk.Label(
            home_box,
            text="------------------------------------------------------------------------------",
            font=("Segoe UI", 20)
        ).pack(pady=(0, 30))








    def _build_cv_page(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedSvrCvFrame(parent)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_kernelridge(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedKernelRidgeCvFrame(parent)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)


    def _build_cv_page_ridge(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedRidgeCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    

    def _build_cv_page_lasso(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedLassoCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_elasticnet(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedElasticNetCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_bayesianridge(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedBayesianRidgeCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_SGDRegressor(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedSGDRegressorCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_KNeighborsRegressor(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedKNeighborsRegressorCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12) 

    def _build_cv_page_GaussianProcessRegressor(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedGaussianProcessRegressorCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_PLSRegression(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedPLSRegressionCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_DecisionTreeRegressor(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedDecisionTreeRegressorCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_GradientBoostingRegressor(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedGradientBoostingRegressorCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_RandomForestRegressor(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedRandomForestRegressorCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_VotingRegressor(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedVotingRegressorCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_cv_page_MLPRegressor(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = NestedMLPRegressorCvWindow(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)

    def _build_summary_page(self, parent):
        # Create the embedded CV UI directly in this page (NO new window)
        self.cv_widget = SummaryPAPlotPage(parent, self)
        self.cv_widget.pack(fill="both", expand=True, padx=12, pady=12)












    def _build_filtering_page(self, parent: ttk.Frame):
        pad = {"padx": 10, "pady": 8}

        # -----------------
        # Files
        # -----------------
        top = ttk.LabelFrame(parent, text="Files")
        top.pack(fill="x", **pad)

        ttk.Label(top, text="Input type:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Combobox(
            top,
            textvariable=self.input_type,
            values=["Numeric matrix", "HapMap"],
            state="readonly",
            width=18
        ).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(top, text="Input file:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.input_path, width=96).grid(row=1, column=1, columnspan=2, sticky="we", **pad)
        ttk.Button(top, text="Browse", command=self.browse_input).grid(row=1, column=3, **pad)

        ttk.Label(top, text="Output (filtered matrix):").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(top, textvariable=self.output_path, width=96).grid(row=2, column=1, columnspan=2, sticky="we", **pad)
        ttk.Button(top, text="Browse", command=self.browse_output).grid(row=2, column=3, **pad)

        top.columnconfigure(2, weight=1)

        # -----------------
        # Filtering parameters (TWO-COLUMN layout)
        # -----------------
        params = ttk.LabelFrame(parent)
        params.pack(fill="x", **pad)

        params.grid_columnconfigure(0, weight=1)
        params.grid_columnconfigure(1, weight=0)

        params_left = ttk.Frame(params)
        params_left.grid(row=0, column=0, sticky="nsew", padx=(0, 10), pady=0)

        params_right = ttk.Frame(params)
        params_right.grid(row=0, column=1, sticky="ne", padx=(10, 0), pady=0)

        ttk.Label(params_left, text="Filtering parameters (Markers)").grid(row=1, column=0, sticky="w", **pad)

        ttk.Label(params_left, text="MAF threshold (0–0.5):").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(params_left, textvariable=self.maf_thr, width=10).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(params_left, text="Max missing per marker (0–1):").grid(row=2, column=2, sticky="w", **pad)
        ttk.Entry(params_left, textvariable=self.max_missing_marker, width=10).grid(row=2, column=3, sticky="w", **pad)

        ttk.Label(params_left, text="Max heterozygosity per marker (0–1):").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(params_left, textvariable=self.max_het_marker, width=10).grid(row=3, column=1, sticky="w", **pad)

        ttk.Separator(params_left, orient="horizontal").grid(row=4, column=0, columnspan=4, sticky="we", padx=10, pady=8)

        ttk.Label(params_left, text="Marker filters (Genotypes)").grid(row=5, column=0, sticky="w", **pad)
        
        ttk.Label(params_left, text="Max missing per genotype (0–1):").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(params_left, textvariable=self.max_missing_genotype, width=10).grid(row=6, column=1, sticky="w", **pad)

        ttk.Label(params_left, text="Max heterozygosity per genotype (0–1):").grid(row=6, column=2, sticky="w", **pad)
        ttk.Entry(params_left, textvariable=self.max_het_genotype, width=10).grid(row=6, column=3, sticky="w", **pad)

        ttk.Separator(params_left, orient="horizontal").grid(row=7, column=0, columnspan=4, sticky="we", padx=10, pady=8)

        ttk.Label(params_left, text="Imputation (applied after filters)").grid(row=8, column=0, sticky="w", **pad)
        ttk.Label(params_left, text="Method:").grid(row=9, column=0, sticky="w", **pad)
        ttk.Combobox(
            params_left,
            textvariable=self.imputation_method,
            values=["None", "Mean", "Major allele"],
            state="readonly",
            width=18
        ).grid(row=9, column=1, sticky="w", **pad)

        # Loading indicator
        self.status_var = tk.StringVar(value="Ready.")
        self.progress = ttk.Progressbar(params_left, mode="indeterminate", length=220)
        self.progress.grid(row=10, column=0, columnspan=1, sticky="w", padx=10, pady=(0, 10))
        ttk.Label(params_left, textvariable=self.status_var).grid(
            row=10, column=1, columnspan=3, sticky="w", padx=10, pady=(0, 10)
        )

        # Run button (right)
        self.run_btn = ttk.Button(params_right, text="Run filtering", command=self.run_filtering)
        self.run_btn.pack(anchor="ne", padx=10, pady=10, ipadx=22, ipady=12)
        self.run_btn.config(width=18)

        # -----------------
        # Results
        # -----------------
        res = ttk.LabelFrame(parent, text="Results (marker table below)")
        res.pack(fill="both", expand=True, **pad)

        self.summary_text = tk.Text(res, height=12, wrap="word")
        self.summary_text.pack(fill="x", padx=10, pady=10)

        table_frame = ttk.Frame(res)
        table_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))

        cols = ("marker", "missing_rate", "het_rate", "maf", "keep")
        self.tree = ttk.Treeview(table_frame, columns=cols, show="headings")
        for c in cols:
            self.tree.heading(c, text=c)
            self.tree.column(c, width=280 if c == "marker" else 150, anchor="center")

        vsb = ttk.Scrollbar(table_frame, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=vsb.set)

        self.tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="right", fill="y")

        # -----------------
        # Bottom actions (filtering page)
        # -----------------
        bottom = ttk.Frame(parent)
        bottom.pack(fill="x", padx=10, pady=(0, 10))

        ttk.Button(bottom, text="Export marker details (CSV)", command=self.export_marker_details).pack(side="left")
        ttk.Button(bottom, text="Export genotype details (CSV)", command=self.export_genotype_details).pack(side="left", padx=10)
        ttk.Button(bottom, text="Clear", command=self.clear).pack(side="left", padx=10)

    # -----------------------------
    # Filtering worker / async
    # -----------------------------
    def _start_loading(self, msg: str = "Filtering in progress..."):
        self.status_var.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop_loading(self, msg: str = "Ready."):
        self.progress.stop()
        self.status_var.set(msg)
        self.run_btn.config(state="normal")

    def browse_input(self):
        path = filedialog.askopenfilename(
            title="Select input file",
            filetypes=[("Text/CSV/TSV/HapMap", "*.csv *.tsv *.txt *.hmp *.hapmap"), ("All files", "*.*")]
        )
        if path:
            self.input_path.set(path)
            base, ext = os.path.splitext(path)
            self.output_path.set(f"{base}_filtered{ext if ext else '.csv'}")

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save filtered matrix as",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("TSV", "*.tsv"), ("Text", "*.txt"), ("All files", "*.*")]
        )
        if path:
            self.output_path.set(path)

    def run_filtering(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        in_path = self.input_path.get().strip()
        out_path = self.output_path.get().strip()

        if not in_path or not os.path.exists(in_path):
            messagebox.showerror("Error", "Please choose a valid input file.")
            return
        if not out_path:
            messagebox.showerror("Error", "Please choose an output file path.")
            return

        maf_thr = float(self.maf_thr.get())
        max_missing_marker = float(self.max_missing_marker.get())
        max_het_marker = float(self.max_het_marker.get())
        max_missing_genotype = float(self.max_missing_genotype.get())
        max_het_genotype = float(self.max_het_genotype.get())
        imp_method = self.imputation_method.get()

        if not (0.0 <= maf_thr <= 0.5):
            messagebox.showerror("Error", "MAF threshold must be between 0 and 0.5.")
            return
        for name, v in [
            ("Max missing per marker", max_missing_marker),
            ("Max heterozygosity per marker", max_het_marker),
            ("Max missing per genotype", max_missing_genotype),
            ("Max heterozygosity per genotype", max_het_genotype),
        ]:
            if not (0.0 <= v <= 1.0):
                messagebox.showerror("Error", f"{name} must be between 0 and 1.")
                return
        if imp_method not in {"None", "Mean", "Major allele"}:
            messagebox.showerror("Error", "Unknown imputation method.")
            return

        self._start_loading("Filtering started...")

        args = dict(
            in_path=in_path,
            out_path=out_path,
            input_type=self.input_type.get(),
            maf_thr=maf_thr,
            max_missing_marker=max_missing_marker,
            max_het_marker=max_het_marker,
            max_missing_genotype=max_missing_genotype,
            max_het_genotype=max_het_genotype,
            imp_method=imp_method
        )

        self._worker_thread = threading.Thread(target=self._worker_filtering, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(120, self._poll_worker_queue)

    def _worker_filtering(self, **kwargs):
        try:
            in_path = kwargs["in_path"]
            out_path = kwargs["out_path"]
            input_type = kwargs["input_type"]
            maf_thr = kwargs["maf_thr"]
            max_missing_marker = kwargs["max_missing_marker"]
            max_het_marker = kwargs["max_het_marker"]
            max_missing_genotype = kwargs["max_missing_genotype"]
            max_het_genotype = kwargs["max_het_genotype"]
            imp_method = kwargs["imp_method"]

            if input_type == "HapMap":
                df = hapmap_to_numeric_matrix(in_path, major_hom=2, minor_hom=0, het_code=1)
            else:
                df = read_numeric_marker_matrix(in_path)

            df_m, mrk_summary, mrk_details = filter_markers(df, maf_thr, max_missing_marker, max_het_marker)
            out_df, geno_summary, geno_details = filter_genotypes(df_m, max_missing_genotype, max_het_genotype)
            out_df_imputed = impute_matrix(out_df, imp_method)

            ext = os.path.splitext(out_path)[1].lower()
            sep = "\t" if ext in [".tsv", ".txt"] else ","
            out_df_imputed.to_csv(out_path, index=False, sep=sep)

            self._result_q.put((
                "ok",
                out_df_imputed,
                mrk_summary,
                mrk_details,
                geno_summary,
                geno_details,
                out_path,
                imp_method
            ))
        except Exception as e:
            self._result_q.put(("err", str(e)))

    def _poll_worker_queue(self):
        try:
            msg = self._result_q.get_nowait()
        except queue.Empty:
            self.after(120, self._poll_worker_queue)
            return

        if msg[0] == "err":
            self._stop_loading("Ready.")
            messagebox.showerror("Filtering failed", msg[1])
            return

        _, out_df_imputed, mrk_summary, mrk_details, geno_summary, geno_details, out_path, imp_method = msg

        self.df = out_df_imputed
        self.marker_details = mrk_details
        self.genotype_details = geno_details

        self._show_summary(mrk_summary, geno_summary, out_path, imp_method)
        self._show_marker_details(mrk_details)

        self._stop_loading("Done.")

    def _show_summary(self, mrk_summary: dict, geno_summary: dict, out_path: str, imp_method: str):
        self.summary_text.delete("1.0", tk.END)
        msg = (
            f"Saved filtered matrix to:\n  {out_path}\n\n"
            f"Markers before: {mrk_summary['n_markers_before']}\n"
            f"Markers after : {mrk_summary['n_markers_after']}\n"
            f"Removed markers total : {mrk_summary['removed_total']}\n"
            f"  - removed by marker missingness  : {mrk_summary['removed_missing']}\n"
            f"  - removed by MAF                : {mrk_summary['removed_maf']}\n"
            f"  - removed by marker heterozygos.: {mrk_summary['removed_het']}\n\n"
            f"Genotypes before: {geno_summary['n_genotypes_before']}\n"
            f"Genotypes after : {geno_summary['n_genotypes_after']}\n"
            f"Removed genotypes total: {geno_summary['removed_genotypes_total']}\n"
            f"  - removed by genotype missingness: {geno_summary['removed_genotypes_missing']}\n"
            f"  - removed by genotype heterozygos.: {geno_summary['removed_genotypes_het']}\n\n"
            f"Imputation method applied after filtering: {imp_method}\n\n"
            f"Note (HapMap): first 11 columns are ignored; markers are converted per marker to 0/1/2 "
            f"(minor=0, heterozygous=1, major=2) and transposed to Taxon × Marker."
        )
        self.summary_text.insert(tk.END, msg)

    def _show_marker_details(self, details: pd.DataFrame):
        for item in self.tree.get_children():
            self.tree.delete(item)

        show_n = min(len(details), 1500)
        d = details.iloc[:show_n].copy()
        d["missing_rate"] = d["missing_rate"].round(4)
        d["het_rate"] = d["het_rate"].round(4)
        d["maf"] = d["maf"].round(4)

        for _, r in d.iterrows():
            self.tree.insert(
                "",
                "end",
                values=(r["marker"], r["missing_rate"], r["het_rate"], r["maf"], bool(r["keep"]))
            )

        if len(details) > show_n:
            self.tree.insert("", "end", values=(f"... ({len(details) - show_n} more)", "", "", "", ""))

    def export_marker_details(self):
        if self.marker_details is None:
            messagebox.showinfo("Nothing to export", "Run filtering first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save per-marker details",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("TSV", "*.tsv"), ("Text", "*.txt")]
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        self.marker_details.to_csv(path, index=False, sep=sep)
        messagebox.showinfo("Exported", f"Saved:\n{path}")

    def export_genotype_details(self):
        if self.genotype_details is None:
            messagebox.showinfo("Nothing to export", "Run filtering first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save per-genotype details",
            defaultextension=".csv",
            filetypes=[("CSV", "*.csv"), ("TSV", "*.tsv"), ("Text", "*.txt")]
        )
        if not path:
            return

        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        self.genotype_details.to_csv(path, index=False, sep=sep)
        messagebox.showinfo("Exported", f"Saved:\n{path}")

    def clear(self):
        self.df = None
        self.marker_details = None
        self.genotype_details = None
        if hasattr(self, "summary_text"):
            self.summary_text.delete("1.0", tk.END)
        if hasattr(self, "tree"):
            for item in self.tree.get_children():
                self.tree.delete(item)
        if hasattr(self, "status_var"):
            self.status_var.set("Ready.")





class SummaryPAPlotPage(ttk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app   # store reference to main app
        ...

        # state
        self._files = []
        self._file2name = {}
        self._combined_df = None

        self._fig = None
        self._canvas = None

        self._build_ui()

    def _build_ui(self):
        style = ttk.Style()
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"))
        style.configure("Sub.TLabel", font=("Segoe UI", 11))
        style.configure("Summary.TButton", font=("Segoe UI", 11), padding=(10, 6))

        # ---------- TOP (settings) ----------
        top = ttk.Frame(self)
        top.pack(side="top", fill="x", padx=20, pady=(18, 10))

        ttk.Label(top, text="Results summary", style="Title.TLabel").pack(anchor="w")
        ttk.Label(
            top,
            text="Load multiple CSVs and plot mean PA_test ± SE for each model/module.",
            style="Sub.TLabel"
        ).pack(anchor="w", pady=(4, 10))

        # controls row
        controls = ttk.Frame(top)
        controls.pack(fill="x", pady=(0, 10))

        ttk.Button(controls, text="Add CSV files", style="Summary.TButton",
                   command=self._add_csv_files).pack(side="left")
        ttk.Button(controls, text="Clear", style="Summary.TButton",
                   command=self._clear).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Plot", style="Summary.TButton",
                   command=self._plot).pack(side="left", padx=(8, 0))
        ttk.Button(controls, text="Download plot", style="Summary.TButton",
                   command=self._save_plot).pack(side="left", padx=(8, 0))

        self._status = ttk.Label(controls, text="No files loaded", style="Sub.TLabel")
        self._status.pack(side="left", padx=(14, 0))

        # file list
        ttk.Label(top, text="Loaded files:", style="Sub.TLabel").pack(anchor="w")

        list_row = ttk.Frame(top)
        list_row.pack(fill="x", pady=(6, 10))

        self._listbox = tk.Listbox(list_row, height=6)
        self._listbox.pack(side="left", fill="x", expand=True)

        sb = ttk.Scrollbar(list_row, orient="vertical", command=self._listbox.yview)
        sb.pack(side="right", fill="y")
        self._listbox.config(yscrollcommand=sb.set)

        # rename row
        rename = ttk.Frame(top)
        rename.pack(fill="x")

        ttk.Label(rename, text="Custom model name for selected file:", style="Sub.TLabel").pack(side="left")

        self._name_var = tk.StringVar()
        ttk.Entry(rename, textvariable=self._name_var, width=35).pack(side="left", padx=(10, 8))

        ttk.Button(rename, text="Set name", style="Summary.TButton",
                   command=self._set_name_for_selected).pack(side="left")

        ttk.Button(rename, text="Auto-name from filename", style="Summary.TButton",
                   command=self._autoname_selected).pack(side="left", padx=(8, 0))

        # ---------- BOTTOM (plot fills the rest) ----------
        # This is the key: plot area is a separate frame that expands,
        # while settings stay at top and never get pushed off-screen.
        self._plot_box = ttk.Frame(self)
        self._plot_box.pack(side="top", fill="both", expand=True, padx=20, pady=(0, 20))

        # Optional placeholder
        self._placeholder = ttk.Label(self._plot_box, text="No plot yet", style="Sub.TLabel")
        self._placeholder.pack(expand=True)

    # --------------------------- logic ---------------------------

    def _add_csv_files(self):
        paths = filedialog.askopenfilenames(
            title="Select one or more results CSV files",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not paths:
            return

        for p in paths:
            if p not in self._files:
                self._files.append(p)
                default = os.path.splitext(os.path.basename(p))[0]
                self._file2name[p] = default
                self._listbox.insert(tk.END, default)

        self._status.config(text=f"{len(self._files)} file(s) loaded")

    def _clear(self):
        self._files.clear()
        self._file2name.clear()
        self._combined_df = None
        self._name_var.set("")
        self._listbox.delete(0, tk.END)
        self._status.config(text="No files loaded")

        self._fig = None
        if self._canvas is not None:
            self._canvas.get_tk_widget().destroy()
            self._canvas = None

        # restore placeholder
        for w in self._plot_box.winfo_children():
            w.destroy()
        self._placeholder = ttk.Label(self._plot_box, text="No plot yet", style="Sub.TLabel")
        self._placeholder.pack(expand=True)

    def _detect_pa_col(self, df):
        df.columns = [c.strip() for c in df.columns]
        pa_candidates = ["PA_test", "PA", "PA_TEST", "pa_test", "pa"]
        pa_col = next((c for c in pa_candidates if c in df.columns), None)
        if pa_col is None:
            for c in df.columns:
                lc = c.lower()
                if "pa" in lc and "test" in lc:
                    return c
        return pa_col

    def _plot(self):
        if not self._files:
            messagebox.showwarning("No files", "Please add at least one CSV file.")
            return

        all_rows = []
        for path in self._files:
            try:
                df = pd.read_csv(path)
            except Exception as e:
                messagebox.showerror("Error", f"Could not read:\n{os.path.basename(path)}\n\n{e}")
                return

            pa_col = self._detect_pa_col(df)
            if pa_col is None:
                messagebox.showerror(
                    "Missing PA_test",
                    f"In file: {os.path.basename(path)}\n"
                    "I couldn't find 'PA_test' (or similar)."
                )
                return

            name = self._file2name.get(path, os.path.splitext(os.path.basename(path))[0])

            tmp = df[[pa_col]].copy()
            tmp.rename(columns={pa_col: "PA_test"}, inplace=True)
            tmp["Model"] = name
            tmp["PA_test"] = pd.to_numeric(tmp["PA_test"], errors="coerce")
            tmp = tmp.dropna(subset=["PA_test"])

            if not tmp.empty:
                all_rows.append(tmp)

        if not all_rows:
            messagebox.showerror("No data", "No valid PA_test values found across files.")
            return

        combined = pd.concat(all_rows, ignore_index=True)
        self._combined_df = combined

        stats = (combined.groupby("Model", as_index=False)
                        .agg(n=("PA_test", "count"),
                             mean=("PA_test", "mean"),
                             sd=("PA_test", "std")))
        stats["se"] = stats["sd"] / np.sqrt(stats["n"])
        stats = stats.sort_values("mean", ascending=False)

        self._draw_barplot(stats)

    def _draw_barplot(self, stats_df):
        # clear plot area
        for w in self._plot_box.winfo_children():
            w.destroy()

        fig = plt.Figure(figsize=(11, 5.5), dpi=110)
        ax = fig.add_subplot(111)

        x = np.arange(len(stats_df))
        y = stats_df["mean"].values
        yerr = stats_df["se"].fillna(0.0).values  # if n=1, SE is NaN -> show 0

        ax.bar(x, y, yerr=yerr, capsize=6)
        ax.set_title("Mean PA_test ± SE across uploaded CSV files")
        ax.set_ylabel("PA_test (Pearson correlation)")
        ax.set_xlabel("Model / Module")
        ax.set_xticks(x)
        ax.set_xticklabels(stats_df["Model"].astype(str).values, rotation=35, ha="right")
        ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
        fig.tight_layout()

        self._fig = fig
        self._canvas = FigureCanvasTkAgg(fig, master=self._plot_box)
        self._canvas.draw()
        self._canvas.get_tk_widget().pack(fill="both", expand=True)

        self._status.config(text=f"Plotted {len(stats_df)} model(s) from {len(self._files)} file(s)")

    def _save_plot(self):
        if self._fig is None:
            messagebox.showwarning("No plot", "Please generate the plot first.")
            return

        path = filedialog.asksaveasfilename(
            title="Save plot",
            defaultextension=".png",
            filetypes=[
                ("PNG image", "*.png"),
                ("PDF document", "*.pdf"),
                ("SVG vector", "*.svg"),
                ("JPEG image", "*.jpg"),
                ("All files", "*.*")
            ]
        )
        if not path:
            return

        try:
            if path.lower().endswith(".png"):
                self._fig.savefig(path, dpi=300, bbox_inches="tight")
            else:
                self._fig.savefig(path, bbox_inches="tight")

            messagebox.showinfo("Saved", f"Plot saved to:\n{path}")
        except Exception as e:
            messagebox.showerror("Save error", f"Could not save plot:\n{e}")

    def _set_name_for_selected(self):
        sel = self._listbox.curselection()
        if not sel:
            messagebox.showwarning("Select a file", "Please select a file in the list.")
            return

        new_name = self._name_var.get().strip()
        if not new_name:
            messagebox.showwarning("Empty name", "Please type a name first.")
            return

        idx = sel[0]
        path = self._files[idx]
        self._file2name[path] = new_name

        self._listbox.delete(idx)
        self._listbox.insert(idx, new_name)
        self._listbox.selection_set(idx)

    def _autoname_selected(self):
        sel = self._listbox.curselection()
        if not sel:
            messagebox.showwarning("Select a file", "Please select a file in the list.")
            return

        idx = sel[0]
        path = self._files[idx]
        default = os.path.splitext(os.path.basename(path))[0]
        self._file2name[path] = default

        self._listbox.delete(idx)
        self._listbox.insert(idx, default)
        self._listbox.selection_set(idx)













# ==========================================================
# Nested CV Window (SVR)
# ==========================================================
class NestedSvrCvFrame(ttk.Frame):

    """
    True nested CV:
      - Outer CV (K folds) repeated for N cycles (different random_state per cycle)
      - Inner CV tunes hyperparameters on outer-train split
      - StandardScaler fit on outer-train only
      - Optionally log-transform y with log1p
    """

    def __init__(self, parent):
        super().__init__(parent)

        # Files
        self.marker_path = tk.StringVar(value="")
        self.pheno_path = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_SVR_results"))

        # Trait / transform
        self.target_col = tk.StringVar(value="")
        self.use_log1p = tk.BooleanVar(value=True)

        # Hyperparameters
        self.kernel_list = tk.StringVar(value="rbf, linear, poly,sigmoid")
        self.C_list = tk.StringVar(value="0.01,0.1,1,10")
        self.eps_list = tk.StringVar(value="0.05,0.1,0.2")
        self.use_gamma = tk.BooleanVar(value=False)
        self.gamma_list = tk.StringVar(value="scale, auto")

        # CV settings
        self.n_cycles = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed = tk.IntVar(value=1000)
        self.inner_seed = tk.IntVar(value=2000)

        # Tuning criterion
        self.tune_by = tk.StringVar(value="MAPE")

        # Threading
        self._worker_thread = None
        self._q = queue.Queue()

        self._build_cv_ui()

    def _build_cv_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="SVR hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="Kernel list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.kernel_list, width=40).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(hp, text="C list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.C_list, width=40).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(hp, text="epsilon list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.eps_list, width=40).grid(row=2, column=1, sticky="w", **pad)

        ttk.Checkbutton(hp, text="Tune gamma", variable=self.use_gamma).grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.gamma_list, width=40).grid(row=3, column=1, sticky="w", **pad)

        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:

            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)

        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")

        # assume first column is ID
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)

        # numeric conversion where possible
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.01,0.1,1,10")

    def _parse_gamma_list(self, s: str):
        s = s.strip()
        if s in {"scale", "auto"}:
            return [s]
        # allow list
        return self._parse_float_list(s, "gamma list")
    

    def _parse_kernel_list(self, s: str):
        allowed = {"rbf", "linear", "poly", "sigmoid"}
        vals = [x.strip().lower() for x in s.split(",") if x.strip() != ""]
        if not vals:
            raise ValueError("Kernel list is empty. Example: rbf,linear")
        bad = [k for k in vals if k not in allowed]
        if bad:
            raise ValueError(f"Invalid kernel(s): {bad}. Allowed: rbf, linear, poly, sigmoid")
        return vals


    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            C_vals = self._parse_float_list(self.C_list.get(), "C list")
            eps_vals = self._parse_float_list(self.eps_list.get(), "epsilon list")
            gamma_vals = self._parse_gamma_list(self.gamma_list.get()) if self.use_gamma.get() else [None]
            kernel_vals = self._parse_kernel_list(self.kernel_list.get())
        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return

        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            kernel_vals=kernel_vals,
            C_vals=C_vals,
            eps_vals=eps_vals,
            gamma_vals=gamma_vals,
            use_gamma=self.use_gamma.get(),
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (SVR) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Kernel={args['kernel_vals']}"),
        self._log(f"C={args['C_vals']}")
        self._log(f"epsilon={args['eps_vals']}")
        if args["use_gamma"]:
            self._log(f"gamma={args['gamma_vals']}")


        self._start_time = time.perf_counter()
        self._start("Running nested CV (SVR)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            # align indices
            markers.index = markers.index.astype(str)
            pheno.index = pheno.index.astype(str)

            # remove duplicated IDs in markers
            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            # hyperparameter grid (match your script: C, epsilon, kernel; optional gamma)
            if args["use_gamma"]:
                grid = list(product(args["C_vals"], args["eps_vals"], args["kernel_vals"], args["gamma_vals"]))
            else:
                grid = list(product(args["C_vals"], args["eps_vals"], args["kernel_vals"]))

            all_rows = []

            # ----- cycles -----
            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                # ----- outer folds -----
                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    # scale within outer fold ONLY
                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []

                    # ----- inner tuning -----
                    if args["use_gamma"]:
                        for C, epsilon, kernel, gamma in grid:
                            fold_pas, fold_mspes, fold_mapes = [], [], []
                            for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                                Xtr = X_train_outer_s[itr_idx]
                                Xva = X_train_outer_s[iva_idx]
                                ytr = y_train_outer[itr_idx]
                                yva = y_train_outer[iva_idx]

                                svr = SVR(C=C, epsilon=epsilon, kernel=kernel, gamma=gamma)
                                svr.fit(Xtr, ytr)
                                pred = svr.predict(Xva)

                                try:
                                    pa = pearsonr(yva, pred)[0]
                                except Exception:
                                    pa = np.nan
                                mspe = mean_squared_error(yva, pred)
                                mape = mean_absolute_error(yva, pred)

                                fold_pas.append(pa)
                                fold_mspes.append(mspe)
                                fold_mapes.append(mape)

                            tuning_rows.append({
                                "C": C, "epsilon": epsilon, "kernel": kernel, "gamma": gamma,
                                "PA": float(np.nanmean(fold_pas)),
                                "MSPE": float(np.mean(fold_mspes)),
                                "MAPE": float(np.mean(fold_mapes))
                            })
                    else:
                        for C, epsilon, kernel in grid:
                            fold_pas, fold_mspes, fold_mapes = [], [], []
                            for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                                Xtr = X_train_outer_s[itr_idx]
                                Xva = X_train_outer_s[iva_idx]
                                ytr = y_train_outer[itr_idx]
                                yva = y_train_outer[iva_idx]

                                svr = SVR(C=C, epsilon=epsilon, kernel=kernel)
                                svr.fit(Xtr, ytr)
                                pred = svr.predict(Xva)

                                try:
                                    pa = pearsonr(yva, pred)[0]
                                except Exception:
                                    pa = np.nan
                                mspe = mean_squared_error(yva, pred)
                                mape = mean_absolute_error(yva, pred)

                                fold_pas.append(pa)
                                fold_mspes.append(mspe)
                                fold_mapes.append(mape)

                            tuning_rows.append({
                                "C": C, "epsilon": epsilon, "kernel": kernel,
                                "PA": float(np.nanmean(fold_pas)),
                                "MSPE": float(np.mean(fold_mspes)),
                                "MAPE": float(np.mean(fold_mapes))
                            })

                    tuning_df = pd.DataFrame(tuning_rows)

                    # select best
                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:  # PA
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    # train final on outer train, evaluate on outer test
                    if args["use_gamma"]:
                        final = SVR(
                            C=float(best["C"]),
                            epsilon=float(best["epsilon"]),
                            kernel=str(best["kernel"]),
                            gamma=best["gamma"]
                        )
                    else:
                        final = SVR(
                            C=float(best["C"]),
                            epsilon=float(best["epsilon"]),
                            kernel=str(best["kernel"])
                        )

                    final.fit(X_train_outer_s, y_train_outer)
                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    # metrics
                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    row = {
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_C": float(best["C"]),
                        "Best_epsilon": float(best["epsilon"]),
                        "Best_kernel": str(best["kernel"]),
                    }
                    if args["use_gamma"]:
                        row["Best_gamma"] = best["gamma"]
                    all_rows.append(row)

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]

           
            
            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")

           
            out_path = os.path.join(outdir, f"SVR_nestedCV_results{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV failed (SVR)", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (SVR) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))











# ==========================================================
# Nested CV Window (ridge)
# ==========================================================

class NestedRidgeCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_ridge_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        self.alpha_list  = tk.StringVar(value="0.001,0.01,0.1,1,10")

        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="Ridge hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="alpha list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.alpha_list, width=60).grid(row=0, column=1, sticky="w", **pad)

        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (Ridge)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")

    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            alpha_vals = self._parse_float_list(self.alpha_list.get(), "alpha list")
        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return

        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            alpha_vals=alpha_vals,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (Ridge) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"alpha={args['alpha_vals']}")
        self._log(f"tune_by={args['tune_by']}")

        self._start_time = time.perf_counter()
        self._start("Running nested CV (Ridge)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []
                    for alpha in args["alpha_vals"]:
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = Ridge(alpha=float(alpha))
                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan
                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "alpha": float(alpha),
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = Ridge(alpha=float(best["alpha"]))
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_alpha": float(best["alpha"]),
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]

            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")



            out_path = os.path.join(outdir, f"Ridge_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (Ridge) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (ridge) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))













# ==========================================================
# Nested CV Window (lasso)
# ==========================================================

class NestedLassoCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_Lasso_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        self.alpha_list  = tk.StringVar(value="0.001,0.01,0.1,1,10")

        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="Lasso hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="alpha list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.alpha_list, width=60).grid(row=0, column=1, sticky="w", **pad)

        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (Lasso)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")

    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            alpha_vals = self._parse_float_list(self.alpha_list.get(), "alpha list")
        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return

        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            alpha_vals=alpha_vals,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (Lasso) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"alpha={args['alpha_vals']}")
        self._log(f"tune_by={args['tune_by']}")

        self._start_time = time.perf_counter()
        self._start("Running nested CV (Lasso)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []
                    for alpha in args["alpha_vals"]:
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = Lasso(alpha=float(alpha))
                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan
                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "alpha": float(alpha),
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = Lasso(alpha=float(best["alpha"]))
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_alpha": float(best["alpha"]),
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]


            
            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")


            out_path = os.path.join(outdir, f"Lasso_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (Lasso) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (Lasso) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))






















# ==========================================================
# Nested CV Window (ElasticNet)
# ==========================================================
class NestedElasticNetCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_ElasticNet_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        # Hyperparameters as STRING LISTS (comma-separated)
        self.alpha_list         = tk.StringVar(value="0.001,0.01,0.1,1,10")
        self.l1_ratio_list      = tk.StringVar(value="0.1,0.5,0.9")
        self.fit_intercept_list = tk.StringVar(value="True,False")
        self.precompute_list    = tk.StringVar(value="True,False")
        self.max_iter_list      = tk.StringVar(value="5000")
        self.tol_list           = tk.StringVar(value="1e-6,1e-4")
        self.warm_start_list    = tk.StringVar(value="True,False")
        self.positive_list      = tk.StringVar(value="True,False")
        self.selection_list     = tk.StringVar(value="cyclic,random")

        # CV settings
        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="ElasticNet hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="alpha list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.alpha_list, width=60).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(hp, text="l1_ratio list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.l1_ratio_list, width=60).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(hp, text="fit_intercept list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.fit_intercept_list, width=60).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(hp, text="precompute list:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.precompute_list, width=60).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(hp, text="max_iter list:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_iter_list, width=60).grid(row=4, column=1, sticky="w", **pad)

        ttk.Label(hp, text="tol list:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.tol_list, width=60).grid(row=5, column=1, sticky="w", **pad)

        ttk.Label(hp, text="warm_start list:").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.warm_start_list, width=60).grid(row=6, column=1, sticky="w", **pad)

        ttk.Label(hp, text="positive list:").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.positive_list, width=60).grid(row=7, column=1, sticky="w", **pad)

        ttk.Label(hp, text="selection list:").grid(row=8, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.selection_list, width=60).grid(row=8, column=1, sticky="w", **pad)

        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (ElasticNet)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")

    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 1000,5000,10000")

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")

    def _parse_precompute_list(self, s: str):
        # ElasticNet precompute can be bool or array-like; for GUI keep it bool
        return self._parse_bool_list(s, "precompute_list")

    def _parse_selection_list(self, s: str):
        vals = [x.strip().lower() for x in s.split(",") if x.strip() != ""]
        allowed = {"cyclic", "random"}
        bad = [v for v in vals if v not in allowed]
        if bad:
            raise ValueError(f"Invalid selection_list values: {bad}. Allowed: cyclic,random")
        return vals

    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            alpha_vals      = self._parse_float_list(self.alpha_list.get(), "alpha_list")
            l1_vals         = self._parse_float_list(self.l1_ratio_list.get(), "l1_ratio_list")
            fit_vals        = self._parse_bool_list(self.fit_intercept_list.get(), "fit_intercept_list")
            pre_vals        = self._parse_precompute_list(self.precompute_list.get())
            iter_vals       = self._parse_int_list(self.max_iter_list.get(), "max_iter_list")
            tol_vals        = self._parse_float_list(self.tol_list.get(), "tol_list")
            warm_vals       = self._parse_bool_list(self.warm_start_list.get(), "warm_start_list")
            pos_vals        = self._parse_bool_list(self.positive_list.get(), "positive_list")
            sel_vals        = self._parse_selection_list(self.selection_list.get())

            grid = list(product(alpha_vals, l1_vals, fit_vals, pre_vals, iter_vals, tol_vals, warm_vals, pos_vals, sel_vals))
        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return

        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (ElasticNet) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"Tune by: {args['tune_by']}")
        self._start_time = time.perf_counter()

        self._start("Running nested CV (ElasticNet)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []

                    for (alpha, l1_ratio, fit_intercept, precompute, max_iter, tol, warm_start, positive, selection) in grid:
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = ElasticNet(
                                alpha=float(alpha),
                                l1_ratio=float(l1_ratio),
                                fit_intercept=bool(fit_intercept),
                                precompute=bool(precompute),
                                max_iter=int(max_iter),
                                tol=float(tol),
                                warm_start=bool(warm_start),
                                positive=bool(positive),
                                selection=str(selection),
                                random_state=0
                            )
                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan

                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "alpha": float(alpha),
                            "l1_ratio": float(l1_ratio),
                            "fit_intercept": bool(fit_intercept),
                            "precompute": bool(precompute),
                            "max_iter": int(max_iter),
                            "tol": float(tol),
                            "warm_start": bool(warm_start),
                            "positive": bool(positive),
                            "selection": str(selection),
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = ElasticNet(
                        alpha=float(best["alpha"]),
                        l1_ratio=float(best["l1_ratio"]),
                        fit_intercept=bool(best["fit_intercept"]),
                        precompute=bool(best["precompute"]),
                        max_iter=int(best["max_iter"]),
                        tol=float(best["tol"]),
                        warm_start=bool(best["warm_start"]),
                        positive=bool(best["positive"]),
                        selection=str(best["selection"]),
                        random_state=0
                    )
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_alpha": float(best["alpha"]),
                        "Best_l1_ratio": float(best["l1_ratio"]),
                        "Best_fit_intercept": bool(best["fit_intercept"]),
                        "Best_precompute": bool(best["precompute"]),
                        "Best_max_iter": int(best["max_iter"]),
                        "Best_tol": float(best["tol"]),
                        "Best_warm_start": bool(best["warm_start"]),
                        "Best_positive": bool(best["positive"]),
                        "Best_selection": str(best["selection"]),
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]


            
            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")


            out_path = os.path.join(outdir, f"ElasticNet_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (ElasticNet) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (ElasticNet) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))


































# ==========================================================
# Nested CV Window (BayesianRidge)
# ==========================================================
class NestedBayesianRidgeCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_BayesianRidge_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        # Hyperparameters as STRING LISTS (comma-separated)
        self.alpha_1_list         = tk.StringVar(value="0.00001,0.0001,0.001,0.01,0.1")
        self.alpha_2_list         = tk.StringVar(value="0.00001,0.0001,0.001,0.01,0.1")
        self.lambda_1_list         = tk.StringVar(value="0.00001,0.0001,0.001,0.01,0.1")
        self.lambda_2_list         = tk.StringVar(value="0.00001,0.0001,0.001,0.01,0.1")
        self.max_iter_list      = tk.StringVar(value="5000")
        self.fit_intercept_list = tk.StringVar(value="True,False")
        self.tol_list           = tk.StringVar(value="1e-6,1e-4,1e-2")
        

        # CV settings
        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="BayesianRidge hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="alpha_1 list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.alpha_1_list, width=60).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(hp, text="alpha_1 list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.alpha_2_list, width=60).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(hp, text="lambda_1 list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.lambda_1_list, width=60).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(hp, text="lambda_2 list:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.lambda_2_list, width=60).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(hp, text="max_iter list:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_iter_list, width=60).grid(row=4, column=1, sticky="w", **pad)

        ttk.Label(hp, text="fit_intercept list:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.fit_intercept_list, width=60).grid(row=5, column=1, sticky="w", **pad)

        ttk.Label(hp, text="tol list:").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.tol_list, width=60).grid(row=6, column=1, sticky="w", **pad)

        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (BayesianRidge)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")

    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 1000,5000,10000")

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")


    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            alpha_1_vals      = self._parse_float_list(self.alpha_1_list.get(), "alpha_1_list")
            alpha_2_vals         = self._parse_float_list(self.alpha_2_list.get(), "alpha_1_list")
            lambda_1_vals        = self._parse_float_list(self.lambda_1_list.get(), "lambda_1_list")
            lambda_2_vals        = self._parse_float_list(self.lambda_2_list.get(), "lambda_2_list")
            iter_vals       = self._parse_int_list(self.max_iter_list.get(), "max_iter_list")
            fit_intercept_vals        = self._parse_bool_list(self.fit_intercept_list.get(), "tol_list")
            tol_vals        = self._parse_float_list(self.tol_list.get(), "tol_list")
         
            grid = list(product(alpha_1_vals, alpha_2_vals, lambda_1_vals, lambda_2_vals, iter_vals, fit_intercept_vals, tol_vals))
        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return

        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (BayesianRidge) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"Tune by: {args['tune_by']}")
        self._start_time = time.perf_counter()
        self._start("Running nested CV (BayesianRidge)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []

                    for (alpha_1, alpha_2, lambda_1, lambda_2, max_iter, fit_intercept, tol) in grid:
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = BayesianRidge(
                                alpha_1=float(alpha_1),
                                alpha_2=float(alpha_2),
                                fit_intercept=bool(fit_intercept),
                                lambda_1=float(lambda_1),
                                max_iter=int(max_iter),
                                tol=float(tol),
                                lambda_2=float(lambda_2)
                            )
                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan

                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "alpha_1": float(alpha_1),
                            "alpha_2": float(alpha_2),
                            "lambda_1": bool(lambda_1),
                            "lambda_2": bool(lambda_2),
                            "fit_intercept": bool(fit_intercept),
                            "max_iter": int(max_iter),
                            "tol": float(tol),
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = BayesianRidge(
                        alpha_1=float(best["alpha_1"]),
                        alpha_2=float(best["alpha_2"]),
                        lambda_1=bool(best["lambda_1"]),
                        lambda_2=bool(best["lambda_2"]),
                        fit_intercept=bool(best["fit_intercept"]),
                        max_iter=int(best["max_iter"]),
                        tol=float(best["tol"])
                    )
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_alpha_1": float(best["alpha_1"]),
                        "Best_alpha_2": float(best["alpha_2"]),
                        "Best_lambda_1": bool(best["lambda_1"]),
                        "Best_lambda_2": bool(best["lambda_2"]),
                        "Best_fit_intercept": bool(best["fit_intercept"]),
                        "Best_max_iter": int(best["max_iter"]),
                        "Best_tol": float(best["tol"]),
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]

          
          
            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")

          
          
            out_path = os.path.join(outdir, f"BayesianRidge_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (BayesianRidge) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (BayesianRidge) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))

























# ==========================================================
# Nested CV Window (KernelRidge)
# ==========================================================
class NestedKernelRidgeCvFrame(ttk.Frame):

    """
    True nested CV:
      - Outer CV (K folds) repeated for N cycles (different random_state per cycle)
      - Inner CV tunes hyperparameters on outer-train split
      - StandardScaler fit on outer-train only
      - Optionally log-transform y with log1p
    """

    def __init__(self, parent):
        super().__init__(parent)

        # Files
        self.marker_path = tk.StringVar(value="")
        self.pheno_path = tk.StringVar(value="")
        self.output_dir = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_KernelRidge_results"))

        # Trait / transform
        self.target_col = tk.StringVar(value="")
        self.use_log1p = tk.BooleanVar(value=True)

        # Hyperparameters
        self.alpha_list = tk.StringVar(value="0.01,0.1,1,10")
        self.kernel_list = tk.StringVar(value="rbf, linear, poly,sigmoid")
        self.use_gamma = tk.BooleanVar(value=False)
        self.gamma_list = tk.StringVar(value="scale, auto")
        self.degree_list = tk.StringVar(value="2,3")
        self.coef_list = tk.StringVar(value="0.1,1,10")

        # CV settings
        self.n_cycles = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed = tk.IntVar(value=1000)
        self.inner_seed = tk.IntVar(value=2000)

        # Tuning criterion
        self.tune_by = tk.StringVar(value="MAPE")

        # Threading
        self._worker_thread = None
        self._q = queue.Queue()

        self._build_cv_ui()

    def _build_cv_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="Kernel Ridge hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="Kernel list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.kernel_list, width=40).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Alpha list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.alpha_list, width=40).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(hp, text="coef0 list").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.coef_list, width=40).grid(row=2, column=1, sticky="w", **pad)

        ttk.Checkbutton(hp, text="Tune gamma", variable=self.use_gamma).grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.gamma_list, width=40).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(hp, text="degree list (for 'poly' kernel):").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.degree_list, width=40).grid(row=4, column=1, sticky="w", **pad)

        
        
        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:

            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)

        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")

        # assume first column is ID
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)

        # numeric conversion where possible
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.01,0.1,1,10")

    def _parse_gamma_list(self, s: str):
        s = s.strip()
        if s in {"scale", "auto"}:
            return [s]
        # allow list
        return self._parse_float_list(s, "gamma list")
    

    def _parse_kernel_list(self, s: str):
        allowed = {"rbf", "linear", "poly", "sigmoid"}
        vals = [x.strip().lower() for x in s.split(",") if x.strip() != ""]
        if not vals:
            raise ValueError("Kernel list is empty. Example: rbf,linear")
        bad = [k for k in vals if k not in allowed]
        if bad:
            raise ValueError(f"Invalid kernel(s): {bad}. Allowed: rbf, linear, poly, sigmoid")
        return vals


    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            alpha_vals = self._parse_float_list(self.alpha_list.get(), "alpha_list")
            gamma_vals = self._parse_gamma_list(self.gamma_list.get()) if self.use_gamma.get() else [None]
            kernel_vals = self._parse_kernel_list(self.kernel_list.get())
            degree_vals = self._parse_float_list(self.degree_list.get(), "degree_list")
            coef_vals   = self._parse_float_list(self.coef_list.get(), "coef_list")

        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return

        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            kernel_vals=kernel_vals,    
            gamma_vals=gamma_vals,
            degree_vals=degree_vals,
            coef_vals=coef_vals,
            alpha_vals=alpha_vals,
            use_gamma=self.use_gamma.get(),
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (Kernel Ridge) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Kernel={args['kernel_vals']}"),
        self._log(f"Alpha={args['alpha_vals']}")
        self._log(f"Degree={args['degree_vals']}")
        self._log(f"Coef0={args['coef_vals']}")
        if args["use_gamma"]:
            self._log(f"gamma={args['gamma_vals']}")

        self._start_time = time.perf_counter()
        self._start("Running nested CV...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            # align indices
            markers.index = markers.index.astype(str)
            pheno.index = pheno.index.astype(str)

            # remove duplicated IDs in markers
            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            # hyperparameter grid (match your script: C, epsilon, kernel; optional gamma)
            if args["use_gamma"]:
                grid = list(product(args["alpha_vals"], args["kernel_vals"], args["gamma_vals"],args["degree_vals"], args["coef_vals"]))
            else:
                grid = list(product(args["alpha_vals"], args["kernel_vals"], args["degree_vals"], args["coef_vals"]))

            all_rows = []

            # ----- cycles -----
            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                # ----- outer folds -----
                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    # scale within outer fold ONLY
                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []

                    # ----- inner tuning -----
                    if args["use_gamma"]:
                        for alpha, kernel, gamma, degree, coef in grid:
                            fold_pas, fold_mspes, fold_mapes = [], [], []
                            for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                                Xtr = X_train_outer_s[itr_idx]
                                Xva = X_train_outer_s[iva_idx]
                                ytr = y_train_outer[itr_idx]
                                yva = y_train_outer[iva_idx]

                                KR = KernelRidge(alpha=alpha, kernel=kernel, gamma=gamma, degree=int(degree), coef0=coef)
                                KR.fit(Xtr, ytr)
                                pred = KR.predict(Xva)

                                try:
                                    pa = pearsonr(yva, pred)[0]
                                except Exception:
                                    pa = np.nan
                                mspe = mean_squared_error(yva, pred)
                                mape = mean_absolute_error(yva, pred)

                                fold_pas.append(pa)
                                fold_mspes.append(mspe)
                                fold_mapes.append(mape)

                            tuning_rows.append({
                                "alpha": alpha, "kernel": kernel, "gamma": gamma, "degree": degree, "coef0": coef,
                                "PA": float(np.nanmean(fold_pas)),
                                "MSPE": float(np.mean(fold_mspes)),
                                "MAPE": float(np.mean(fold_mapes))
                            })
                    else:
                        for alpha, kernel, degree, coef in grid:
                            fold_pas, fold_mspes, fold_mapes = [], [], []
                            for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                                Xtr = X_train_outer_s[itr_idx]
                                Xva = X_train_outer_s[iva_idx]
                                ytr = y_train_outer[itr_idx]
                                yva = y_train_outer[iva_idx]

                                KR = KernelRidge(alpha=alpha, kernel=kernel, degree=int(degree), coef0=coef)
                                KR.fit(Xtr, ytr)
                                pred = KR.predict(Xva)

                                try:
                                    pa = pearsonr(yva, pred)[0]
                                except Exception:
                                    pa = np.nan
                                mspe = mean_squared_error(yva, pred)
                                mape = mean_absolute_error(yva, pred)

                                fold_pas.append(pa)
                                fold_mspes.append(mspe)
                                fold_mapes.append(mape)

                            tuning_rows.append({
                                "kernel": kernel, "alpha": alpha, "degree": degree, "coef0": coef,
                                "PA": float(np.nanmean(fold_pas)),
                                "MSPE": float(np.mean(fold_mspes)),
                                "MAPE": float(np.mean(fold_mapes))
                            })

                    tuning_df = pd.DataFrame(tuning_rows)

                    # select best
                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:  # PA
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    # train final on outer train, evaluate on outer test
                    if args["use_gamma"]:
                        final = KernelRidge(
                            alpha=float(best["alpha"]),
                            degree=int(best["degree"]),
                            coef0=float(best["coef0"]),
                            kernel=str(best["kernel"]),
                            gamma=best["gamma"]
                        )
                    else:
                        final = KernelRidge(
                            alpha=float(best["alpha"]),
                            degree=int(best["degree"]),
                            coef0=float(best["coef0"]),
                            kernel=str(best["kernel"]),
                        )

                    final.fit(X_train_outer_s, y_train_outer)
                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    # metrics
                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    row = {
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_alpha": float(best["alpha"]),
                        "Best_degree": int(best["degree"]),
                        "Best_coef0": float(best["coef0"]),
                        "Best_kernel": str(best["kernel"]),
                    }
                    if args["use_gamma"]:
                        row["Best_gamma"] = best["gamma"]
                    all_rows.append(row)

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]


        
            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")



            out_path = os.path.join(outdir, f"KernelRidge_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (KernelRidge) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (KernelRidge) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))


































# ==========================================================
# Nested CV Window (SGDRegressor)
# ==========================================================
class NestedSGDRegressorCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_SGDRegressor_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        # Hyperparameters as STRING LISTS (comma-separated)
        self.loss_list          = tk.StringVar(value="squared_error, huber, epsilon_insensitive, squared_epsilon_insensitive")
        self.penalty_list       = tk.StringVar(value="none, l2, l1, elasticnet")
        self.alpha_list         = tk.StringVar(value="1e-6, 1e-5, 1e-4, 1e-3")
        self.l1_ratio_list      = tk.StringVar(value="0.15, 0.5, 0.85")
        self.max_iter_list      = tk.StringVar(value="1000, 2000, 5000")
        self.tol_list           = tk.StringVar(value="1e-3, 1e-4, 1e-5")
        self.fit_intercept_list = tk.StringVar(value="True, False")
        self.shuffle_list       = tk.StringVar(value="True, False")
        self.epsilon_list      = tk.StringVar(value="0.1, 0.2, 0.5")
        self.learning_rate_list = tk.StringVar(value="constant, optimal, invscaling, adaptive")
        self.eta0_list          = tk.StringVar(value=" 1e-4, 1e-3, 1e-2")
        self.power_t_list      = tk.StringVar(value="0.25, 0.5, 0.75")
        self.early_stopping_list = tk.StringVar(value="True, False")
        self.validation_fraction_list = tk.StringVar(value="0.1, 0.3, 0.5")
        self.n_iter_no_change_list = tk.StringVar(value="5, 10, 15")
        self.warm_start_list    = tk.StringVar(value="True, False")
        self.average_list       = tk.StringVar(value="True, False")

        # CV settings
        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="SGDRegressor hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="Loss list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.loss_list, width=40).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Penalty list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.penalty_list, width=40).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Alpha list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.alpha_list, width=40).grid(row= 2, column=1, sticky="w", **pad)

        ttk.Label(hp, text="l1_ratio list:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.l1_ratio_list, width=40).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Max iter list:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_iter_list, width=40).grid(row=4, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Tol list:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.tol_list, width=40).grid(row=5, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Fit intercept list:").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.fit_intercept_list, width=40).grid(row=6, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Shuffle list:").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.shuffle_list, width=40).grid(row=7, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Epsilon list:").grid(row=8, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.epsilon_list, width=40).grid(row=8, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Learning rate list:").grid(row=9, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.learning_rate_list, width=40).grid(row=9, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Eta0 list:").grid(row=10, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.eta0_list, width=40).grid(row=10, column=1, sticky="w", **pad)  

        ttk.Label(hp, text="Power t list:").grid(row=11, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.power_t_list, width=40).grid(row=11, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Early stopping list:").grid(row=12, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.early_stopping_list, width=40).grid(row=12, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Validation fraction list:").grid(row=13, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.validation_fraction_list, width=40).grid(row=13, column=1, sticky="w", **pad)

        ttk.Label(hp, text="n_iter_no_change list:").grid(row=14, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.n_iter_no_change_list, width=40).grid(row=14, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Warm start list:").grid(row=15, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.warm_start_list, width=40).grid(row=15, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Average list:").grid(row=16, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.average_list, width=40).grid(row=16, column=1, sticky="w", **pad)


        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (SGDRegressor)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")

    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 1000,5000,10000")

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")


    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            loss_vals          = [x.strip() for x in self.loss_list.get().split(",") if x.strip() != ""]
            penalty_vals       = [x.strip() for x in self.penalty_list.get().split(",") if x.strip() != ""]
            alpha_vals        = self._parse_float_list(self.alpha_list.get(), "alpha_list")
            l1_ratio_vals     = self._parse_float_list(self.l1_ratio_list.get(), "l1_ratio_list")
            fit_intercept_vals = self._parse_bool_list(self.fit_intercept_list.get(), "fit_intercept_list")
            max_iter_vals     = self._parse_int_list(self.max_iter_list.get(), "max_iter_list")
            tol_vals          = self._parse_float_list(self.tol_list.get(), "tol_list")
            shuffle_vals      = self._parse_bool_list(self.shuffle_list.get(), "shuffle_list")
            epsilon_vals      = self._parse_float_list(self.epsilon_list.get(), "epsilon_list")
            learning_rate_vals = [x.strip() for x in self.learning_rate_list.get().split(",") if x.strip() != ""]
            eta0_vals         = self._parse_float_list(self.eta0_list.get(), "eta0_list")
            power_t_vals      = self._parse_float_list(self.power_t_list.get(), "power_t_list")
            early_stopping_vals = self._parse_bool_list(self.early_stopping_list.get(), "early_stopping_list")
            validation_fraction_vals = self._parse_float_list(self.validation_fraction_list.get(), "validation_fraction_list")
            n_iter_no_change_vals = self._parse_int_list(self.n_iter_no_change_list.get(), "n_iter_no_change_list")
            warm_start_vals   = self._parse_bool_list(self.warm_start_list.get(), "warm_start_list")
            average_vals      = self._parse_bool_list(self.average_list.get(), "average_list")
         
            grid = list(product(loss_vals, penalty_vals, alpha_vals, l1_ratio_vals, fit_intercept_vals, max_iter_vals,
                                tol_vals, shuffle_vals, epsilon_vals, learning_rate_vals,
                                eta0_vals, power_t_vals, early_stopping_vals, validation_fraction_vals, n_iter_no_change_vals,
                                warm_start_vals, average_vals))
        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return

        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (SGDRegressor) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"Tune by: {args['tune_by']}")
        self._start_time = time.perf_counter()
        self._start("Running nested CV (SGDRegressor)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []

                    for (loss, penalty, alpha, L1_ratio, fit_intercept, max_iter, tol, shuffle, epsilon, learning_rate,
                        eta0, power_t, early_stopping, validation_fraction, n_inter_no_changes, warm_start, average) in grid:
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = SGDRegressor(
                                loss=loss,
                                penalty=penalty,
                                alpha=alpha,
                                l1_ratio=L1_ratio,
                                fit_intercept=fit_intercept,
                                max_iter=max_iter,
                                tol=tol,
                                shuffle=shuffle,
                                epsilon=epsilon,
                                learning_rate=learning_rate,
                                eta0=eta0,
                                power_t=power_t,
                                early_stopping=early_stopping,
                                validation_fraction=validation_fraction,
                                n_iter_no_change=n_inter_no_changes,
                                warm_start=warm_start,
                                average=average,
                            )
                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan

                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "loss": loss,
                            "penalty": penalty,
                            "alpha": alpha,
                            "l1_ratio": L1_ratio,
                            "fit_intercept": fit_intercept,
                            "max_iter": max_iter,
                            "tol": tol,
                            "shuffle": shuffle,
                            "epsilon": epsilon,
                            "learning_rate": learning_rate,
                            "eta0": eta0,
                            "power_t": power_t,
                            "early_stopping": early_stopping,
                            "validation_fraction": validation_fraction,
                            "n_iter_no_change": n_inter_no_changes,
                            "warm_start": warm_start,
                            "average": average,
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = SGDRegressor(
                        loss=best["loss"],
                        penalty=best["penalty"],
                        alpha=best["alpha"],
                        l1_ratio=best["l1_ratio"],
                        fit_intercept=best["fit_intercept"],
                        max_iter=best["max_iter"],
                        tol=best["tol"],
                        shuffle=best["shuffle"],
                        epsilon=best["epsilon"],
                        learning_rate=best["learning_rate"],
                        eta0=best["eta0"],
                        power_t=best["power_t"],
                        early_stopping=best["early_stopping"],
                        validation_fraction=best["validation_fraction"],
                        n_iter_no_change=best["n_iter_no_change"],
                        warm_start=best["warm_start"],
                        average=best["average"],
                    )
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_loss": best["loss"],
                        "Best_penalty": best["penalty"],
                        "Best_alpha": best["alpha"],
                        "Best_l1_ratio": best["l1_ratio"],
                        "Best_fit_intercept": best["fit_intercept"],
                        "Best_max_iter": best["max_iter"],
                        "Best_tol": best["tol"],
                        "Best_shuffle": best["shuffle"],
                        "Best_epsilon": best["epsilon"],
                        "Best_learning_rate": best["learning_rate"],
                        "Best_eta0": best["eta0"],
                        "Best_power_t": best["power_t"],
                        "Best_early_stopping": best["early_stopping"],
                        "Best_validation_fraction": best["validation_fraction"],
                        "Best_n_iter_no_change": best["n_iter_no_change"],
                        "Best_warm_start": best["warm_start"],
                        "Best_average": best["average"]
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]

        
            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")



            out_path = os.path.join(outdir, f"SGDRegressor_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (SGDRegressor) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (SGDRegressor) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))




























# ==========================================================
# Nested CV Window (KNeighborsRegressor)
# ==========================================================
class NestedKNeighborsRegressorCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_KNeighborsRegressor_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        # Hyperparameters as STRING LISTS (comma-separated)
        self.n_neighbors_list    = tk.StringVar(value="3,5,7,9,11")
        self.weights_list        = tk.StringVar(value="uniform, distance")
        self.algorithm_list      = tk.StringVar(value="auto, ball_tree, kd_tree, brute")
        self.leaf_size_list      = tk.StringVar(value="30, 50, 70")
        self.p_list              = tk.StringVar(value="1, 2")
        self.metric_list         = tk.StringVar(value="minkowski, euclidean, manhattan")
        self.metric_params_list  = tk.StringVar(value="None")
        self.n_jobs_list         = tk.StringVar(value="-1, 1")


        # CV settings
        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="KNeighborsRegressor hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="n_neighbors list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.n_neighbors_list, width=40).grid(row=0, column=1, sticky="w", **pad)
        
        ttk.Label(hp, text="weights list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.weights_list, width=40).grid(row=1, column=1, sticky="w", **pad)
        
        ttk.Label(hp, text="algorithm list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.algorithm_list, width=40).grid(row=2, column=1, sticky="w", **pad)
        
        ttk.Label(hp, text="leaf_size list:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.leaf_size_list, width=40).grid(row=3, column=1, sticky="w", **pad)
        
        ttk.Label(hp, text="p list:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.p_list, width=40).grid(row=4, column=1, sticky="w", **pad)
        
        ttk.Label(hp, text="metric list:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.metric_list, width=40).grid(row=5, column=1, sticky="w", **pad)
        
        ttk.Label(hp, text="metric_params list:").grid(row=6, column    =0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.metric_params_list, width=40).grid(row=6, column=1, sticky="w", **pad)
        
        ttk.Label(hp, text="n_jobs list:").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.n_jobs_list, width=40).grid(row=7, column=1, sticky="w", **pad)


        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (SGDRegression)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")

    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 1000,5000,10000")

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")


    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            n_neighbors_vals   = self._parse_int_list(self.n_neighbors_list.get(), "n_neighbors_list")
            weights_vals       = [x.strip() for x in self.weights_list.get().split(",") if x.strip() != ""]
            algorithm_vals     = [x.strip() for x in self.algorithm_list.get().split(",") if x.strip() != ""]
            leaf_size_vals     = self._parse_int_list(self.leaf_size_list.get(), "leaf_size_list")
            p_vals             = self._parse_int_list(self.p_list.get(), "p_list")
            metric_vals        = [x.strip() for x in self.metric_list.get().split(",") if x.strip() != ""]
            metric_params_vals = [None if x.strip().lower() == "none" else eval(x.strip()) for x in self.metric_params_list.get().split(",")]
            n_jobs_vals        = self._parse_int_list(self.n_jobs_list.get(), "n_jobs_list")    
    
         
            grid = list(product(n_neighbors_vals, weights_vals, algorithm_vals, leaf_size_vals, p_vals,
                                metric_vals, metric_params_vals, n_jobs_vals))
        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return

        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (KNeighborsRegressor) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"Tune by: {args['tune_by']}")
        self._start_time = time.perf_counter()
        self._start("Running nested CV (KNeighborsRegressor)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []

                    for (n_neighbors, weights, algorithm, leaf_size, p, metric, metric_params, n_jobs) in grid:
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = KNeighborsRegressor(
                                n_neighbors=n_neighbors,
                                weights=weights,
                                algorithm=algorithm,
                                leaf_size=leaf_size,
                                p=p,
                                metric=metric,
                                metric_params=metric_params,
                                n_jobs=n_jobs
                            )
                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan

                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "n_neighbors": n_neighbors,
                            "weights": weights,
                            "algorithm": algorithm,
                            "leaf_size": leaf_size,
                            "p": p,
                            "metric": metric,
                            "metric_params": metric_params,
                            "n_jobs": n_jobs,
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = KNeighborsRegressor(
                        n_neighbors=best["n_neighbors"],
                        weights=best["weights"],
                        algorithm=best["algorithm"],
                        leaf_size=best["leaf_size"],
                        p=best["p"],
                        metric=best["metric"],
                        metric_params=best["metric_params"],
                        n_jobs=best["n_jobs"]
                    )
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_n_neighbors": best["n_neighbors"],
                        "Best_weights": best["weights"],
                        "Best_algorithm": best["algorithm"],
                        "Best_leaf_size": best["leaf_size"],
                        "Best_p": best["p"],
                        "Best_metric": best["metric"],
                        "Best_metric_params": best["metric_params"],
                        "Best_n_jobs": best["n_jobs"],
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]


            
            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")



            out_path = os.path.join(outdir, f"KNeighborsRegressor_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (KNeighborsRegressor) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (KNeighborsRegressor) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))
























# ==========================================================
# Nested CV Window (GaussianProcessRegressor)
# ==========================================================
class NestedGaussianProcessRegressorCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_GaussianProcessRegressor_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        # Hyperparameters as STRING LISTS (comma-separated)

        rbf = 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))
        rq = 1.0 * RationalQuadratic(length_scale=1.0, alpha=0.1, alpha_bounds=(1e-5, 1e15))
        ess = 1.0 * ExpSineSquared(
                        length_scale=1.0,
                        periodicity=3.0,
                        length_scale_bounds=(0.1, 10.0),
                        periodicity_bounds=(1.0, 10.0),
                        )   
        ck = ConstantKernel(0.1, (0.01, 10.0)) * (DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2)
        mtr = 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)

        self.kernel_values      = tk.StringVar(value="rbf,rq,ess,ck,mtr")
        self.alpha_values       = tk.StringVar(value="1e-10,1e-5,1e-2")
        self.optimizer_values   = tk.StringVar(value="fmin_l_bfgs_b, None")
        self.n_restarts_values  = tk.StringVar(value="0, 5, 10")
        self.normalize_y_values = tk.StringVar(value="True, False")
        self.copy_X_train_values= tk.StringVar(value="True, False")



        # CV settings
        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="GaussianProcessRegressor hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="kernel values:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.kernel_values, width=40).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(hp, text="alpha values:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.alpha_values, width=40).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(hp, text="optimizer values:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.optimizer_values, width=40).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(hp, text="n_restarts values:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.n_restarts_values, width=40).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(hp, text="normalize_y values:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.normalize_y_values, width=40).grid(row=4, column=1, sticky="w", **pad)

        ttk.Label(hp, text="copy_X_train values:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.copy_X_train_values, width=40).grid(row=5, column=1, sticky="w", **pad)

        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (GaussianProcessRegressor)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")

    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 1000,5000,10000")

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")

    def _parse_kernel_list(self, s: str, name: str = "kernel_values"):
        allowed = {"rbf", "rq", "ess", "ck", "mtr"}
        vals = [x.strip().lower() for x in s.split(",") if x.strip()]
        if not vals:
            raise ValueError(f"{name} is empty. Example: rbf,rq,ess,ck,mtr")
        bad = [k for k in vals if k not in allowed]
        if bad:
            raise ValueError(f"Invalid {name}: {bad}. Allowed: rbf, rq, ess, ck, mtr")
        return vals

    def _kernel_from_name(self, k: str):
        k = k.strip().lower()

        if k == "rbf":
            return 1.0 * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 10.0))

        if k == "rq":
            return 1.0 * RationalQuadratic(
                length_scale=1.0,
                alpha=0.1,
                alpha_bounds=(1e-5, 1e15)
            )

        if k == "ess":
            return 1.0 * ExpSineSquared(
                length_scale=1.0,
                periodicity=3.0,
                length_scale_bounds=(0.1, 10.0),
                periodicity_bounds=(1.0, 10.0),
            )

        if k == "ck":
            return ConstantKernel(0.1, (0.01, 10.0)) * (
                DotProduct(sigma_0=1.0, sigma_0_bounds=(0.1, 10.0)) ** 2
            )

        if k == "mtr":
            return 1.0 * Matern(length_scale=1.0, length_scale_bounds=(1e-1, 10.0), nu=1.5)

        raise ValueError(f"Unknown kernel name: {k}")



    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            kernel_names      = self._parse_kernel_list(self.kernel_values.get(), "kernel_values")
            kernel_objs       = [self._kernel_from_name(k) for k in kernel_names]

            alpha_vals        = self._parse_float_list(self.alpha_values.get(), "alpha_values")
            optimizer_vals    = [None if x.strip().lower() == "none" else x.strip()
                                for x in self.optimizer_values.get().split(",") if x.strip() != ""]
            n_restarts_vals   = self._parse_int_list(self.n_restarts_values.get(), "n_restarts_values")
            normalize_y_vals  = self._parse_bool_list(self.normalize_y_values.get(), "normalize_y_values")
            copy_X_train_vals = self._parse_bool_list(self.copy_X_train_values.get(), "copy_X_train_values")

            # ✅ grid now contains REAL kernel objects, not strings
            grid = list(product(
                kernel_objs, alpha_vals, optimizer_vals, n_restarts_vals,
                normalize_y_vals, copy_X_train_vals
            ))

        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return

        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (GaussianProcessRegressor) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"Tune by: {args['tune_by']}")
        self._start_time = time.perf_counter()
        self._start("Running nested CV (GaussianProcessRegressor)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []

                    for (kernel, alpha, optimizer, n_restarts_optimizer, normalize_y, copy_X_train) in grid:
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = GaussianProcessRegressor(
                                kernel=kernel,
                                alpha=float(alpha),
                                optimizer=optimizer,  # "fmin_l_bfgs_b" or None
                                n_restarts_optimizer=int(n_restarts_optimizer),
                                normalize_y=bool(normalize_y),
                                copy_X_train=bool(copy_X_train),
                                random_state=0
                            )
                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan

                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "kernel": kernel,
                            "alpha": alpha,
                            "optimizer": optimizer,
                            "n_restarts_optimizer": n_restarts_optimizer,
                            "normalize_y": normalize_y,
                            "copy_X_train": copy_X_train,
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = GaussianProcessRegressor(
                        kernel=best["kernel"],
                        alpha=best["alpha"],
                        optimizer=best["optimizer"],
                        n_restarts_optimizer=best["n_restarts_optimizer"],
                        normalize_y=best["normalize_y"],
                        copy_X_train=best["copy_X_train"],
                    )
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_kernel": best["kernel"],
                        "Best_alpha": best["alpha"],
                        "Best_optimizer": best["optimizer"],
                        "Best_n_restarts_optimizer": best["n_restarts_optimizer"],
                        "Best_normalize_y": best["normalize_y"],
                        "Best_copy_X_train": best["copy_X_train"]
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]



        
            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")




            out_path = os.path.join(outdir, f"GaussianProcessRegressor_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (GaussianProcessRegressor) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (GaussianProcessRegressor) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))







































# ==========================================================
# Nested CV Window (PLSRegression)
# ==========================================================

class NestedPLSRegressionCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_PLSRegression_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        self.n_components_list = tk.StringVar(value="2,5,10,20,50")
        self.scale_y_list      = tk.StringVar(value="True, False")
        self.max_iter_list      = tk.StringVar(value="500, 1000, 2000")
        self.tol_list           = tk.StringVar(value="1e-6, 1e-5, 1e-4")
        self.copy_list         = tk.StringVar(value="True, False")

        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="PLSRegression hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="N components list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.n_components_list, width=60).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Scale y values:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.scale_y_list, width=60).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Max iterations:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_iter_list, width=60).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Tolerance:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.tol_list, width=60).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Copy input X:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.copy_list, width=60).grid(row=4, column=1, sticky="w", **pad)

        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (PLSRegression)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df
    
    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 2,5,10,20")

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")
        

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")

    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            n_components_vals = self._parse_int_list(self.n_components_list.get(), "n_components_list")
            scale_y_vals      = self._parse_bool_list(self.scale_y_list.get(), "scale_y_list")
            max_iter_vals     = self._parse_int_list(self.max_iter_list.get(), "max_iter_list")
            tol_vals          = self._parse_float_list(self.tol_list.get(), "tol_list")
            copy_vals         = self._parse_bool_list(self.copy_list.get(), "copy_list")


            grid = list(product(
                n_components_vals, scale_y_vals, max_iter_vals, tol_vals, copy_vals
            ))

        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return




        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (PLSRegression) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"tune_by={args['tune_by']}")
        self._start_time = time.perf_counter()
        self._start("Running nested CV (PLSRegression)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    


    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []
                    for n_components,  scale, max_iter, tol, copy in grid:
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = PLSRegression(
                                n_components=int(n_components),
                                scale=bool(scale),
                                max_iter=int(max_iter),
                                tol=float(tol),
                                copy=bool(copy)
                            )

                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan
                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "n_components": n_components,
                            "scale": scale,
                            "max_iter": max_iter,
                            "tol": tol,
                            "copy": copy,
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = PLSRegression(
                        n_components=int(best["n_components"]),
                        scale=bool(best["scale"]),
                        max_iter=int(best["max_iter"]),
                        tol=float(best["tol"]),
                        copy=bool(best["copy"])
                    )
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_n_components": best["n_components"],
                        "Best_scale": best["scale"],
                        "Best_max_iter": best["max_iter"],
                        "Best_tol": best["tol"],
                        "Best_copy": best["copy"]
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]


            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")


            out_path = os.path.join(outdir, f"PLSRegression_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (PLSRegression) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (PLSRegression) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))




















# ==========================================================
# Nested CV Window (DecisionTreeRegressor)
# ==========================================================

class NestedDecisionTreeRegressorCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_DecisionTreeRegressor_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        self.criterion_list        = tk.StringVar(value="squared_error, friedman_mse, absolute_error, poisson")
        self.splitter_list         = tk.StringVar(value="best, random")
        self.max_depth_list        = tk.StringVar(value="None, 5, 10, 20, 50")
        self.min_samples_split_list = tk.StringVar(value="2, 5, 10")
        self.min_samples_leaf_list  = tk.StringVar(value="1, 2, 5")
        self.min_weight_fraction_leaf_list = tk.StringVar(value="0.0, 0.1, 0.2")
        self.max_features_list     = tk.StringVar(value="None, sqrt, log2")
        self.max_leaf_nodes_list    = tk.StringVar(value="None, 5, 10, 20, 50")
        self.min_impurity_decrease_list = tk.StringVar(value="0.0, 0.01, 0.1")
        self.ccp_alpha_list        = tk.StringVar(value="0.0, 0.01, 0.1")
        

        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="DecisionTreeRegressor hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="Criterion list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.criterion_list, width=60).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Splitter list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.splitter_list, width=60).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Max depth list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_depth_list, width=60).grid(row=2, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Min samples split list:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_samples_split_list, width=60).grid(row=3, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Min samples leaf list:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_samples_leaf_list, width=60).grid(row=4, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Min weight fraction leaf list:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_weight_fraction_leaf_list, width=60).grid(row=5, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Max features list:").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_features_list, width=60).grid(row=6, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Max leaf nodes list:").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_leaf_nodes_list, width=60).grid(row=7, column=1, sticky="w", **pad)

        ttk.Label(hp, text="Min impurity decrease list:").grid(row=8, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_impurity_decrease_list, width=60).grid(row=8, column=1, sticky="w", **pad)

        ttk.Label(hp, text="CCP alpha list:").grid(row=9, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.ccp_alpha_list, width=60).grid(row=9, column=1, sticky="w", **pad)









        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (DecisionTreeRegressor)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df
    
    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 2,5,10,20")

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")
        

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")


    def _parse_optional_int_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if not x:
                    continue
                if x == "none":
                    vals.append(None)
                else:
                    vals.append(int(x))
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(
                f"Invalid {name}. Use comma-separated integers or None, e.g. None,5,10,20"
            )



    def _parse_max_features_list(self, s: str, name: str):
        vals = []
        for x in s.split(","):
            t = x.strip()
            if not t:
                continue

            low = t.lower()

            if low == "none":
                vals.append(None)
            elif low in {"sqrt", "log2"}:
                vals.append(low)
            else:
                # try int, then float
                try:
                    vals.append(int(float(t)))
                except ValueError:
                    try:
                        vals.append(float(t))
                    except ValueError:
                        raise ValueError(
                            f"Invalid {name}: '{t}'. "
                            "Allowed: None, sqrt, log2, int, float"
                        )

        if not vals:
            raise ValueError(f"{name} is empty.")

        return vals



    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            criterion_vals        = [x.strip() for x in self.criterion_list.get().split(",") if x.strip() != ""]
            splitter_vals         = [x.strip() for x in self.splitter_list.get().split(",") if x.strip() != ""]
            max_depth_vals        = self._parse_optional_int_list(self.max_depth_list.get(), "max_depth_list")
            min_samples_split_vals = [int(float(x.strip())) for x in self.min_samples_split_list.get().split(",") if x.strip() != ""]
            min_samples_leaf_vals  = [int(float(x.strip())) for x in self.min_samples_leaf_list.get().split(",") if x.strip() != ""]
            min_weight_fraction_leaf_vals = [float(x.strip()) for x in self.min_weight_fraction_leaf_list.get().split(",") if x.strip() != ""]
            max_features_vals = self._parse_max_features_list(self.max_features_list.get(), "max_features_list")

            max_leaf_nodes_vals    = self._parse_optional_int_list(self.max_leaf_nodes_list.get(), "max_leaf_nodes_list")
            min_impurity_decrease_vals = [float(x.strip()) for x in self.min_impurity_decrease_list.get().split(",") if x.strip() != ""]
            ccp_alpha_vals        = [float(x.strip()) for x in self.ccp_alpha_list.get().split(",") if x.strip() != ""]


            grid = list(product(
                criterion_vals, splitter_vals, max_depth_vals,
                min_samples_split_vals, min_samples_leaf_vals, min_weight_fraction_leaf_vals,
                max_features_vals, max_leaf_nodes_vals, min_impurity_decrease_vals, ccp_alpha_vals
            ))

        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return




        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (DecisionTreeRegressor) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"tune_by={args['tune_by']}")
        self._start_time = time.perf_counter()
        self._start("Running nested CV (DecisionTreeRegressor)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    


    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []
                    for criterion, splitter, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, vmin_impurity_decrease, ccp_alpha in grid:
                        
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = DecisionTreeRegressor(
                                criterion=criterion,
                                splitter=splitter,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_features=max_features,
                                max_leaf_nodes=max_leaf_nodes,
                                min_impurity_decrease=vmin_impurity_decrease,
                                ccp_alpha=ccp_alpha
                            )

                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan
                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "criterion": criterion,
                            "splitter": splitter,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                            "min_weight_fraction_leaf": min_weight_fraction_leaf,
                            "max_features": max_features,
                            "max_leaf_nodes": max_leaf_nodes,
                            "min_impurity_decrease": vmin_impurity_decrease,
                            "ccp_alpha": ccp_alpha,
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = DecisionTreeRegressor(
                        criterion=best["criterion"],
                        splitter=best["splitter"],
                        max_depth=best["max_depth"],
                        min_samples_split=best["min_samples_split"],
                        min_samples_leaf=best["min_samples_leaf"],
                        min_weight_fraction_leaf=best["min_weight_fraction_leaf"],
                        max_features=best["max_features"],
                        max_leaf_nodes=best["max_leaf_nodes"],
                        min_impurity_decrease=best["min_impurity_decrease"],
                        ccp_alpha=best["ccp_alpha"]
                    )
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_criterion": best["criterion"],
                        "Best_splitter": best["splitter"],
                        "Best_max_depth": best["max_depth"],
                        "Best_min_samples_split": best["min_samples_split"],
                        "Best_min_samples_leaf": best["min_samples_leaf"],
                        "Best_min_weight_fraction_leaf": best["min_weight_fraction_leaf"],
                        "Best_max_features": best["max_features"],
                        "Best_max_leaf_nodes": best["max_leaf_nodes"],
                        "Best_min_impurity_decrease": best["min_impurity_decrease"],
                        "Best_ccp_alpha": best["ccp_alpha"]
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]


        
            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")


            out_path = os.path.join(outdir, f"DecisionTreeRegressor_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (DecisionTreeRegressor) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (DecisionTreeRegressor) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))






























# ==========================================================
# Nested CV Window (GradientBoostingRegressor)
# ==========================================================

class NestedGradientBoostingRegressorCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_GradientBoostingRegressor_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        self.loss_list              = tk.StringVar(value="squared_error, absolute_error, huber, quantile")
        self.learning_rate_list     = tk.StringVar(value="0.01, 0.1, 0.5, 1")
        self.n_estimators_list      = tk.StringVar(value="100, 200, 500")
        self.subsample_list         = tk.StringVar(value="0.5, 0.7, 1.0")
        self.criterion_list        = tk.StringVar(value="friedman_mse, squared_error")
        self.min_samples_split_list = tk.StringVar(value="2, 5, 10")
        self.min_samples_leaf_list  = tk.StringVar(value="1, 2, 5")
        self.min_weight_fraction_leaf_list = tk.StringVar(value="0.0, 0.1, 0.2")
        self.max_depth_list        = tk.StringVar(value="3, 5, 10")
        self.min_impurity_decrease_list = tk.StringVar(value="0.0, 0.01, 0.1")
        self.max_features_list     = tk.StringVar(value="None, sqrt, log2")
        self.alpha_list             = tk.StringVar(value="0.1, 0.5, 0.9")
        self.max_leaf_nodes_list    = tk.StringVar(value="None, 2, 5, 10, 20")
        self.warm_start_list        = tk.StringVar(value="True, False")
        self.n_in_iter_no_change_list = tk.StringVar(value="None, 5, 10, 20")
        self.tol_list               = tk.StringVar(value="1e-4, 1e-3, 1e-2")
        self.ccp_alpha_list        = tk.StringVar(value="0.0, 0.01, 0.1")



        

        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="GradientBoostingRegressor hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="Loss list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.loss_list, width=60).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Learning rate list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.learning_rate_list, width=60).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Number of estimators list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.n_estimators_list, width=60).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Subsample list:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.subsample_list, width=60).grid(row=3, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Criterion list:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.criterion_list, width=60).grid(row=4, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min samples split list:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_samples_split_list, width=60).grid(row=5, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min samples leaf list:").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_samples_leaf_list, width=60).grid(row=6, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min weight fraction leaf list:").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_weight_fraction_leaf_list, width=60).grid(row=7, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max depth list:").grid(row=8, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_depth_list, width=60).grid(row=8, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min impurity decrease list:").grid(row=9, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_impurity_decrease_list, width=60).grid(row=9, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max features list:").grid(row=10, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_features_list, width=60).grid(row=10, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Alpha list:").grid(row=11, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.alpha_list, width=60).grid(row=11, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max leaf nodes list:").grid(row=12, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_leaf_nodes_list, width=60).grid(row=12, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Warm start list:").grid(row=13, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.warm_start_list, width=60).grid(row=13, column=1, sticky="w", **pad)
        ttk.Label(hp, text="N in iter no change list:").grid(row=14, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.n_in_iter_no_change_list, width=60).grid(row=14, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Tol list:").grid(row=15, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.tol_list, width=60).grid(row=15, column=1, sticky="w", **pad)
        ttk.Label(hp, text="CCP alpha list:").grid(row=16, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.ccp_alpha_list, width=60).grid(row=16, column=1, sticky="w", **pad)





        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (GradientBoostingRegressor)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df
    
    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 2,5,10,20")

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")
        

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")


    def _parse_optional_int_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if not x:
                    continue
                if x == "none":
                    vals.append(None)
                else:
                    vals.append(int(x))
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(
                f"Invalid {name}. Use comma-separated integers or None, e.g. None,5,10,20"
            )



    def _parse_max_features_list(self, s: str, name: str):
        vals = []
        for x in s.split(","):
            t = x.strip()
            if not t:
                continue

            low = t.lower()

            if low == "none":
                vals.append(None)
            elif low in {"sqrt", "log2"}:
                vals.append(low)
            else:
                # try int, then float
                try:
                    vals.append(int(float(t)))
                except ValueError:
                    try:
                        vals.append(float(t))
                    except ValueError:
                        raise ValueError(
                            f"Invalid {name}: '{t}'. "
                            "Allowed: None, sqrt, log2, int, float"
                        )

        if not vals:
            raise ValueError(f"{name} is empty.")

        return vals



    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            loss_vals           = [x.strip() for x in self.loss_list.get().split(",") if x.strip() != ""]
            learning_rate_vals  = self._parse_float_list(self.learning_rate_list.get(), "learning_rate_list")
            n_estimators_vals   = self._parse_int_list(self.n_estimators_list.get(), "n_estimators_list")
            subsample_vals      = self._parse_float_list(self.subsample_list.get(), "subsample_list")
            criterion_vals        = [x.strip() for x in self.criterion_list.get().split(",") if x.strip() != ""]
            min_samples_split_vals = self._parse_int_list(self.min_samples_split_list.get(), "min_samples_split_list")
            min_samples_leaf_vals  = self._parse_int_list(self.min_samples_leaf_list.get(), "min_samples_leaf_list")
            min_weight_fraction_leaf_vals = [float(x.strip()) for x in self.min_weight_fraction_leaf_list.get().split(",") if x.strip() != ""]
            max_depth_vals        = self._parse_int_list(self.max_depth_list.get(), "max_depth_list")
            min_impurity_decrease_vals = [float(x.strip()) for x in self.min_impurity_decrease_list.get().split(",") if x.strip() != ""]
            max_features_vals     = self._parse_max_features_list(self.max_features_list.get(), "max_features_list")
            alpha_vals             = self._parse_float_list(self.alpha_list.get(), "alpha_list")
            max_leaf_nodes_vals    = self._parse_optional_int_list(self.max_leaf_nodes_list.get(), "max_leaf_nodes_list")
            warm_start_vals        = self._parse_bool_list(self.warm_start_list.get(), "warm_start_list")
            n_in_iter_no_change_vals = self._parse_optional_int_list(self.n_in_iter_no_change_list.get(), "n_in_iter_no_change_list")
            tol_vals               = self._parse_float_list(self.tol_list.get(), "tol_list")
            ccp_alpha_vals        = self._parse_float_list(self.ccp_alpha_list.get(), "ccp_alpha_list")

            grid = list(product(
                loss_vals,
                learning_rate_vals,
                n_estimators_vals,
                subsample_vals,
                criterion_vals,
                min_samples_split_vals,
                min_samples_leaf_vals,
                min_weight_fraction_leaf_vals,
                max_depth_vals,
                min_impurity_decrease_vals,
                max_features_vals,
                alpha_vals,
                max_leaf_nodes_vals,
                warm_start_vals,
                n_in_iter_no_change_vals,
                tol_vals,
                ccp_alpha_vals
            ))

        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return




        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (GradientBoostingRegressor) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"tune_by={args['tune_by']}")
        self._start_time = time.perf_counter()
        self._start("Running nested CV (GradientBoostingRegressor)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    


    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []
                    for loss, learning_rate, n_estimators, subsample, criterion, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_depth, min_impurity_decrease, max_features, alpha, max_leaf_nodes, warm_start, n_in_iter_no_change, tol, ccp_alpha in grid:
                        
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = GradientBoostingRegressor(
                                loss=loss,
                                learning_rate=learning_rate,
                                n_estimators=n_estimators,
                                subsample=subsample,
                                criterion=criterion,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_depth=max_depth,
                                min_impurity_decrease=min_impurity_decrease,
                                max_features=max_features,
                                alpha=alpha,
                                max_leaf_nodes=max_leaf_nodes,
                                warm_start=warm_start,
                                n_iter_no_change=n_in_iter_no_change,
                                tol=tol,
                                ccp_alpha=ccp_alpha
                            )

                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan
                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "loss": loss,
                            "learning_rate": learning_rate,
                            "n_estimators": n_estimators,
                            "subsample": subsample,
                            "criterion": criterion,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                            "min_weight_fraction_leaf": min_weight_fraction_leaf,
                            "max_depth": max_depth,
                            "min_impurity_decrease": min_impurity_decrease,
                            "max_features": max_features,
                            "alpha": alpha,
                            "max_leaf_nodes": max_leaf_nodes,
                            "warm_start": warm_start,
                            "n_iter_no_change": n_in_iter_no_change,
                            "tol": tol,
                            "ccp_alpha": ccp_alpha,
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = GradientBoostingRegressor(
                        loss=best["loss"],
                        learning_rate=best["learning_rate"],
                        n_estimators=best["n_estimators"],
                        subsample=best["subsample"],
                        criterion=best["criterion"],
                        min_samples_split=best["min_samples_split"],
                        min_samples_leaf=best["min_samples_leaf"],
                        min_weight_fraction_leaf=best["min_weight_fraction_leaf"],
                        max_depth=best["max_depth"],
                        min_impurity_decrease=best["min_impurity_decrease"],
                        max_features=best["max_features"],
                        alpha=best["alpha"],
                        max_leaf_nodes=best["max_leaf_nodes"],
                        warm_start=best["warm_start"],
                        n_iter_no_change=best["n_iter_no_change"],
                        tol=best["tol"],
                        ccp_alpha=best["ccp_alpha"]
                    )
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_loss": best["loss"],
                        "Best_learning_rate": best["learning_rate"],
                        "Best_n_estimators": best["n_estimators"],
                        "Best_subsample": best["subsample"],
                        "Best_criterion": best["criterion"],
                        "Best_min_samples_split": best["min_samples_split"],
                        "Best_min_samples_leaf": best["min_samples_leaf"],
                        "Best_min_weight_fraction_leaf": best["min_weight_fraction_leaf"],
                        "Best_max_depth": best["max_depth"],
                        "Best_min_impurity_decrease": best["min_impurity_decrease"],
                        "Best_max_features": best["max_features"],
                        "Best_alpha": best["alpha"],
                        "Best_max_leaf_nodes": best["max_leaf_nodes"],
                        "Best_warm_start": best["warm_start"],
                        "Best_n_iter_no_change": best["n_iter_no_change"],
                        "Best_tol": best["tol"],
                        "Best_ccp_alpha": best["ccp_alpha"]
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]


            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")


            out_path = os.path.join(outdir, f"GradientBoostingRegressor_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (GradientBoostingRegressor) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (GradientBoostingRegressor) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))

















































# ==========================================================
# Nested CV Window (RandomForestRegressor)
# ==========================================================

class NestedRandomForestRegressorCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_RandomForestRegressor_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)


        self.n_estimators_list      = tk.StringVar(value="100, 200, 500, 1000")
        self.criterion_list        = tk.StringVar(value="friedman_mse, squared_error")
        self.max_depth_list        = tk.StringVar(value="3, 5, 10")
        self.min_samples_split_list = tk.StringVar(value="2, 5, 10")
        self.min_samples_leaf_list  = tk.StringVar(value="1, 2, 5")
        self.min_weight_fraction_leaf_list = tk.StringVar(value="0.0, 0.1, 0.2")
        self.max_features_list     = tk.StringVar(value="None, sqrt, log2")
        self.max_leaf_nodes_list    = tk.StringVar(value="None, 10, 20")
        self.min_impurity_decrease_list = tk.StringVar(value="0.0, 0.01, 0.1")
        self.bootstrap_list        = tk.StringVar(value="True, False")
        self.oob_score_list       = tk.StringVar(value="True, False")
        self.warm_start_list        = tk.StringVar(value="True, False")
        self.ccp_alpha_list        = tk.StringVar(value="0.0, 0.01, 0.1")
        self.max_saples_list        = tk.StringVar(value="None, 0.5, 0.75")

        

        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="RandomForestRegressor hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="n_estimators list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.n_estimators_list, width=60).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Criterion list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.criterion_list, width=60).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min samples split list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_samples_split_list, width=60).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min samples leaf list:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_samples_leaf_list, width=60).grid(row=3, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min weight fraction leaf list:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_weight_fraction_leaf_list, width=60).grid(row=4, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max depth list:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_depth_list, width=60).grid(row=5, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min impurity decrease list:").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_impurity_decrease_list, width=60).grid(row=6, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max features list:").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_features_list, width=60).grid(row=7, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max leaf nodes list:").grid(row=8, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_leaf_nodes_list, width=60).grid(row=8, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Bootstrap list:").grid(row=9, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.bootstrap_list, width=60).grid(row=9, column=1, sticky="w", **pad)
        ttk.Label(hp, text="OOB score list:").grid(row=10, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.oob_score_list, width=60).grid(row=10, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Warm start list:").grid(row=11, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.warm_start_list, width=60).grid(row=11, column=1, sticky="w", **pad)
        ttk.Label(hp, text="CCP alpha list:").grid(row=12, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.ccp_alpha_list, width=60).grid(row=12, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max samples list:").grid(row=13, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_saples_list, width=60).grid(row=13, column=1, sticky="w", **pad)



        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (RandomForestRegressor)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df
    
    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 2,5,10,20")

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")
        

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")


    def _parse_optional_int_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if not x:
                    continue
                if x == "none":
                    vals.append(None)
                else:
                    vals.append(int(x))
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(
                f"Invalid {name}. Use comma-separated integers or None, e.g. None,5,10,20"
            )



    def _parse_max_features_list(self, s: str, name: str):
        vals = []
        for x in s.split(","):
            t = x.strip()
            if not t:
                continue

            low = t.lower()

            if low == "none":
                vals.append(None)
            elif low in {"sqrt", "log2"}:
                vals.append(low)
            else:
                # try int, then float
                try:
                    vals.append(int(float(t)))
                except ValueError:
                    try:
                        vals.append(float(t))
                    except ValueError:
                        raise ValueError(
                            f"Invalid {name}: '{t}'. "
                            "Allowed: None, sqrt, log2, int, float"
                        )

        if not vals:
            raise ValueError(f"{name} is empty.")

        return vals



    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            n_estimators_vals      = self._parse_int_list(self.n_estimators_list.get(), "n_estimators_list")
            criterion_vals        = [x.strip() for x in self.criterion_list.get().split(",") if x.strip() != ""]
            max_depth_vals        = self._parse_optional_int_list(self.max_depth_list.get(), "max_depth_list")
            min_samples_split_vals = self._parse_int_list(self.min_samples_split_list.get(), "min_samples_split_list")
            min_samples_leaf_vals  = self._parse_int_list(self.min_samples_leaf_list.get(), "min_samples_leaf_list")
            min_weight_fraction_leaf_vals = self._parse_float_list(self.min_weight_fraction_leaf_list.get(), "min_weight_fraction_leaf_list")
            max_features_vals     = self._parse_max_features_list(self.max_features_list.get(), "max_features_list")
            max_leaf_nodes_vals    = self._parse_optional_int_list(self.max_leaf_nodes_list.get(), "max_leaf_nodes_list")
            min_impurity_decrease_vals = self._parse_float_list(self.min_impurity_decrease_list.get(), "min_impurity_decrease_list")
            bootstrap_vals        = self._parse_bool_list(self.bootstrap_list.get(), "bootstrap_list")
            oob_score_vals       = self._parse_bool_list(self.oob_score_list.get(), "oob_score_list")
            warm_start_vals        = self._parse_bool_list(self.warm_start_list.get(), "warm_start_list")
            ccp_alpha_vals        = self._parse_float_list(self.ccp_alpha_list.get(), "ccp_alpha_list")
            max_samples_vals        = self._parse_optional_int_list(self.max_saples_list.get(), "max_samples_list")

            grid = list(product(
                n_estimators_vals,
                criterion_vals,
                max_depth_vals,
                min_samples_split_vals,
                min_samples_leaf_vals,
                min_weight_fraction_leaf_vals,
                max_features_vals,
                max_leaf_nodes_vals,
                min_impurity_decrease_vals,
                bootstrap_vals,
                oob_score_vals,
                warm_start_vals,
                ccp_alpha_vals,
                max_samples_vals
            ))

        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return




        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (RandomForestRegressor) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"tune_by={args['tune_by']}")
        self._start_time = time.perf_counter()
        self._start("Running nested CV (RandomForestRegressor)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    


    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []
                    for n_estimators, criterion, max_depth, min_samples_split, min_samples_leaf, min_weight_fraction_leaf, max_features, max_leaf_nodes, min_impurity_decrease, bootstrap, oob_score, warm_start, ccp_alpha, max_samples in grid:
                        
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = RandomForestRegressor(
                                n_estimators=n_estimators,
                                criterion=criterion,
                                max_depth=max_depth,
                                min_samples_split=min_samples_split,
                                min_samples_leaf=min_samples_leaf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf,
                                max_features=max_features,
                                max_leaf_nodes=max_leaf_nodes,
                                min_impurity_decrease=min_impurity_decrease,
                                bootstrap=bootstrap,
                                oob_score=oob_score,
                                warm_start=warm_start,
                                ccp_alpha=ccp_alpha,
                                max_samples=max_samples
                            )

                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan
                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "n_estimators": n_estimators,
                            "criterion": criterion,
                            "max_depth": max_depth,
                            "min_samples_split": min_samples_split,
                            "min_samples_leaf": min_samples_leaf,
                            "min_weight_fraction_leaf": min_weight_fraction_leaf,
                            "max_features": max_features,
                            "max_leaf_nodes": max_leaf_nodes,
                            "min_impurity_decrease": min_impurity_decrease,
                            "bootstrap": bootstrap,
                            "oob_score": oob_score,
                            "warm_start": warm_start,
                            "ccp_alpha": ccp_alpha,
                            "max_samples": max_samples,
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = RandomForestRegressor(
                        n_estimators=best["n_estimators"],
                        criterion=best["criterion"],
                        max_depth=best["max_depth"],
                        min_samples_split=best["min_samples_split"],
                        min_samples_leaf=best["min_samples_leaf"],
                        min_weight_fraction_leaf=best["min_weight_fraction_leaf"],
                        max_features=best["max_features"],
                        max_leaf_nodes=best["max_leaf_nodes"],
                        min_impurity_decrease=best["min_impurity_decrease"],
                        bootstrap=best["bootstrap"],
                        oob_score=best["oob_score"],
                        warm_start=best["warm_start"],
                        ccp_alpha=best["ccp_alpha"],
                        max_samples=best["max_samples"]

                    )
                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_n_estimators": best["n_estimators"],
                        "Best_criterion": best["criterion"],
                        "Best_max_depth": best["max_depth"],
                        "Best_min_samples_split": best["min_samples_split"],
                        "Best_min_samples_leaf": best["min_samples_leaf"],
                        "Best_min_weight_fraction_leaf": best["min_weight_fraction_leaf"],
                        "Best_max_features": best["max_features"],
                        "Best_max_leaf_nodes": best["max_leaf_nodes"],
                        "Best_min_impurity_decrease": best["min_impurity_decrease"],
                        "Best_bootstrap": best["bootstrap"],
                        "Best_oob_score": best["oob_score"],
                        "Best_warm_start": best["warm_start"],
                        "Best_ccp_alpha": best["ccp_alpha"],
                        "Best_max_samples": best["max_samples"]
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]


            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")


            out_path = os.path.join(outdir, f"RandomForestRegressor_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (RandomForestRegressor) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (RandomForestRegressor) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))












































# ==========================================================
# Nested CV Window (VotingRegressor)
# ==========================================================

class NestedVotingRegressorCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_VotingRegressor_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)


        self.n_estimators_rf_list      = tk.StringVar(value="100, 200, 500, 1000")
        self.criterion_rf_list        = tk.StringVar(value="friedman_mse, squared_error")
        self.max_depth_rf_list        = tk.StringVar(value="3, 5, 10")
        self.min_samples_split_rf_list = tk.StringVar(value="2, 5, 10")
        self.min_samples_leaf_rf_list  = tk.StringVar(value="1, 2, 5")
        self.min_weight_fraction_leaf_rf_list = tk.StringVar(value="0.0, 0.1, 0.2")
        self.max_features_rf_list     = tk.StringVar(value="None, sqrt, log2")
        self.max_leaf_nodes_rf_list    = tk.StringVar(value="None, 10, 20")
        self.min_impurity_decrease_rf_list = tk.StringVar(value="0.0, 0.01, 0.1")
        self.bootstrap_rf_list        = tk.StringVar(value="True, False")
        self.oob_score_rf_list       = tk.StringVar(value="True, False")
        self.warm_start_rf_list        = tk.StringVar(value="True, False")
        self.ccp_alpha_rf_list        = tk.StringVar(value="0.0, 0.01, 0.1")
        self.max_saples_rf_list        = tk.StringVar(value="None, 0.5, 0.75")

        
        self.ccp_alpha_lr_list        = tk.StringVar(value="0.0, 0.01, 0.1")

        self.n_neighbors_knr_list    = tk.StringVar(value="3,5,7,9,11")
        self.weights_knr_list        = tk.StringVar(value="uniform, distance")
        self.algorithm_knr_list      = tk.StringVar(value="auto, ball_tree, kd_tree, brute")
        self.leaf_size_knr_list      = tk.StringVar(value="30, 50, 70")
        self.p_knr_list              = tk.StringVar(value="1, 2")
        self.metric_knr_list         = tk.StringVar(value="minkowski, euclidean, manhattan")
        self.metric_params_knr_list  = tk.StringVar(value="None")
        self.n_jobs_knr_list         = tk.StringVar(value="-1, 1")

        self.weights_vr_list = tk.StringVar(value="None, 1,2,3,4,5")


        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="RandomForestRegressor hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        ttk.Label(hp, text="n_estimators list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.n_estimators_rf_list, width=60).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Criterion list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.criterion_rf_list, width=60).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min samples split list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_samples_split_rf_list, width=60).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min samples leaf list:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_samples_leaf_rf_list, width=60).grid(row=3, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min weight fraction leaf list:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_weight_fraction_leaf_rf_list, width=60).grid(row=4, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max depth list:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_depth_rf_list, width=60).grid(row=5, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Min impurity decrease list:").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.min_impurity_decrease_rf_list, width=60).grid(row=6, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max features list:").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_features_rf_list, width=60).grid(row=7, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max leaf nodes list:").grid(row=8, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_leaf_nodes_rf_list, width=60).grid(row=8, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Bootstrap list:").grid(row=9, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.bootstrap_rf_list, width=60).grid(row=9, column=1, sticky="w", **pad)
        ttk.Label(hp, text="OOB score list:").grid(row=10, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.oob_score_rf_list, width=60).grid(row=10, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Warm start list:").grid(row=11, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.warm_start_rf_list, width=60).grid(row=11, column=1, sticky="w", **pad)
        ttk.Label(hp, text="CCP alpha list:").grid(row=12, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.ccp_alpha_rf_list, width=60).grid(row=12, column=1, sticky="w", **pad)
        ttk.Label(hp, text="Max samples list:").grid(row=13, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_saples_rf_list, width=60).grid(row=13, column=1, sticky="w", **pad)

        hp_1 = ttk.LabelFrame(self, text="LinearRegression hyperparameters (inner tuning grid)")
        hp_1.pack(fill="x", **pad)

        ttk.Label(hp_1, text="alpha list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp_1, textvariable=self.ccp_alpha_lr_list, width=60).grid(row=0, column=1, sticky="w", **pad)


        hp_2 = ttk.LabelFrame(self, text="KNeighborsRegressor hyperparameters (inner tuning grid)")
        hp_2.pack(fill="x", **pad)

        ttk.Label(hp_2, text="n_neighbors list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp_2, textvariable=self.n_neighbors_knr_list, width=60).grid(row=0, column=1, sticky="w", **pad)
        ttk.Label(hp_2, text="weights list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp_2, textvariable=self.weights_knr_list, width=60).grid(row=1, column=1, sticky="w", **pad)
        ttk.Label(hp_2, text="algorithm list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp_2, textvariable=self.algorithm_knr_list, width=60).grid(row=2, column=1, sticky="w", **pad)
        ttk.Label(hp_2, text="leaf_size list:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp_2, textvariable=self.leaf_size_knr_list, width=60).grid(row=3, column=1, sticky="w", **pad)
        ttk.Label(hp_2, text="p list:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp_2, textvariable=self.p_knr_list, width=60).grid(row=4, column=1, sticky="w", **pad)
        ttk.Label(hp_2, text="metric list:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp_2, textvariable=self.metric_knr_list, width=60).grid(row=5, column=1, sticky="w", **pad)
        ttk.Label(hp_2, text="metric_params list:").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(hp_2, textvariable=self.metric_params_knr_list, width=60).grid(row=6, column=1, sticky="w", **pad)
        ttk.Label(hp_2, text="n_jobs list:").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(hp_2, textvariable=self.n_jobs_knr_list, width=60).grid(row=7, column=1, sticky="w", **pad)


        hp_3 = ttk.LabelFrame(self, text="VotingRegression hyperparameters (inner tuning grid)")
        hp_3.pack(fill="x", **pad)

        ttk.Label(hp_3, text="weights list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp_3, textvariable=self.weights_vr_list, width=60).grid(row=0, column=1, sticky="w", **pad)



        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (VotingRegressor)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df
    
    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 2,5,10,20")

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")
        

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")


    def _parse_optional_int_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if not x:
                    continue
                if x == "none":
                    vals.append(None)
                else:
                    vals.append(int(x))
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(
                f"Invalid {name}. Use comma-separated integers or None, e.g. None,5,10,20"
            )



    def _parse_max_features_list(self, s: str, name: str):
        vals = []
        for x in s.split(","):
            t = x.strip()
            if not t:
                continue

            low = t.lower()

            if low == "none":
                vals.append(None)
            elif low in {"sqrt", "log2"}:
                vals.append(low)
            else:
                # try int, then float
                try:
                    vals.append(int(float(t)))
                except ValueError:
                    try:
                        vals.append(float(t))
                    except ValueError:
                        raise ValueError(
                            f"Invalid {name}: '{t}'. "
                            "Allowed: None, sqrt, log2, int, float"
                        )

        if not vals:
            raise ValueError(f"{name} is empty.")

        return vals



    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
            n_estimators_rf_vals      = self._parse_int_list(self.n_estimators_rf_list.get(), "n_estimators_list")
            criterion_rf_vals        = [x.strip() for x in self.criterion_rf_list.get().split(",") if x.strip() != ""]
            max_depth_rf_vals        = self._parse_optional_int_list(self.max_depth_rf_list.get(), "max_depth_list")
            min_samples_split_rf_vals = self._parse_int_list(self.min_samples_split_rf_list.get(), "min_samples_split_list")
            min_samples_leaf_rf_vals  = self._parse_int_list(self.min_samples_leaf_rf_list.get(), "min_samples_leaf_list")
            min_weight_fraction_leaf_rf_vals = self._parse_float_list(self.min_weight_fraction_leaf_rf_list.get(), "min_weight_fraction_leaf_list")
            max_features_rf_vals     = self._parse_max_features_list(self.max_features_rf_list.get(), "max_features_list")
            max_leaf_nodes_rf_vals    = self._parse_optional_int_list(self.max_leaf_nodes_rf_list.get(), "max_leaf_nodes_list")
            min_impurity_decrease_rf_vals = self._parse_float_list(self.min_impurity_decrease_rf_list.get(), "min_impurity_decrease_list")
            bootstrap_rf_vals        = self._parse_bool_list(self.bootstrap_rf_list.get(), "bootstrap_list")
            oob_score_rf_vals       = self._parse_bool_list(self.oob_score_rf_list.get(), "oob_score_list")
            warm_start_rf_vals        = self._parse_bool_list(self.warm_start_rf_list.get(), "warm_start_list")
            ccp_alpha_rf_vals        = self._parse_float_list(self.ccp_alpha_rf_list.get(), "ccp_alpha_list")
            max_samples_rf_vals        = self._parse_optional_int_list(self.max_saples_rf_list.get(), "max_samples_list")

            alpha_lr_vals        = self._parse_float_list(self.ccp_alpha_lr_list.get(), "alpha_lr_list")

            n_neighbors_knr_vals    = self._parse_int_list(self.n_neighbors_knr_list.get(), "n_neighbors_knr_list")
            weights_knr_vals        = [x.strip() for x in self.weights_knr_list.get().split(",") if x.strip() != ""]
            algorithm_knr_vals      = [x.strip() for x in self.algorithm_knr_list.get().split(",") if x.strip() != ""]
            leaf_size_knr_vals      = self._parse_int_list(self.leaf_size_knr_list.get(), "leaf_size_knr_list")
            p_knr_vals              = self._parse_int_list(self.p_knr_list.get(), "p_knr_list")
            metric_knr_vals         = [x.strip() for x in self.metric_knr_list.get().split(",") if x.strip() != ""]
            metric_params_knr_vals  = [None if x.strip().lower() == "none" else x.strip() for x in self.metric_params_knr_list.get().split(",") if x.strip() != ""]
            n_jobs_knr_vals         = self._parse_int_list(self.n_jobs_knr_list.get(), "n_jobs_knr_list")

            weights_vr_vals = self._parse_optional_int_list(self.weights_vr_list.get(), "weights_vr_list")

            grid = list(product(
                n_estimators_rf_vals,
                criterion_rf_vals,
                max_depth_rf_vals,
                min_samples_split_rf_vals,
                min_samples_leaf_rf_vals,
                min_weight_fraction_leaf_rf_vals,
                max_features_rf_vals,
                max_leaf_nodes_rf_vals,
                min_impurity_decrease_rf_vals,
                bootstrap_rf_vals,
                oob_score_rf_vals,
                warm_start_rf_vals,
                ccp_alpha_rf_vals,
                max_samples_rf_vals,
                alpha_lr_vals,
                n_neighbors_knr_vals,
                weights_knr_vals,
                algorithm_knr_vals,
                leaf_size_knr_vals,
                p_knr_vals,
                metric_knr_vals,
                metric_params_knr_vals,
                n_jobs_knr_vals,
                weights_vr_vals
            ))

        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return




        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (VotingRegressor) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"tune_by={args['tune_by']}")
        self._start_time = time.perf_counter()
        self._start("Running nested CV (VotingRegressor)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    


    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []
                    for n_estimators_rf, criterion_rf, max_depth_rf, min_samples_split_rf, min_samples_leaf_rf, min_weight_fraction_leaf_rf, max_features_rf, max_leaf_nodes_rf, min_impurity_decrease_rf, bootstrap_rf, oob_score_rf, warm_start_rf, ccp_alpha_rf, max_samples_rf, alpha_lr, n_neighbors_knr, weights_knr, algorithm_knr, leaf_size_knr, p_knr, metric_knr, metric_params_knr, n_jobs_knr, weights_vr   in grid:
                        
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            rf = RandomForestRegressor(
                                n_estimators=n_estimators_rf,
                                criterion=criterion_rf,
                                max_depth=max_depth_rf,
                                min_samples_split=min_samples_split_rf,
                                min_samples_leaf=min_samples_leaf_rf,
                                min_weight_fraction_leaf=min_weight_fraction_leaf_rf,
                                max_features=max_features_rf,
                                max_leaf_nodes=max_leaf_nodes_rf,
                                min_impurity_decrease=min_impurity_decrease_rf,
                                bootstrap=bootstrap_rf,
                                oob_score=oob_score_rf,
                                warm_start=warm_start_rf,
                                ccp_alpha=ccp_alpha_rf,
                                max_samples=max_samples_rf
                            )

                            lr=Ridge(
                                alpha=alpha_lr
                            )

                            knr=KNeighborsRegressor(
                                n_neighbors=n_neighbors_knr,
                                weights=weights_knr,
                                algorithm=algorithm_knr,
                                leaf_size=leaf_size_knr,
                                p=p_knr,
                                metric=metric_knr,
                                metric_params=metric_params_knr,
                                n_jobs=n_jobs_knr
                            )


                            model = VotingRegressor(
                                estimators=[
                                    ('rf', rf),
                                    ('lr', lr),
                                    ('knr', knr)
                                ],
                                weights=weights_vr
                            )

                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan
                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "n_estimators_rf": n_estimators_rf,
                            "criterion_rf": criterion_rf,
                            "max_depth_rf": max_depth_rf,
                            "min_samples_split_rf": min_samples_split_rf,
                            "min_samples_leaf_rf": min_samples_leaf_rf,
                            "min_weight_fraction_leaf_rf": min_weight_fraction_leaf_rf,
                            "max_features_rf": max_features_rf,
                            "max_leaf_nodes_rf": max_leaf_nodes_rf,
                            "min_impurity_decrease_rf": min_impurity_decrease_rf,
                            "bootstrap_rf": bootstrap_rf,
                            "oob_score_rf": oob_score_rf,
                            "warm_start_rf": warm_start_rf,
                            "ccp_alpha_rf": ccp_alpha_rf,
                            "max_samples_rf": max_samples_rf,
                            "alpha_lr": alpha_lr,
                            "n_neighbors_knr": n_neighbors_knr,
                            "weights_knr": weights_knr,
                            "algorithm_knr": algorithm_knr,
                            "leaf_size_knr": leaf_size_knr,
                            "p_knr": p_knr,
                            "metric_knr": metric_knr,
                            "metric_params_knr": metric_params_knr,
                            "n_jobs_knr": n_jobs_knr,
                            "weights_vr":weights_vr,
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    rf_final = RandomForestRegressor(
                        n_estimators=best["n_estimators_rf"],
                        criterion=best["criterion_rf"],
                        max_depth=best["max_depth_rf"],
                        min_samples_split=best["min_samples_split_rf"],
                        min_samples_leaf=best["min_samples_leaf_rf"],
                        min_weight_fraction_leaf=best["min_weight_fraction_leaf_rf"],
                        max_features=best["max_features_rf"],
                        max_leaf_nodes=best["max_leaf_nodes_rf"],
                        min_impurity_decrease=best["min_impurity_decrease_rf"],
                        bootstrap=best["bootstrap_rf"],
                        oob_score=best["oob_score_rf"],
                        warm_start=best["warm_start_rf"],
                        ccp_alpha=best["ccp_alpha_rf"],
                        max_samples=best["max_samples_rf"],
                    )

                    lr_final=Ridge(
                        alpha=best["alpha_lr"]
                    )

                    knr_final=KNeighborsRegressor(
                        n_neighbors=best["n_neighbors_knr"],
                        weights=best["weights_knr"],
                        algorithm=best["algorithm_knr"],
                        leaf_size=best["leaf_size_knr"],
                        p=best["p_knr"],
                        metric=best["metric_knr"],
                        metric_params=best["metric_params_knr"],
                        n_jobs=best["n_jobs_knr"]
                    )


                    final=VotingRegressor(estimators=[('rf', rf_final), ('lr', lr_final), ('knr', knr_final)],
                                        weights=best["weights_vr"])

                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_n_estimators_rf": best["n_estimators_rf"],
                        "Best_criterion_rf": best["criterion_rf"],
                        "Best_max_depth_rf": best["max_depth_rf"],
                        "Best_min_samples_split_rf": best["min_samples_split_rf"],
                        "Best_min_samples_leaf_rf": best["min_samples_leaf_rf"],
                        "Best_min_weight_fraction_leaf_rf": best["min_weight_fraction_leaf_rf"],
                        "Best_max_features_rf": best["max_features_rf"],
                        "Best_max_leaf_nodes_rf": best["max_leaf_nodes_rf"],
                        "Best_min_impurity_decrease_rf": best["min_impurity_decrease_rf"],
                        "Best_bootstrap_rf": best["bootstrap_rf"],
                        "Best_oob_score_rf": best["oob_score_rf"],
                        "Best_warm_start_rf": best["warm_start_rf"],
                        "Best_ccp_alpha_rf": best["ccp_alpha_rf"],
                        "Best_max_samples_rf": best["max_samples_rf"],
                        "Best_alpha_lr": best["alpha_lr"],
                        "Best_n_neighbors_knr": best["n_neighbors_knr"],
                        "Best_weights_knr": best["weights_knr"],
                        "Best_algorithm_knr": best["algorithm_knr"],
                        "Best_leaf_size_knr": best["leaf_size_knr"],
                        "Best_p_knr": best["p_knr"],
                        "Best_metric_knr": best["metric_knr"],
                        "Best_metric_params_knr": best["metric_params_knr"],
                        "Best_n_jobs_knr": best["n_jobs_knr"],
                        "Best_weights_vr":best["weights_vr"]
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]


            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")


            out_path = os.path.join(outdir, f"VotingRegressor_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (VotingRegressor) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (VotingRegressor) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))



































# ==========================================================
# Nested CV Window (MLPRegressor)
# ==========================================================

class NestedMLPRegressorCvWindow(tk.Frame):
    def __init__(self, parent, app):
        super().__init__(parent)
        self.app = app

        # --- define variables FIRST ---
        self.marker_path = tk.StringVar(value="")
        self.pheno_path  = tk.StringVar(value="")
        self.output_dir  = tk.StringVar(value=os.path.join(os.getcwd(), "nested_cv_MLPRegressor_results"))

        self.target_col  = tk.StringVar(value="")
        self.use_log1p   = tk.BooleanVar(value=True)

        self.loss_list = tk.StringVar(value="squared_error, poisson")
        self.hidden_layer_sizes_list = tk.StringVar(value="50,100,200")
        self.activation_list = tk.StringVar(value="identity, logistic, tanh, relu")
        self. solver_list = tk.StringVar(value="lbfgs, sgd, adam")
        self.alpha_list = tk.StringVar(value="0.000001, 0.00001, 0.0001, 0.001")
        self.batch_size_list = tk.StringVar(value="32, 64, 128, 256")
        self.learning_rate_list = tk.StringVar(value="constant, invscaling, adaptive")
        self.learning_rate_init_list = tk.StringVar(value="1e-4, 1e-3")
        self.power_t_list = tk.StringVar(value="0.25, 0.5, 0.75")
        self.max_iter_list = tk.StringVar(value="200,300")
        self.shuffle_list = tk.StringVar(value="True, False")
        self.tol_list = tk.StringVar(value="1e-4, 1e-3, 1e-2")
        self.warm_start_list = tk.StringVar(value="True, False")
        self.momentum_list = tk.StringVar(value="0.25, 0.5, 0.75")
        self.nesterovs_momentum_list = tk.StringVar(value="True, False")
        self.early_stopping_list = tk.StringVar(value="True, False")
        self.validation_fraction_list = tk.StringVar(value="0.25, 0.5, 0.75")
        self.beta_1_list = tk.StringVar(value="0.1, 0.5, 0.9")
        self.beta_2_list = tk.StringVar(value="0.1, 0.5, 0.9")
        self.epsilon_list = tk.StringVar(value="1e-10, 1e-8, 1e-6")
        self.n_iter_no_change_list = tk.StringVar(value="10,20,40,60")
        self.max_fun_list = tk.StringVar(value="10000,15000")



        self.n_cycles    = tk.IntVar(value=10)
        self.outer_folds = tk.IntVar(value=5)
        self.inner_folds = tk.IntVar(value=4)
        self.outer_seed  = tk.IntVar(value=1000)
        self.inner_seed  = tk.IntVar(value=2000)

        self.tune_by     = tk.StringVar(value="MAPE")  # MAPE / MSPE / PA

        # threading
        self._worker_thread = None
        self._q = queue.Queue()

        # --- build UI LAST ---
        self._build_ui()

    # ---------------- UI ----------------
    def _build_ui(self):
        pad = {"padx": 10, "pady": 6}

        files = ttk.LabelFrame(self, text="Input files")
        files.pack(fill="x", **pad)
        files.columnconfigure(1, weight=1)

        ttk.Label(files, text="Markers file (rows=GID, cols=markers):").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.marker_path, width=80).grid(row=0, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_markers).grid(row=0, column=2, **pad)

        ttk.Label(files, text="Phenotype file (rows=GID):").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.pheno_path, width=80).grid(row=1, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_pheno).grid(row=1, column=2, **pad)

        ttk.Label(files, text="Output folder:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(files, textvariable=self.output_dir, width=80).grid(row=2, column=1, sticky="we", **pad)
        ttk.Button(files, text="Browse", command=self._browse_outdir).grid(row=2, column=2, **pad)

        trait = ttk.LabelFrame(self, text="Target")
        trait.pack(fill="x", **pad)

        ttk.Label(trait, text="Target phenotype column:").grid(row=0, column=0, sticky="w", **pad)
        self.target_combo = ttk.Combobox(trait, textvariable=self.target_col, state="readonly", width=45, values=[])
        self.target_combo.grid(row=0, column=1, sticky="w", **pad)

        ttk.Checkbutton(trait, text="Apply log1p(y)", variable=self.use_log1p).grid(row=0, column=2, sticky="w", **pad)

        hp = ttk.LabelFrame(self, text="MLPRegressor hyperparameters (inner tuning grid)")
        hp.pack(fill="x", **pad)

        
        
        ttk.Label(hp, text="Hidden Layers Sizes list:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.hidden_layer_sizes_list, width=60).grid(row=0, column=1, sticky="w", **pad)

        tk.Label(hp, text="Activation Function list:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.activation_list, width=60).grid(row=1, column=1, sticky="w", **pad)

        tk.Label(hp, text="Solver list:").grid(row=2, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.solver_list, width=60).grid(row=2, column=1, sticky="w", **pad)

        tk.Label(hp, text="Alpha list:").grid(row=3, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.alpha_list, width=60).grid(row=3, column=1, sticky="w", **pad)

        tk.Label(hp, text="Batch Size list:").grid(row=4, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.batch_size_list, width=60).grid(row=4, column=1, sticky="w", **pad)

        tk.Label(hp, text="Learning Rate list:").grid(row=5, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.learning_rate_list, width=60).grid(row=5, column=1, sticky="w", **pad)

        tk.Label(hp, text="Learning Rate Init list:").grid(row=6, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.learning_rate_init_list, width=60).grid(row=6, column=1, sticky="w", **pad)

        tk.Label(hp, text="Power T list:").grid(row=7, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.power_t_list, width=60).grid(row=7, column=1, sticky="w", **pad)

        tk.Label(hp, text="Max Iter list:").grid(row=8, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_iter_list, width=60).grid(row=8, column=1, sticky="w", **pad)

        tk.Label(hp, text="Shuffle list:").grid(row=9, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.shuffle_list, width=60).grid(row=9, column=1, sticky="w", **pad)

        tk.Label(hp, text="Tollerance list:").grid(row=10, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.tol_list, width=60).grid(row=10, column=1, sticky="w", **pad)

        tk.Label(hp, text="Warm Start list:").grid(row=11, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.warm_start_list, width=60).grid(row=11, column=1, sticky="w", **pad)

        tk.Label(hp, text="Momentum list:").grid(row=12, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.momentum_list, width=60).grid(row=12, column=1, sticky="w", **pad)

        tk.Label(hp, text="Nesterovs Momentum list:").grid(row=13, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.nesterovs_momentum_list, width=60).grid(row=13, column=1, sticky="w", **pad)

        tk.Label(hp, text="Early Stopping list:").grid(row=14, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.early_stopping_list, width=60).grid(row=14, column=1, sticky="w", **pad)

        tk.Label(hp, text="Validation Fraction list:").grid(row=15, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.validation_fraction_list, width=60).grid(row=15, column=1, sticky="w", **pad)

        tk.Label(hp, text="Beta 1 list:").grid(row=16, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.beta_1_list, width=60).grid(row=16, column=1, sticky="w", **pad)

        tk.Label(hp, text="Beta 2 list:").grid(row=17, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.beta_2_list, width=60).grid(row=17, column=1, sticky="w", **pad)

        tk.Label(hp, text="Epsilon list:").grid(row=18, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.epsilon_list, width=60).grid(row=18, column=1, sticky="w", **pad)

        tk.Label(hp, text="N Iteration with no changes list:").grid(row=19, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.n_iter_no_change_list, width=60).grid(row=19, column=1, sticky="w", **pad)

        tk.Label(hp, text="Max Fun list:").grid(row=20, column=0, sticky="w", **pad)
        ttk.Entry(hp, textvariable=self.max_fun_list, width=60).grid(row=20, column=1, sticky="w", **pad)

        



        cv = ttk.LabelFrame(self, text="Nested CV settings")
        cv.pack(fill="x", **pad)

        ttk.Label(cv, text="Cycles:").grid(row=0, column=0, sticky="w", **pad)
        ttk.Spinbox(cv, from_=1, to=100, textvariable=self.n_cycles, width=6).grid(row=0, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Outer folds:").grid(row=0, column=2, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.outer_folds, width=6).grid(row=0, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Inner folds:").grid(row=0, column=4, sticky="w", **pad)
        ttk.Spinbox(cv, from_=2, to=20, textvariable=self.inner_folds, width=6).grid(row=0, column=5, sticky="w", **pad)

        ttk.Label(cv, text="Outer seed base:").grid(row=1, column=0, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.outer_seed, width=8).grid(row=1, column=1, sticky="w", **pad)

        ttk.Label(cv, text="Inner seed base:").grid(row=1, column=2, sticky="w", **pad)
        ttk.Entry(cv, textvariable=self.inner_seed, width=8).grid(row=1, column=3, sticky="w", **pad)

        ttk.Label(cv, text="Tune by:").grid(row=1, column=4, sticky="w", **pad)
        ttk.Combobox(cv, textvariable=self.tune_by, state="readonly",
                     values=["MAPE", "MSPE", "PA"], width=10).grid(row=1, column=5, sticky="w", **pad)

        runbar = ttk.Frame(self)
        runbar.pack(fill="x", padx=10, pady=(4, 8))

        self.run_btn = ttk.Button(runbar, text="Run Nested CV (MLPRegressor)", command=self._run_clicked)
        self.run_btn.pack(side="left")

        self.progress = ttk.Progressbar(runbar, mode="indeterminate", length=240)
        self.progress.pack(side="left", padx=10)

        self.status = tk.StringVar(value="Ready.")
        ttk.Label(runbar, textvariable=self.status).pack(side="left")

        out = ttk.LabelFrame(self, text="Log")
        out.pack(fill="both", expand=True, **pad)

        self.txt = tk.Text(out, height=18, wrap="word")
        self.txt.pack(fill="both", expand=True, padx=10, pady=10)

    # ---------------- helpers ----------------
    def _log(self, msg: str):
        self.txt.insert(tk.END, msg + "\n")
        self.txt.see(tk.END)

    def _start(self, msg="Running..."):
        self.status.set(msg)
        self.run_btn.config(state="disabled")
        self.progress.start(12)

    def _stop(self, msg="Ready."):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.status.set(msg)

    def _browse_markers(self):
        p = filedialog.askopenfilename(
            title="Select markers file",
            filetypes=[("Text/CSV/TSV", "*.txt *.tsv *.csv"), ("All files", "*.*")]
        )
        if p:
            self.marker_path.set(p)

    def _browse_pheno(self):
        p = filedialog.askopenfilename(
            title="Select phenotype file",
            filetypes=[("CSV/TSV/Text", "*.csv *.txt *.tsv"), ("All files", "*.*")]
        )
        if not p:
            return
        self.pheno_path.set(p)
        try:
            ph = self._read_table(p)
            cols = list(ph.columns)
            self.target_combo["values"] = cols
            if cols and not self.target_col.get():
                self.target_col.set(cols[0])
            self._log(f"Loaded phenotype traits: {len(cols)}")
        except Exception as e:
            messagebox.showerror("Phenotype load failed", str(e))

    def _browse_outdir(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)

    def _read_table(self, path: str) -> pd.DataFrame:
        ext = os.path.splitext(path)[1].lower()
        sep = "\t" if ext in [".tsv", ".txt"] else ","
        df = pd.read_csv(path, sep=sep, dtype=str)
        if df.shape[1] < 2:
            raise ValueError("File must have at least 2 columns (ID + data).")
        id_col = df.columns[0]
        df[id_col] = df[id_col].astype(str)
        df = df.set_index(id_col)
        df = df.apply(pd.to_numeric, errors="coerce")
        df.index = df.index.astype(str)
        return df
    
    def _parse_int_list(self, s: str, name: str):
        try:
            vals = [int(float(x.strip())) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated integers, e.g. 2,5,10,20")

    def _parse_float_list(self, s: str, name: str):
        try:
            vals = [float(x.strip()) for x in s.split(",") if x.strip() != ""]
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated numbers, e.g. 0.001,0.01,0.1,1")
        

    def _parse_bool_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if x in {"true", "1", "yes"}:
                    vals.append(True)
                elif x in {"false", "0", "no"}:
                    vals.append(False)
                else:
                    raise ValueError
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(f"Invalid {name}. Use comma-separated booleans, e.g. True,False")


    def _parse_optional_int_list(self, s: str, name: str):
        try:
            vals = []
            for x in s.split(","):
                x = x.strip().lower()
                if not x:
                    continue
                if x == "none":
                    vals.append(None)
                else:
                    vals.append(int(x))
            if not vals:
                raise ValueError
            return vals
        except Exception:
            raise ValueError(
                f"Invalid {name}. Use comma-separated integers or None, e.g. None,5,10,20"
            )



    def _parse_max_features_list(self, s: str, name: str):
        vals = []
        for x in s.split(","):
            t = x.strip()
            if not t:
                continue

            low = t.lower()

            if low == "none":
                vals.append(None)
            elif low in {"sqrt", "log2"}:
                vals.append(low)
            else:
                # try int, then float
                try:
                    vals.append(int(float(t)))
                except ValueError:
                    try:
                        vals.append(float(t))
                    except ValueError:
                        raise ValueError(
                            f"Invalid {name}: '{t}'. "
                            "Allowed: None, sqrt, log2, int, float"
                        )

        if not vals:
            raise ValueError(f"{name} is empty.")

        return vals



    # ---------------- run (threaded) ----------------
    def _run_clicked(self):
        if self._worker_thread is not None and self._worker_thread.is_alive():
            return

        mp = self.marker_path.get().strip()
        pp = self.pheno_path.get().strip()
        if not os.path.exists(mp):
            messagebox.showerror("Missing input", "Please select a markers file.")
            return
        if not os.path.exists(pp):
            messagebox.showerror("Missing input", "Please select a phenotype file.")
            return
        if not self.target_col.get().strip():
            messagebox.showerror("Missing target", "Please choose a target phenotype column.")
            return

        outdir = self.output_dir.get().strip()
        os.makedirs(outdir, exist_ok=True)

        try:
           #numbers n_estimators_vals      = self._parse_int_list(self.n_estimators_list.get(), "n_estimators_list")
            #text criterion_vals        = [x.strip() for x in self.criterion_list.get().split(",") if x.strip() != ""]
            # tex and numbers max_depth_vals        = self._parse_optional_int_list(self.max_depth_list.get(), "max_depth_list")
            # float min_weight_fraction_leaf_vals = self._parse_float_list(self.min_weight_fraction_leaf_list.get(), "min_weight_fraction_leaf_list")
            # bool bootstrap_vals        = self._parse_bool_list(self.bootstrap_list.get(), "bootstrap_list")
            
        # Parse each UI field into typed Python lists
            hidden_layer_sizes_vals = self._parse_int_list(self.hidden_layer_sizes_list.get(), "hidden_layer_sizes_list")
            activation_vals = [x.strip() for x in self.activation_list.get().split(",") if x.strip() != ""]
            solver_vals = [x.strip() for x in self.solver_list.get().split(",") if x.strip() != ""]
            alpha_vals = self._parse_float_list(self.alpha_list.get(), "alpha_list")
            batch_size_vals = self._parse_int_list(self.batch_size_list.get(), "batch_size_list")
            learning_rate_vals = [x.strip() for x in self.learning_rate_list.get().split(",") if x.strip() != ""]
            learning_rate_init_vals = self._parse_float_list(self.learning_rate_init_list.get(), "learning_rate_init_list")
            power_t_vals = self._parse_float_list(self.power_t_list.get(), "power_t_list")
            max_iter_vals = self._parse_int_list(self.max_iter_list.get(), "max_iter_list")
            shuffle_vals = self._parse_bool_list(self.shuffle_list.get(), "shuffle_list")
            tol_vals = self._parse_float_list(self.tol_list.get(), "tol_list")
            warm_start_vals = self._parse_bool_list(self.warm_start_list.get(), "warm_start_list")
            momentum_vals = self._parse_float_list(self.momentum_list.get(), "momentum_list")
            nesterovs_momentum_vals = self._parse_bool_list(self.nesterovs_momentum_list.get(), "nesterovs_momentum_list")
            early_stopping_vals = self._parse_bool_list(self.early_stopping_list.get(), "early_stopping_list")
            validation_fraction_vals = self._parse_float_list(self.validation_fraction_list.get(), "validation_fraction_list")
            beta_1_vals = self._parse_float_list(self.beta_1_list.get(), "beta_1_list")
            beta_2_vals = self._parse_float_list(self.beta_2_list.get(), "beta_2_list")
            epsilon_vals = self._parse_float_list(self.epsilon_list.get(), "epsilon_list")
            n_iter_no_change_vals = self._parse_int_list(self.n_iter_no_change_list.get(), "n_iter_no_change_list")
            max_fun_vals = self._parse_int_list(self.max_fun_list.get(), "max_fun_list")



            grid = list(product(
                hidden_layer_sizes_vals,
                activation_vals,
                solver_vals,
                alpha_vals,
                batch_size_vals,
                learning_rate_vals,
                learning_rate_init_vals,
                power_t_vals,
                max_iter_vals,
                shuffle_vals,
                tol_vals,
                warm_start_vals,
                momentum_vals,
                nesterovs_momentum_vals,
                early_stopping_vals,
                validation_fraction_vals,
                beta_1_vals,
                beta_2_vals,
                epsilon_vals,
                n_iter_no_change_vals,
                max_fun_vals
            ))

        except Exception as e:
            messagebox.showerror("Hyperparameter error", str(e))
            return




        args = dict(
            marker_path=mp,
            pheno_path=pp,
            outdir=outdir,
            target_col=self.target_col.get().strip(),
            log1p=self.use_log1p.get(),
            grid=grid,
            n_cycles=int(self.n_cycles.get()),
            outer_folds=int(self.outer_folds.get()),
            inner_folds=int(self.inner_folds.get()),
            outer_seed=int(self.outer_seed.get()),
            inner_seed=int(self.inner_seed.get()),
            tune_by=self.tune_by.get().strip(),
        )

        self._log("--------------------------------------------------")
        self._log("Nested CV (MLPRegressor) started")
        self._log(f"Target: {args['target_col']}   | log1p: {args['log1p']}")
        self._log(f"Cycles={args['n_cycles']}, outer={args['outer_folds']}, inner={args['inner_folds']}")
        self._log(f"Grid size: {len(grid)}")
        self._log(f"tune_by={args['tune_by']}")
        self._start_time = time.perf_counter()
        self._start("Running nested CV (MLPRegressor)...")

        self._worker_thread = threading.Thread(target=self._worker, kwargs=args, daemon=True)
        self._worker_thread.start()
        self.after(200, self._poll)

    


    def _worker(self, **args):
        try:
            markers = self._read_table(args["marker_path"])
            pheno = self._read_table(args["pheno_path"])

            if markers.index.duplicated().any():
                markers = markers[~markers.index.duplicated(keep="first")]

            common = sorted(set(markers.index).intersection(pheno.index))
            if len(common) < 10:
                raise ValueError(f"Too few common IDs between markers and pheno: {len(common)}")

            markers = markers.reindex(common)
            pheno = pheno.reindex(common)

            target_col = args["target_col"]
            if target_col not in pheno.columns:
                raise ValueError(f"Target column '{target_col}' not found in phenotype file.")

            valid = pheno[target_col].notna()
            X = markers.loc[valid]
            y = pheno.loc[valid, target_col].values

            if len(y) < args["outer_folds"] * 2:
                raise ValueError(f"Not enough samples after dropping NaNs: n={len(y)}")

            if args["log1p"]:
                y = np.log1p(y)

            grid = args["grid"]
            all_rows = []

            for cycle in range(1, args["n_cycles"] + 1):
                outer_cv = KFold(
                    n_splits=args["outer_folds"],
                    shuffle=True,
                    random_state=args["outer_seed"] + cycle
                )

                for outer_fold, (tr_idx, te_idx) in enumerate(outer_cv.split(X), start=1):
                    X_train_outer = X.iloc[tr_idx]
                    X_test_outer = X.iloc[te_idx]
                    y_train_outer = y[tr_idx]
                    y_test_outer = y[te_idx]

                    scaler = StandardScaler()
                    X_train_outer_s = scaler.fit_transform(X_train_outer)
                    X_test_outer_s = scaler.transform(X_test_outer)

                    inner_cv = KFold(
                        n_splits=args["inner_folds"],
                        shuffle=True,
                        random_state=args["inner_seed"] + cycle
                    )

                    tuning_rows = []
                    for hidden_layer_sizes, activation, solver, alpha, batch_size, learning_rate, learning_rate_init, power_t, max_iter, shuffle, tol, warm_start, momentum, nesterovs_momentum, early_stopping, validation_fraction, beta_1, beta_2, epsilon, n_iter_no_change, max_fun in grid:
                        
                        fold_pas, fold_mspes, fold_mapes = [], [], []

                        for itr_idx, iva_idx in inner_cv.split(X_train_outer_s):
                            Xtr = X_train_outer_s[itr_idx]
                            Xva = X_train_outer_s[iva_idx]
                            ytr = y_train_outer[itr_idx]
                            yva = y_train_outer[iva_idx]

                            model = MLPRegressor(
                                hidden_layer_sizes=hidden_layer_sizes,
                                activation=activation,
                                solver=solver,
                                alpha=alpha,
                                batch_size=batch_size,
                                learning_rate=learning_rate,
                                learning_rate_init=learning_rate_init,
                                power_t=power_t,
                                max_iter=max_iter,
                                shuffle=shuffle,
                                tol=tol,
                                warm_start=warm_start,
                                momentum=momentum,
                                nesterovs_momentum=nesterovs_momentum,
                                early_stopping=early_stopping,
                                validation_fraction=validation_fraction,
                                beta_1=beta_1,
                                beta_2=beta_2,
                                epsilon=epsilon,
                                n_iter_no_change=n_iter_no_change,
                                max_fun=max_fun
                            )

                            model.fit(Xtr, ytr)
                            pred = model.predict(Xva)

                            try:
                                pa = pearsonr(yva, pred)[0]
                            except Exception:
                                pa = np.nan
                            mspe = mean_squared_error(yva, pred)
                            mape = mean_absolute_error(yva, pred)

                            fold_pas.append(pa)
                            fold_mspes.append(mspe)
                            fold_mapes.append(mape)

                        tuning_rows.append({
                            "hidden_layer_sizes":hidden_layer_sizes,
                            "activation":activation,
                            "solver":solver,
                            "alpha":alpha,
                            "batch_size":batch_size,
                            "learning_rate":learning_rate,
                            "learning_rate_init":learning_rate_init,
                            "power_t":power_t,
                            "max_iter":max_iter,
                            "shuffle":shuffle,
                            "tol":tol,
                            "warm_start":warm_start,
                            "momentum":momentum,
                            "nesterovs_momentum":nesterovs_momentum,
                            "early_stopping":early_stopping,
                            "validation_fraction":validation_fraction,
                            "beta_1":beta_1,
                            "beta_2":beta_2,
                            "epsilon":epsilon,
                            "n_iter_no_change":n_iter_no_change,
                            "max_fun":max_fun,
                            "PA": float(np.nanmean(fold_pas)),
                            "MSPE": float(np.mean(fold_mspes)),
                            "MAPE": float(np.mean(fold_mapes))
                        })

                    tuning_df = pd.DataFrame(tuning_rows)

                    tune_by = args["tune_by"]
                    if tune_by == "MAPE":
                        best = tuning_df.loc[tuning_df["MAPE"].idxmin()]
                    elif tune_by == "MSPE":
                        best = tuning_df.loc[tuning_df["MSPE"].idxmin()]
                    else:
                        best = tuning_df.loc[tuning_df["PA"].idxmax()]

                    final = MLPRegressor(
                                hidden_layer_sizes=best["hidden_layer_sizes"],
                                activation=best["activation"],
                                solver=best["solver"],
                                alpha=best["alpha"],
                                batch_size=best["batch_size"],
                                learning_rate=best["learning_rate"],
                                learning_rate_init=best["learning_rate_init"],
                                power_t=best["power_t"],
                                max_iter=best["max_iter"],
                                shuffle=best["shuffle"],
                                tol=best["tol"],
                                warm_start=best["warm_start"],
                                momentum=best["momentum"],
                                nesterovs_momentum=best["nesterovs_momentum"],
                                early_stopping=best["early_stopping"],
                                validation_fraction=best["validation_fraction"],
                                beta_1=best["beta_1"],
                                beta_2=best["beta_2"],
                                epsilon=best["epsilon"],
                                n_iter_no_change=best["n_iter_no_change"],
                                max_fun=best["max_fun"]
                    )


                    final.fit(X_train_outer_s, y_train_outer)

                    y_train_pred = final.predict(X_train_outer_s)
                    y_test_pred = final.predict(X_test_outer_s)

                    try:
                        pa_train = pearsonr(y_train_outer, y_train_pred)[0]
                    except Exception:
                        pa_train = np.nan
                    try:
                        pa_test = pearsonr(y_test_outer, y_test_pred)[0]
                    except Exception:
                        pa_test = np.nan

                    mspe = mean_squared_error(y_test_outer, y_test_pred)
                    mape = mean_absolute_error(y_test_outer, y_test_pred)

                    all_rows.append({
                        "Cycle": cycle,
                        "OuterFold": outer_fold,
                        "n_train": int(len(tr_idx)),
                        "n_test": int(len(te_idx)),
                        "PA_train": pa_train,
                        "PA_test": pa_test,
                        "MSPE": mspe,
                        "MAPE": mape,
                        "Best_hidden_layer_sizes":best["hidden_layer_sizes"],
                        "Best_activation":best["activation"],
                        "Best_solver":best["solver"],
                        "Best_alpha":best["alpha"],
                        "Best_batch_size":best["batch_size"],
                        "Best_learning_rate":best["learning_rate"],
                        "Best_learning_rate_init":best["learning_rate_init"],
                        "Best_power_t":best["power_t"],
                        "Best_max_iter":best["max_iter"],
                        "Best_shuffle":best["shuffle"],
                        "Best_tol":best["tol"],
                        "Best_warm_start":best["warm_start"],
                        "Best_momentum":best["momentum"],
                        "Best_nesterovs_momentum":best["nesterovs_momentum"],
                        "Best_early_stopping":best["early_stopping"],
                        "Best_validation_fraction":best["validation_fraction"],
                        "Best_beta_1":best["beta_1"],
                        "Best_beta_2":best["beta_2"],
                        "Best_epsilon":best["epsilon"],
                        "Best_n_iter_no_change":best["n_iter_no_change"],
                        "Best_max_fun":best["max_fun"]
                    })

            outdir = args["outdir"]
            results_df = pd.DataFrame(all_rows)
            results_df["PA_gap"] = results_df["PA_train"] - results_df["PA_test"]

            trait = args["target_col"]

            # make it filesystem-safe
            trait_safe = trait.replace(" ", "_").replace("/", "_")

            out_path = os.path.join(outdir, f"MLPRegressor_nestedCV_results_{trait_safe}.csv")
            results_df.to_csv(out_path, index=False)

            self._q.put(("ok", results_df, out_path))
        except Exception as e:
            self._q.put(("err", str(e)))

    def _poll(self):
        try:
            msg = self._q.get_nowait()
        except queue.Empty:
            self.after(200, self._poll)
            return

        if msg[0] == "err":
            self._stop("Ready.")
            messagebox.showerror("Nested CV (MLPRegressor) failed", msg[1])
            self._log(f"ERROR: {msg[1]}")
            return

        _, results_df, out_path = msg
        self._stop("Done.")

        elapsed = time.perf_counter() - self._start_time
        h, rem = divmod(elapsed, 3600)
        m, s = divmod(rem, 60)

        self._log("✅ Nested CV (MLPRegressor) finished.")
        self._log(f"Saved: {out_path}")
        self._log(f"⏱ Runtime: {int(h):02d}:{int(m):02d}:{s:05.2f}")
        self._log("")
        self._log("Head of results:")
        self._log(results_df.head(10).to_string(index=False))





if __name__ == "__main__":
    # Run with:
    #   Windows: python marker_filter_gui.py
    #   Mac/Linux: python3 marker_filter_gui.py
    app = MarkerFilterApp()
    app.mainloop()
