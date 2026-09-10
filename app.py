```python
import math
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt


# ============================================================
# PCB CORROSION LIFETIME PREDICTION SYSTEM
# Based on:
# "Investigation of Oxidation and Corrosion Behaviour
#  in Panel Circuit Board"
# ============================================================


# ------------------------------------------------------------
# 1. PROJECT CONSTANTS
# ------------------------------------------------------------

# Copper properties used in the project
CU_EQUIVALENT_WEIGHT = 31.77      # g/equiv
CU_DENSITY = 8.96                 # g/cm³

# ASTM G102/Faraday-law conversion factor
# Corrosion rate in mm/year when Icorr is in µA/cm²
ASTM_FACTOR = 3.27e-3

# Experimental electrochemical results from the project
CONTROL_ICORR = 1.47              # µA/cm² (upper-bound estimate)
CORRODED_ICORR = 19.0             # µA/cm²

CONTROL_CORROSION_RATE = 0.017    # mm/year (upper bound)
CORRODED_CORROSION_RATE = 0.220   # mm/year

CONTROL_EOCP = -0.452             # V vs SCE
CORRODED_EOCP = -0.465            # V vs SCE
CORRODED_ECORR = -0.555           # V vs SCE

# Tafel slopes for corroded PCB
BETA_ANODIC = 104.0               # mV/decade
BETA_CATHODIC = 351.0             # mV/decade

# Mechanical results from project
CONTROL_UTS = 61.11               # MPa
CORRODED_UTS = 35.66              # MPa

CONTROL_MODULUS = 806.43          # MPa
CORRODED_MODULUS = 307.35         # MPa

CONTROL_ELONGATION = 19.8         # %
CORRODED_ELONGATION = 25.3        # %


# ------------------------------------------------------------
# 2. CALCULATION FUNCTIONS
# ------------------------------------------------------------

def calculate_corrosion_rate(icorr, equivalent_weight, density):
    """
    Calculate corrosion rate in mm/year.

    Equation used in the project:

        CR = 3.27 × 10^-3 × Icorr × EW / density

    Icorr       = corrosion current density in µA/cm²
    EW          = equivalent weight in g/equiv
    density     = density in g/cm³
    """

    return (
        ASTM_FACTOR
        * icorr
        * equivalent_weight
        / density
    )


def mm_year_to_mpy(mm_year):
    """
    Convert mm/year to mils per year (mpy).
    """

    return mm_year * 39.3701


def percentage_change(new_value, old_value):
    """
    Calculate percentage change.
    """

    if old_value == 0:
        return np.nan

    return ((new_value - old_value) / abs(old_value)) * 100


def lifetime_constant_rate(
    thickness_mm,
    corrosion_rate,
    allowable_loss_fraction
):
    """
    Estimate lifetime assuming constant corrosion rate.

    Lifetime = allowable thickness loss / corrosion rate
    """

    if corrosion_rate <= 0:
        return np.inf

    allowable_loss = (
        thickness_mm * allowable_loss_fraction
    )

    return allowable_loss / corrosion_rate


def fit_power_law(time, corrosion_rate):
    """
    Fit:

        CR = a × t^b

    using logarithmic linear regression.
    """

    time = np.asarray(time, dtype=float)
    corrosion_rate = np.asarray(
        corrosion_rate,
        dtype=float
    )

    valid = (
        (time > 0)
        & (corrosion_rate > 0)
        & np.isfinite(time)
        & np.isfinite(corrosion_rate)
    )

    time = time[valid]
    corrosion_rate = corrosion_rate[valid]

    if len(time) < 3:
        raise ValueError(
            "At least 3 valid time/corrosion-rate "
            "measurements are required."
        )

    log_time = np.log(time)
    log_rate = np.log(corrosion_rate)

    slope, intercept = np.polyfit(
        log_time,
        log_rate,
        1
    )

    b = slope
    a = np.exp(intercept)

    predicted = a * time ** b

    ss_res = np.sum(
        (corrosion_rate - predicted) ** 2
    )

    ss_tot = np.sum(
        (corrosion_rate - np.mean(corrosion_rate)) ** 2
    )

    if ss_tot == 0:
        r_squared = np.nan
    else:
        r_squared = 1 - (ss_res / ss_tot)

    return a, b, r_squared


def power_law_corrosion_rate(a, b, time):
    """
    Calculate corrosion rate from:

        CR = a × t^b
    """

    return a * np.asarray(time) ** b


def power_law_lifetime(
    a,
    b,
    thickness_mm,
    allowable_fraction
):
    """
    Calculate lifetime by integrating:

        CR = a × t^b

    Thickness loss:

        L(t) = a × t^(b+1)/(b+1)

    """

    if a <= 0:
        return np.nan

    if b <= -1:
        return np.nan

    allowable_loss = (
        thickness_mm
        * allowable_fraction
    )

    lifetime = (
        allowable_loss
        * (b + 1)
        / a
    ) ** (1 / (b + 1))

    return lifetime


# ------------------------------------------------------------
# 3. PAGE CONFIGURATION
# ------------------------------------------------------------

st.set_page_config(
    page_title="PCB Corrosion Lifetime Prediction",
    page_icon="⚙️",
    layout="wide"
)


# ------------------------------------------------------------
# 4. APPLICATION HEADER
# ------------------------------------------------------------

st.title(
    "PCB Corrosion Lifetime Prediction System"
)

st.write(
    """
    **Python-based computational system for analysing
    and predicting corrosion behaviour in printed
    circuit board (PCB) assemblies.**
    """
)

st.info(
    """
    This application is based on the experimental data,
    equations and methodology contained in the project
    "Investigation of Oxidation and Corrosion Behaviour
    in Panel Circuit Board".
    """
)


# ------------------------------------------------------------
# 5. SIDEBAR
# ------------------------------------------------------------

st.sidebar.header("Experimental Conditions")

salt_concentration = st.sidebar.number_input(
    "Salt concentration (wt.% NaCl)",
    min_value=0.0,
    value=5.0,
    step=0.5
)

salt_temperature = st.sidebar.number_input(
    "Salt-spray temperature (°C)",
    min_value=0.0,
    value=35.0,
    step=1.0
)

electrochemical_temperature = (
    st.sidebar.number_input(
        "Electrochemical temperature (°C)",
        min_value=0.0,
        value=25.0,
        step=1.0
    )
)

electrode_area = st.sidebar.number_input(
    "Exposed electrode area (cm²)",
    min_value=0.01,
    value=1.0,
    step=0.1
)

st.sidebar.divider()

st.sidebar.subheader("Copper Properties")

equivalent_weight = st.sidebar.number_input(
    "Copper equivalent weight (g/equiv)",
    min_value=0.0001,
    value=CU_EQUIVALENT_WEIGHT,
    step=0.01
)

density = st.sidebar.number_input(
    "Copper density (g/cm³)",
    min_value=0.0001,
    value=CU_DENSITY,
    step=0.01
)


# ------------------------------------------------------------
# 6. APPLICATION TABS
# ------------------------------------------------------------

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Project Results",
        "Corrosion Calculator",
        "Lifetime Prediction",
        "Time-Series Model"
    ]
)


# ============================================================
# TAB 1 — PROJECT RESULTS
# ============================================================

with tab1:

    st.header(
        "Experimental Results from the Project"
    )

    results = pd.DataFrame(
        {
            "Parameter": [
                "Open Circuit Potential, Eocp (V vs SCE)",
                "Corrosion Potential, Ecorr (V vs SCE)",
                "Corrosion Current Density, Icorr (µA/cm²)",
                "Corrosion Rate (mm/year)",
                "Corrosion Rate (mpy)",
                "Ultimate Tensile Strength (MPa)",
                "Young's Modulus (MPa)",
                "Elongation (%)"
            ],

            "Control / New PCB": [
                CONTROL_EOCP,
                "N/A",
                f"≤ {CONTROL_ICORR:.2f}",
                f"≤ {CONTROL_CORROSION_RATE:.3f}",
                f"≤ {mm_year_to_mpy(CONTROL_CORROSION_RATE):.2f}",
                CONTROL_UTS,
                CONTROL_MODULUS,
                CONTROL_ELONGATION
            ],

            "Corroded PCB": [
                CORRODED_EOCP,
                CORRODED_ECORR,
                CORRODED_ICORR,
                CORRODED_CORROSION_RATE,
                mm_year_to_mpy(
                    CORRODED_CORROSION_RATE
                ),
                CORRODED_UTS,
                CORRODED_MODULUS,
                CORRODED_ELONGATION
            ]
        }
    )

    st.dataframe(
        results,
        use_container_width=True,
        hide_index=True
    )

    st.subheader(
        "Corrosion Comparison"
    )

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Control Icorr",
        "≤ 1.47 µA/cm²"
    )

    col2.metric(
        "Corroded Icorr",
        "19.0 µA/cm²"
    )

    ratio = (
        CORRODED_ICORR
        / CONTROL_ICORR
    )

    col3.metric(
        "Icorr increase",
        f"{ratio:.1f} ×"
    )

    st.subheader(
        "Mechanical Degradation"
    )

    uts_loss = abs(
        percentage_change(
            CORRODED_UTS,
            CONTROL_UTS
        )
    )

    modulus_loss = abs(
        percentage_change(
            CORRODED_MODULUS,
            CONTROL_MODULUS
        )
    )

    c1, c2 = st.columns(2)

    c1.metric(
        "UTS reduction",
        f"{uts_loss:.1f}%"
    )

    c2.metric(
        "Young's modulus reduction",
        f"{modulus_loss:.1f}%"
    )

    st.warning(
        """
        The project notes that the modulus values should mainly
        be interpreted comparatively because specimen geometry,
        clamping and fixture compliance introduced uncertainty.
        """
    )


# ============================================================
# TAB 2 — CORROSION CALCULATOR
# ============================================================

with tab2:

    st.header(
        "Corrosion Rate Calculator"
    )

    st.write(
        """
        Calculate corrosion rate from corrosion current
        density using the equation used in the project.
        """
    )

    st.latex(
        r"""
        CR =
        3.27\times10^{-3}
        \frac{i_{corr}\times EW}{\rho}
        """
    )

    col1, col2 = st.columns(2)

    with col1:

        icorr_input = st.number_input(
            "Icorr (µA/cm²)",
            min_value=0.000001,
            value=19.0,
            step=0.1,
            format="%.6f"
        )

        ew_input = st.number_input(
            "Equivalent Weight (g/equiv)",
            min_value=0.000001,
            value=equivalent_weight,
            step=0.01
        )

    with col2:

        density_input = st.number_input(
            "Density (g/cm³)",
            min_value=0.000001,
            value=density,
            step=0.01
        )

        baseline_rate = st.number_input(
            "Baseline corrosion rate (mm/year)",
            min_value=0.000001,
            value=CONTROL_CORROSION_RATE,
            step=0.001,
            format="%.6f"
        )

    calculated_rate = calculate_corrosion_rate(
        icorr_input,
        ew_input,
        density_input
    )

    calculated_mpy = mm_year_to_mpy(
        calculated_rate
    )

    change = percentage_change(
        calculated_rate,
        baseline_rate
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Corrosion Rate",
        f"{calculated_rate:.4f} mm/year"
    )

    col2.metric(
        "Corrosion Rate",
        f"{calculated_mpy:.2f} mpy"
    )

    col3.metric(
        "Change from Baseline",
        f"{change:.1f}%"
    )

    st.subheader(
        "Verification with Project Data"
    )

    verification = calculate_corrosion_rate(
        CORRODED_ICORR,
        CU_EQUIVALENT_WEIGHT,
        CU_DENSITY
    )

    st.write(
        f"""
        For the project's corroded PCB:

        **Icorr = {CORRODED_ICORR:.1f} µA/cm²**

        Calculated corrosion rate:

        **{verification:.4f} mm/year**

        Project-reported value:

        **approximately {CORRODED_CORROSION_RATE:.3f} mm/year**
        """
    )


# ============================================================
# TAB 3 — LIFETIME PREDICTION
# ============================================================

with tab3:

    st.header(
        "PCB Lifetime Prediction"
    )

    st.warning(
        """
        The current project does not contain enough multi-year
        time-series measurements to establish a statistically
        validated service-life model. Therefore, this section
        provides a transparent engineering screening estimate
        based on a selected corrosion rate.
        """
    )

    st.subheader(
        "Input PCB Conductor Information"
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        conductor_thickness = (
            st.number_input(
                "Initial copper thickness (mm)",
                min_value=0.000001,
                value=0.035,
                step=0.001,
                format="%.6f"
            )
        )

    with col2:

        allowable_loss_percent = (
            st.slider(
                "Allowable thickness loss (%)",
                min_value=1.0,
                max_value=100.0,
                value=100.0,
                step=1.0
            )
        )

    with col3:

        selected_rate = (
            st.number_input(
                "Corrosion rate (mm/year)",
                min_value=0.000001,
                value=CORRODED_CORROSION_RATE,
                step=0.001,
                format="%.6f"
            )
        )

    allowable_fraction = (
        allowable_loss_percent / 100
    )

    lifetime = lifetime_constant_rate(
        conductor_thickness,
        selected_rate,
        allowable_fraction
    )

    allowable_loss = (
        conductor_thickness
        * allowable_fraction
    )

    st.divider()

    c1, c2, c3 = st.columns(3)

    c1.metric(
        "Allowable Thickness Loss",
        f"{allowable_loss:.4f} mm"
    )

    c2.metric(
        "Corrosion Rate",
        f"{selected_rate:.3f} mm/year"
    )

    c3.metric(
        "Estimated Lifetime",
        f"{lifetime:.2f} years"
    )

    # --------------------------------------------------------
    # Lifetime Graph
    # --------------------------------------------------------

    st.subheader(
        "Predicted Thickness Loss"
    )

    maximum_year = max(
        lifetime * 1.2,
        1
    )

    years = np.linspace(
        0,
        maximum_year,
        250
    )

    thickness_loss = (
        selected_rate * years
    )

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.plot(
        years,
        thickness_loss,
        label="Predicted thickness loss"
    )

    ax.axhline(
        allowable_loss,
        linestyle="--",
        label="Allowable thickness loss"
    )

    ax.axvline(
        lifetime,
        linestyle="--",
        label=f"Estimated lifetime = {lifetime:.2f} years"
    )

    ax.set_xlabel(
        "Time (years)"
    )

    ax.set_ylabel(
        "Cumulative thickness loss (mm)"
    )

    ax.set_title(
        "PCB Conductor Corrosion Lifetime Prediction"
    )

    ax.grid(
        True,
        alpha=0.25
    )

    ax.legend()

    st.pyplot(fig)

    st.info(
        """
        **Model assumption:** the corrosion rate is assumed to
        remain constant throughout the prediction period.

        Actual PCB corrosion may change with humidity,
        chloride contamination, temperature, surface condition,
        protective films and localized corrosion.
        """
    )


# ============================================================
# TAB 4 — TIME-SERIES MODEL
# ============================================================

with tab4:

    st.header(
        "Empirical Time-Series Corrosion Model"
    )

    st.write(
        """
        This section allows you to use repeated experimental
        measurements to build a more meaningful corrosion
        progression model.
        """
    )

    st.latex(
        r"""
        CR = a t^b
        """
    )

    st.write(
        """
        Where:

        - **CR** = corrosion rate
        - **t** = exposure time
        - **a** = model coefficient
        - **b** = corrosion progression exponent
        """
    )

    # --------------------------------------------------------
    # CSV template
    # --------------------------------------------------------

    st.subheader(
        "CSV Data Format"
    )

    sample_data = pd.DataFrame(
        {
            "time_years": [
                0.25,
                0.50,
                1.00,
                2.00
            ],

            "corrosion_rate_mm_year": [
                0.04,
                0.06,
                0.09,
                0.13
            ]
        }
    )

    st.dataframe(
        sample_data,
        use_container_width=True,
        hide_index=True
    )

    csv_data = sample_data.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="Download CSV Template",
        data=csv_data,
        file_name="pcb_corrosion_time_series.csv",
        mime="text/csv"
    )

    uploaded_file = st.file_uploader(
        "Upload your corrosion time-series CSV",
        type=["csv"]
    )

    if uploaded_file is not None:

        data = pd.read_csv(
            uploaded_file
        )

    else:

        st.caption(
            "No file uploaded. Demonstration data are being used."
        )

        data = sample_data.copy()

    st.subheader(
        "Input Data"
    )

    st.dataframe(
        data,
        use_container_width=True,
        hide_index=True
    )

    required_columns = {
        "time_years",
        "corrosion_rate_mm_year"
    }

    if not required_columns.issubset(
        data.columns
    ):

        st.error(
            """
            Your CSV must contain exactly these
            required columns:

            time_years

            corrosion_rate_mm_year
            """
        )

    else:

        try:

            a, b, r_squared = fit_power_law(
                data["time_years"],
                data["corrosion_rate_mm_year"]
            )

            st.success(
                f"""
                Model successfully fitted:

                CR = {a:.6g} × t^{b:.4f}
                """
            )

            c1, c2 = st.columns(2)

            c1.metric(
                "Model exponent (b)",
                f"{b:.4f}"
            )

            c2.metric(
                "R²",
                f"{r_squared:.4f}"
            )

            # ------------------------------------------------
            # Prediction graph
            # ------------------------------------------------

            st.subheader(
                "Corrosion Progression Graph"
            )

            minimum_time = max(
                float(
                    data["time_years"].min()
                ),
                0.000001
            )

            maximum_time = (
                float(
                    data["time_years"].max()
                ) * 2
            )

            prediction_time = np.linspace(
                minimum_time,
                maximum_time,
                250
            )

            predicted_rate = (
                power_law_corrosion_rate(
                    a,
                    b,
                    prediction_time
                )
            )

            fig, ax = plt.subplots(
                figsize=(10, 5)
            )

            ax.scatter(
                data["time_years"],
                data["corrosion_rate_mm_year"],
                label="Experimental data"
            )

            ax.plot(
                prediction_time,
                predicted_rate,
                label="Power-law model"
            )

            ax.set_xlabel(
                "Exposure time (years)"
            )

            ax.set_ylabel(
                "Corrosion rate (mm/year)"
            )

            ax.set_title(
                "PCB Corrosion Rate vs Exposure Time"
            )

            ax.grid(
                True,
                alpha=0.25
            )

            ax.legend()

            st.pyplot(fig)

            # ------------------------------------------------
            # Lifetime calculation
            # ------------------------------------------------

            st.subheader(
                "Lifetime from Time-Series Model"
            )

            col1, col2 = st.columns(2)

            with col1:

                thickness_model = (
                    st.number_input(
                        "Copper thickness (mm)",
                        min_value=0.000001,
                        value=0.035,
                        step=0.001,
                        format="%.6f"
                    )
                )

            with col2:

                allowable_model = (
                    st.slider(
                        "Allowable thickness loss (%)",
                        min_value=1.0,
                        max_value=100.0,
                        value=100.0,
                        step=1.0
                    )
                )

            allowable_model_fraction = (
                allowable_model / 100
            )

            predicted_lifetime = (
                power_law_lifetime(
                    a,
                    b,
                    thickness_model,
                    allowable_model_fraction
                )
            )

            if np.isfinite(
                predicted_lifetime
            ):

                st.metric(
                    "Predicted Lifetime",
                    f"{predicted_lifetime:.2f} years"
                )

                st.info(
                    """
                    This prediction is based on the fitted
                    time-series corrosion behaviour. It is
                    only appropriate when the experimental
                    exposure conditions are representative
                    of the intended service environment.
                    """
                )

            else:

                st.error(
                    "Lifetime could not be calculated from this model."
                )

        except Exception as error:

            st.error(
                f"Model fitting error: {error}"
            )


# ============================================================
# 7. PROJECT LIMITATIONS
# ============================================================

st.divider()

st.header(
    "Scientific Limitations"
)

st.markdown(
    """
    ### 1. Limited number of specimens

    The project used a limited specimen set, including one
    specimen per condition for the main characterization.
    Therefore, formal statistical confidence intervals cannot
    be established from the present results.

    ### 2. Accelerated corrosion exposure

    The laboratory salt-spray/humidity exposure is an
    accelerated environment and does not reproduce every
    variable encountered during actual field service.

    ### 3. Lifetime validation

    A reliable long-term PCB lifetime model requires repeated
    corrosion measurements over time and preferably natural
    field-exposure data.

    ### 4. Conductor thickness

    A real lifetime calculation requires the actual copper
    conductor thickness and an engineering failure criterion.

    ### 5. Extrapolation

    Predictions outside the experimental range should be
    treated as estimates rather than guaranteed service life.
    """
)


# ============================================================
# 8. FOOTER
# ============================================================

st.divider()

st.caption(
    """
    PCB Corrosion Lifetime Prediction System |
    Python + Streamlit + NumPy + Pandas + Matplotlib
    """
)
```

### Install the required Python packages

Create a file called `requirements.txt`:

```text
streamlit
numpy
pandas
matplotlib
```

Then run:

```bash
pip install -r requirements.txt
```

And start the program with:

```bash
streamlit run app.py
```

The important next stage is to connect this program to the **actual raw polarization data in your project** so that it can automatically calculate `Ecorr`, `Icorr`, Tafel slopes and corrosion rate instead of requiring you to enter `Icorr` manually. Your document contains those raw polarization measurements, so that would make the software much stronger as part of your project.
