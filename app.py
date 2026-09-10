import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PCB CORROSION LIFETIME PREDICTION SYSTEM
# ============================================================

st.set_page_config(
    page_title="PCB Corrosion Lifetime Prediction",
    page_icon="🔬",
    layout="wide"
)

# ------------------------------------------------------------
# PROJECT DATA
# ------------------------------------------------------------

CONTROL_ICORR = 1.47          # µA/cm², upper-bound value
CORRODED_ICORR = 19.0         # µA/cm²
CONTROL_RATE = 0.017          # mm/year, upper-bound
CORRODED_RATE = 0.220         # mm/year

COPPER_EQUIVALENT_WEIGHT = 31.77
COPPER_DENSITY = 8.96

# Tafel slopes from project analysis
BETA_A = 104.0                # mV/decade
BETA_C = 351.0                # mV/decade


# ============================================================
# FUNCTIONS
# ============================================================

def calculate_corrosion_rate(icorr, equivalent_weight=31.77,
                             density=8.96):
    """
    Calculate corrosion rate using the ASTM G102/Faraday-law
    form used in the project.

    CR = 3.27 × 10^-3 × Icorr × EW / density

    Icorr: µA/cm²
    EW: g/equiv
    density: g/cm³

    Returns:
        corrosion rate in mm/year
    """
    return (
        3.27e-3
        * icorr
        * equivalent_weight
        / density
    )


def estimate_lifetime(thickness_mm, corrosion_rate):
    """
    Simple constant-rate screening estimate.

    Lifetime = thickness / corrosion rate

    This is NOT a validated field-life prediction.
    """
    if corrosion_rate <= 0:
        return np.inf

    return thickness_mm / corrosion_rate


def remaining_thickness(initial_thickness, corrosion_rate, years):
    """
    Estimate remaining thickness assuming a constant
    corrosion rate.
    """
    remaining = initial_thickness - corrosion_rate * years
    return max(remaining, 0)


def fit_power_law(time, corrosion_rate):
    """
    Fit empirical model:

        CR = a * t^b

    by linear regression in log-log space.
    """

    time = np.asarray(time, dtype=float)
    corrosion_rate = np.asarray(corrosion_rate, dtype=float)

    valid = (
        np.isfinite(time)
        & np.isfinite(corrosion_rate)
        & (time > 0)
        & (corrosion_rate > 0)
    )

    time = time[valid]
    corrosion_rate = corrosion_rate[valid]

    if len(time) < 3:
        return None

    log_time = np.log(time)
    log_rate = np.log(corrosion_rate)

    slope, intercept = np.polyfit(log_time, log_rate, 1)

    b = slope
    a = np.exp(intercept)

    predictions = a * time ** b

    ss_res = np.sum((corrosion_rate - predictions) ** 2)
    ss_tot = np.sum((corrosion_rate - np.mean(corrosion_rate)) ** 2)

    if ss_tot == 0:
        r_squared = 1.0
    else:
        r_squared = 1 - (ss_res / ss_tot)

    return a, b, r_squared


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🔬 PCB Corrosion Model")

st.sidebar.markdown(
    """
This application provides a screening-level analysis of
PCB corrosion behaviour using electrochemical corrosion data
and simple lifetime extrapolation models.
"""
)

page = st.sidebar.radio(
    "Select analysis",
    [
        "Dashboard",
        "Corrosion Calculator",
        "Lifetime Prediction",
        "Time-Series Model",
        "Mechanical Degradation",
        "Limitations"
    ]
)


# ============================================================
# DASHBOARD
# ============================================================

if page == "Dashboard":

    st.title("🔬 PCB Corrosion Lifetime Prediction System")

    st.markdown(
        """
    ### Investigation of Oxidation and Corrosion Behaviour
    in Panel Circuit Boards

    This software compares the corrosion behaviour of an
    **as-received control PCB** with a **deliberately corroded PCB**
    using electrochemical and mechanical results from the project.
    """
    )

    st.divider()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Control Icorr",
            "≤ 1.47 µA/cm²"
        )

    with col2:
        st.metric(
            "Corroded Icorr",
            "19.0 µA/cm²"
        )

    with col3:
        st.metric(
            "Control corrosion rate",
            "≤ 0.017 mm/year"
        )

    with col4:
        st.metric(
            "Corroded corrosion rate",
            "0.220 mm/year"
        )

    st.divider()

    st.subheader("Electrochemical Comparison")

    comparison = pd.DataFrame({
        "Condition": [
            "Control PCB",
            "Corroded PCB"
        ],
        "Icorr (µA/cm²)": [
            CONTROL_ICORR,
            CORRODED_ICORR
        ],
        "Corrosion Rate (mm/year)": [
            CONTROL_RATE,
            CORRODED_RATE
        ]
    })

    st.dataframe(
        comparison,
        use_container_width=True,
        hide_index=True
    )

    st.subheader("Corrosion Rate Comparison")

    fig, ax = plt.subplots(figsize=(8, 4))

    ax.bar(
        comparison["Condition"],
        comparison["Corrosion Rate (mm/year)"]
    )

    ax.set_ylabel("Corrosion rate (mm/year)")
    ax.set_title("PCB Corrosion Rate Comparison")

    st.pyplot(fig)

    st.info(
        "The lifetime calculations in this application are "
        "screening estimates. Actual PCB lifetime requires "
        "measured conductor thickness, environmental exposure "
        "history and a defined failure criterion."
    )


# ============================================================
# CORROSION CALCULATOR
# ============================================================

elif page == "Corrosion Calculator":

    st.title("🧮 Corrosion Rate Calculator")

    st.write(
        "Calculate corrosion rate from corrosion current density "
        "using the equation applied in the project."
    )

    col1, col2 = st.columns(2)

    with col1:
        icorr = st.number_input(
            "Corrosion current density, Icorr (µA/cm²)",
            min_value=0.000001,
            value=19.0,
            step=0.1
        )

        equivalent_weight = st.number_input(
            "Equivalent weight (g/equiv)",
            min_value=0.001,
            value=COPPER_EQUIVALENT_WEIGHT,
            step=0.01
        )

    with col2:

        density = st.number_input(
            "Material density (g/cm³)",
            min_value=0.001,
            value=COPPER_DENSITY,
            step=0.01
        )

        st.latex(
            r"CR = 3.27 \times 10^{-3}"
            r"\frac{I_{corr} \times EW}{\rho}"
        )

    if st.button("Calculate corrosion rate", type="primary"):

        rate = calculate_corrosion_rate(
            icorr,
            equivalent_weight,
            density
        )

        st.success(
            f"Estimated corrosion rate: **{rate:.4f} mm/year**"
        )


# ============================================================
# LIFETIME PREDICTION
# ============================================================

elif page == "Lifetime Prediction":

    st.title("⏳ PCB Lifetime Screening Model")

    st.warning(
        "This model assumes a constant corrosion rate. "
        "It is a screening extrapolation, not a validated "
        "service-life prediction."
    )

    col1, col2 = st.columns(2)

    with col1:

        thickness = st.number_input(
            "Initial conductor thickness (mm)",
            min_value=0.001,
            value=0.035,
            step=0.001,
            help=(
                "Example value only. Replace this with the "
                "actual conductor thickness of the PCB."
            )
        )

    with col2:

        rate = st.number_input(
            "Corrosion rate (mm/year)",
            min_value=0.000001,
            value=CORRODED_RATE,
            step=0.001
        )

    lifetime = estimate_lifetime(
        thickness,
        rate
    )

    st.metric(
        "Screening lifetime",
        f"{lifetime:.2f} years"
    )

    st.divider()

    st.subheader("Projected Remaining Thickness")

    years = np.linspace(
        0,
        max(lifetime * 1.2, 1),
        200
    )

    remaining = [
        remaining_thickness(
            thickness,
            rate,
            year
        )
        for year in years
    ]

    fig, ax = plt.subplots(figsize=(9, 5))

    ax.plot(
        years,
        remaining
    )

    ax.axhline(
        0,
        linestyle="--"
    )

    ax.set_xlabel("Exposure time (years)")
    ax.set_ylabel("Remaining thickness (mm)")
    ax.set_title(
        "Constant-Rate PCB Conductor Thickness Projection"
    )

    st.pyplot(fig)

    st.subheader("Important assumptions")

    st.markdown(
        """
        - Corrosion rate remains constant.
        - Corrosion is treated as uniform thickness loss.
        - No acceleration or deceleration of corrosion is included.
        - Mechanical damage is not included.
        - Electrical failure criteria are not included.
        - Actual conductor thickness must be measured.
        """
    )


# ============================================================
# TIME-SERIES MODEL
# ============================================================

elif page == "Time-Series Model":

    st.title("📈 Empirical Time-Series Corrosion Model")

    st.markdown(
        """
        Upload experimental corrosion-rate measurements to fit
        the empirical model:

        **CR = a × tᵇ**

        where:

        - **CR** = corrosion rate
        - **t** = exposure time
        - **a** = fitted coefficient
        - **b** = fitted exponent
        """
    )

    uploaded_file = st.file_uploader(
        "Upload CSV file",
        type=["csv"]
    )

    st.info(
        "CSV must contain two columns: "
        "`time` and `corrosion_rate`."
    )

    if uploaded_file is not None:

        try:

            data = pd.read_csv(uploaded_file)

            required_columns = {
                "time",
                "corrosion_rate"
            }

            if not required_columns.issubset(data.columns):

                st.error(
                    "CSV must contain columns named "
                    "'time' and 'corrosion_rate'."
                )

            else:

                st.subheader("Experimental Data")

                st.dataframe(
                    data,
                    use_container_width=True
                )

                result = fit_power_law(
                    data["time"],
                    data["corrosion_rate"]
                )

                if result is None:

                    st.error(
                        "At least 3 positive data points are "
                        "required for the empirical model."
                    )

                else:

                    a, b, r_squared = result

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "a",
                            f"{a:.6f}"
                        )

                    with col2:
                        st.metric(
                            "b",
                            f"{b:.4f}"
                        )

                    with col3:
                        st.metric(
                            "R²",
                            f"{r_squared:.4f}"
                        )

                    st.latex(
                        rf"CR = {a:.6f}t^{{{b:.4f}}}"
                    )

                    time_values = np.linspace(
                        data["time"].min(),
                        data["time"].max(),
                        200
                    )

                    predicted = (
                        a * time_values ** b
                    )

                    fig, ax = plt.subplots(
                        figsize=(9, 5)
                    )

                    ax.scatter(
                        data["time"],
                        data["corrosion_rate"],
                        label="Experimental data"
                    )

                    ax.plot(
                        time_values,
                        predicted,
                        label="Power-law fit"
                    )

                    ax.set_xlabel(
                        "Exposure time"
                    )

                    ax.set_ylabel(
                        "Corrosion rate"
                    )

                    ax.set_title(
                        "Empirical Corrosion Model"
                    )

                    ax.legend()

                    st.pyplot(fig)

        except Exception as error:

            st.error(
                f"Error reading file: {error}"
            )

    else:

        st.write("No CSV file uploaded yet.")


# ============================================================
# MECHANICAL DEGRADATION
# ============================================================

elif page == "Mechanical Degradation":

    st.title("⚙️ Mechanical Degradation Analysis")

    st.markdown(
        """
        The project reported changes in tensile properties between
        the control and corroded specimens.
        """
    )

    control_uts = 61.11
    corroded_uts = 35.66

    control_modulus = 806.43
    corroded_modulus = 307.35

    control_elongation = 19.6
    corroded_elongation = 25.3

    uts_reduction = (
        (control_uts - corroded_uts)
        / control_uts
        * 100
    )

    modulus_reduction = (
        (control_modulus - corroded_modulus)
        / control_modulus
        * 100
    )

    col1, col2 = st.columns(2)

    with col1:

        st.metric(
            "UTS reduction",
            f"{uts_reduction:.1f}%"
        )

        st.metric(
            "Young's modulus reduction",
            f"{modulus_reduction:.1f}%"
        )

    with col2:

        st.metric(
            "Control UTS",
            f"{control_uts:.2f} MPa"
        )

        st.metric(
            "Corroded UTS",
            f"{corroded_uts:.2f} MPa"
        )

    mechanical_data = pd.DataFrame({
        "Property": [
            "UTS (MPa)",
            "Young's modulus (MPa)",
            "Elongation (%)"
        ],
        "Control": [
            control_uts,
            control_modulus,
            control_elongation
        ],
        "Corroded": [
            corroded_uts,
            corroded_modulus,
            corroded_elongation
        ]
    })

    st.subheader("Mechanical Properties")

    st.dataframe(
        mechanical_data,
        use_container_width=True,
        hide_index=True
    )

    fig, ax = plt.subplots(figsize=(9, 5))

    x = np.arange(len(mechanical_data))
    width = 0.35

    ax.bar(
        x - width / 2,
        mechanical_data["Control"],
        width,
        label="Control"
    )

    ax.bar(
        x + width / 2,
        mechanical_data["Corroded"],
        width,
        label="Corroded"
    )

    ax.set_xticks(x)
    ax.set_xticklabels(
        mechanical_data["Property"]
    )

    ax.set_ylabel("Measured value")
    ax.set_title("Control vs Corroded PCB")
    ax.legend()

    st.pyplot(fig)

    st.warning(
        "The project notes that only one coupon was tested per "
        "condition, so these results should be interpreted "
        "comparatively rather than as statistically validated "
        "population estimates."
    )


# ============================================================
# LIMITATIONS
# ============================================================

elif page == "Limitations":

    st.title("⚠️ Scientific Limitations")

    st.markdown(
        """
### 1. Limited number of specimens

The project used one specimen per condition for the reported
mechanical testing. Therefore, statistical uncertainty cannot
be reliably estimated.

### 2. Lifetime extrapolation

The constant-rate model assumes that the corrosion rate remains
constant over the entire service period.

Real PCB corrosion can vary with:

- humidity
- temperature
- chloride exposure
- contamination
- material composition
- surface condition
- electrical potential
- corrosion-product formation

### 3. Conductor thickness

The lifetime calculation requires the actual thickness of the
conductive material. The default thickness in this application
is only an example.

### 4. Failure criterion

A meaningful PCB lifetime prediction should define what counts
as failure, such as:

- critical conductor thickness
- unacceptable resistance increase
- electrical open circuit
- mechanical failure
- loss of required functionality

### 5. Electrochemical measurements

The project used potentiodynamic polarization to characterize
corrosion behaviour. A single polarization measurement should
not automatically be interpreted as a complete long-term
environmental degradation model.

### 6. Future improvement

The system can be improved by incorporating:

- multiple polarization measurements
- raw polarization curves
- EIS data
- resistance measurements
- humidity and temperature histories
- salt concentration
- conductor thickness measurements
- multiple PCB specimens
- field exposure data
- statistical uncertainty
- machine-learning models after sufficient experimental data
        """
    )

    st.success(
        "The model should be presented as a corrosion-analysis "
        "and lifetime-screening tool rather than a validated "
        "prediction of actual PCB service life."
    )
