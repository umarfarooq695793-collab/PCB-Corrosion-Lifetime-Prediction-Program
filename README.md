# PCB Corrosion Lifetime Prediction System

A Python-based Streamlit application for analysing corrosion
behaviour and estimating screening-level PCB conductor lifetime.

## Project

Investigation of Oxidation and Corrosion Behaviour in Panel
Circuit Boards.

## Features

- PCB corrosion-rate calculation
- Electrochemical corrosion comparison
- Icorr-based corrosion-rate calculation
- Constant-rate lifetime screening
- Remaining conductor thickness projection
- Empirical power-law time-series modelling
- Mechanical degradation comparison
- Experimental CSV data upload
- Corrosion-rate graphs
- Scientific limitations and assumptions

## Project Results Included

The application includes the reported electrochemical results:

| Parameter | Control | Corroded |
|---|---:|---:|
| Icorr | ≤ 1.47 µA/cm² | 19.0 µA/cm² |
| Corrosion rate | ≤ 0.017 mm/year | 0.220 mm/year |

Mechanical results included:

| Property | Control | Corroded |
|---|---:|---:|
| UTS | 61.11 MPa | 35.66 MPa |
| Young's modulus | 806.43 MPa | 307.35 MPa |
| Elongation | ~19.6% | 25.3% |

## Installation

Install Python 3.9 or newer.

Create a virtual environment:

```bash
python -m venv venv
