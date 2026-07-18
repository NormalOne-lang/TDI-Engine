[🇮🇷 نسخه فارسی در دسترس است](README.fa.md)

# TDI-Engine: Football Data ETL & Tactical Simulation

A Python-based data engineering pipeline designed to extract, process, and visualize tactical football data. 

**Disclaimer:** This project is developed primarily as a technical showcase of data extraction techniques, logical analysis, and system architecture. It is a tactical simulation tool, *not* an oracle for predicting exact match outcomes or a betting utility. 

## Project Overview

Modern football data platforms heavily restrict access to their APIs. The objective of this project is to demonstrate how to programmatically bypass these restrictions using advanced extraction techniques, bypassing the need for heavy browser automation (like Selenium). The engine extracts Server-Side Rendered (SSR) JSON payloads directly from Next.js web applications, parses raw tactical metrics, and applies a logical mathematical model to simulate a head-to-head tactical clash.

## Key Technical Features

*   **Browserless Network Extraction:** Utilizes `cloudscraper` to navigate Layer 7 protections and fetch the underlying `__NEXT_DATA__` JSON payload efficiently.
*   **Dynamic Data Parsing:** Identifies Home/Away context dynamically and uses regular expressions (regex) to extract clean floats from complex string arrays.
*   **Possession-Dampening Algorithm:** Instead of relying on raw historical averages, the simulation adjusts tactical outputs (like xG and box touches) based on possession share. This demonstrates critical thinking by ensuring counter-attacking systems are not statistically punished in the simulation.
*   **Automated Data Visualization:** Renders a broadcast-quality static dashboard (PNG) using `matplotlib` and `scipy.ndimage` to visualize match momentum and key metrics symmetrically.
*   **CLI Integration:** Fully operable via the command-line interface using `argparse`, making it ready for CI/CD pipelines or batch processing.

## Prerequisites

    pip install -r requirements.txt

## Usage

The script is executed via the command line. You must provide the former match URLs for the two teams you wish to simulate, along with their names.

**Example Command:**

    python TDI-Engine.py --match1 "https://www.fotmob.com/matches/team-1-url" --match2 "https://www.fotmob.com/matches/team-2-url" --team1 "France" --team2 "Spain"

Upon successful execution, the script will process the data and output a `prediction_dashboard.png` file in the root directory.

## Architecture Highlights

*   Object-Oriented Programming (OOP) with clear separation of concerns (Extractor, Parser, Model, Dashboard).
*   Standardized logging system instead of standard prints.
*   Graceful error handling for JSON structure changes or network timeouts.