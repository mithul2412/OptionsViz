# OptionsViz 

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Workflow Status](https://github.com/rrichards7/OptionsViz/actions/workflows/build_test.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/rrichards7/OptionsViz/badge.svg?branch=main)](https://coveralls.io/github/rrichards7/OptionsViz?branch=main)

<!-- ![Build/Test Workflow](https://github.com/UWDATA515/ci_example/actions/workflows/build_test.yml/badge.svg)
[![Coverage Status](https://coveralls.io/repos/github/UWDATA515/ci_example/badge.svg?branch=main)](https://coveralls.io/github/UWDATA515/ci_example?branch=main) -->

*Options Data Visualization*

Options are contracts that give the right (but not the obligation) to buy or sell a stock at a set price before a deadline; allows traders to leverage positions with limited upfront cost. Options chains contain multivariate data (strikes, expiration, Greeks, volume, open interest, etc) that is difficult to interpret in tabular format.  Traders struggle to identify patterns and profitable opportunities: only ~10% of traders make money. Our goal will be to create a series of pivotal, insightful, and impactful visualizations that allow traders to make more informed decisions regarding the options contract(s) in which they are investing to maximize the return and mitigate the risk. 

## Project Type

Tool and data presentation

## Questions of interest

*How can visualizing options chain data improve a trader’s ability to identify profitable strategies?*

*How can visualizing options strategies mitigate risk?*

*How can time-varying (dynamic) options chains provide greater insight?*

## Goal 

Web-app (streamlit) or interface that allows users to upload a dataframe (e.g. csv) of a single or multiple options chains and automatically generate visuals for their data. This web-app will be deployed and hosted using streamlit's free service through our public git repo.

## Data sources

Think-or-Swim (TOS); and Interactive Brokers (IBKR) Trader Workstation (TWS) API


## Installation and Setup

### 1. Clone the repository:

Clone the repo through HTTPS:
```bash
git clone https://github.com/rrichards7/OptionsViz.git
```

*[Optional] or clone via ssh:*
```bash
git clone git@github.com:rrichards7/OptionsViz.git
```

*[Optional] or clone via GitHub CLI:*
```bash
gh repo clone rrichards7/OptionsViz
```

cd into the repo after cloning:
```
cd OptionsViz
```

### 2. Create conda environment

Create new conda environment (which will be called "optviz" by default) from the yaml file:

```bash
conda env create -f environment.yml
```

Activate the environment after creation:

```bash
conda activate optviz
```

### 3. Install

Install OptionsViz from the main repo through the pyproject.toml file by running:

```bash
pip install .
```

## Execution

To run the streamlit page, run this cmd from the main repo level:

```bash
streamlit run optionsviz/app.py
```

Then navigate in a browser to:
```
http://localhost:8501/
```

The app selects port 8501 by default, but can be controlled by including the flag: --server.port, e.g. to run on a specified port, use:

```bash
streamlit run optionsviz/app.py --server.port 5555
```

*Make sure you are forwarding the correct ports if running remotely*
