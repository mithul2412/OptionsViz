# Functional Specification

## Background
Options are contracts that give traders the right (but not the obligation) to buy or sell a stock at a set price (strike) before an expiration date by paying a relatively small premium. This allows traders to leverage positions with limited upfront costs. 

Options chains are tabular datasets that list all call and put contracts across various strikes and expiration dates for a particular underlying asset. These chains contain multivariate data (e.g., strikes, expiration, Greeks, volume, open interest, theoretical Black-Scholes price), which can be difficult to interpret in a tabular format. Identifying trends in live (non-historical) data is challenging. 

Traders struggle to analyze this data effectively and recognize repeatable patterns and profitable opportunities—less than 10% of options traders actually make money. Our goal is to create a series of insightful visualizations and strategy comparisons that enable traders to make more informed decisions regarding options contracts, maximizing returns while mitigating risk.

## User Profile

### A. Financial Institutions and Banks
- **Who They Are:** Quantitative researchers and analysts working in financial institutions and banks.
- **Domain Expertise:** Specialized knowledge in quantitative finance and options strategies.
- **Computing Skills:** Proficient in technical analysis and data manipulation, but require a user-friendly interface to rapidly apply insights without extensive programming overhead.

### B. Trading Platforms
- **Who They Are:** Companies such as Robinhood, Interactive Brokers, or Thinkorswim (Schwab) seeking to enhance their trading interfaces.
- **Domain Expertise:** In-house teams with deep understanding of both financial products and technology integration.
- **Computing Skills:** Skilled in API integration, capable of supporting and maintaining complex systems for smooth data exchanges and continual updates.

### C. Individual Traders (Retail Investors)
- **Who They Are:** Retail traders using platforms like Robinhood or Interactive Brokers.
- **Domain Expertise:** Basic to intermediate knowledge of options trading and risk assessment.
- **Computing Skills:** Comfortable with web applications; can easily navigate tools to upload CSV data and interpret interactive visualizations.

## Data Sources

### **Primary Data**
#### **Options Chains** (Key Metrics)
- Greeks (Delta, Theta, Vega)
- Volume
- Open Interest
- Strikes across expiration dates

#### **Sources**
- **Thinkorswim (TOS):** Provides detailed options chain data.
- **Interactive Brokers (IBKR) Trader Workstation (TWS) API:** Supplies real-time and historical options data.

### **Supplementary Data**
- **Financial News:** Integrated via an LLM/NLP module that scrapes and summarizes real-time news related to options trading.

### **Data Structure**
- **CSV Format:** For user-uploaded options chain data.
- **API Feeds:** For real-time data integration with trading platforms and news sources.

## Use Cases

### 1. Retail Trader
- **Objective:** Identify repeatable patterns and unusual options activity.
- **Expected Interactions:** 
    - Upload data.
    - Generate visualizations of the options chain.
    - Manually analyze potential profitable trade events.

### 2. **Risk Assessment for Possible Strategies**
- **Objective:** Assess risk versus reward of multiple options strategies.
- **Expected Interactions:** 
    - Upload data.
    - Input risk tolerance, budget, and monetary outlook for the underlying stock.
    - Click to generate a sorted list of options strategies.

## User Interaction

### **1. Data Upload**
- The trader uploads a CSV file containing options chain data (including strikes, expiration dates, Greeks, volume, and open interest) from platforms like Robinhood or IBKR.

### **2. Parameter Input**
- The system prompts the trader to input relevant parameters (such as risk tolerance, target return, etc.).

### **3. Data Processing & Visualization**
- The system processes the uploaded data using pre-set algorithms and calculations to derive key metrics.
- Interactive visualizations highlight critical insights such as risk-to-reward ratios and potential profitable patterns.

### **4. Decision Making**
- The trader analyzes the visualizations to identify favorable trade opportunities.
- Decisions are refined based on assessed risk.

### **5. Enhanced Decision Support**
- The platform incorporates an NLP-based module that provides real-time news updates and qualitative insights.
- This ensures a comprehensive view before executing trades.
