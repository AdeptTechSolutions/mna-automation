import json
import os
from pathlib import Path
from typing import Annotated, Dict

import financedatabase as fd
import numpy as np
import pandas as pd
import yfinance as yf
from dotenv import load_dotenv
from financetoolkit import Toolkit
from smolagents import tool
from typing_extensions import Annotated

from finmodels_tools import FinModelsTools

load_dotenv()


@tool
def human_intervention(
    scenario: Annotated[str, "Type of human intervention scenario"],
    message_for_human: Annotated[str, "Message to display to the human"],
    choices: Annotated[list, "List of choices for multiple_choice scenario"] = None,
) -> str:
    """
    A universal human-in-the-loop tool for various scenarios.

    Args:
        scenario: The type of human intervention scenario. Must be 'clarification', 'approval', or 'multiple_choice'.
        message_for_human: The message to be displayed to the human.
        choices: A list of choices for the 'multiple_choice' scenario.
    """
    if scenario not in ["clarification", "approval", "multiple_choice"]:
        raise ValueError("Must be 'clarification', 'approval', or 'multiple_choice'.")

    print("\n[HUMAN INTERVENTION]")
    print(f"Scenario: {scenario}")
    print(f"Agent says: {message_for_human}")

    if scenario == "clarification":
        user_input = input("\nType your response: ")
        return user_input

    elif scenario == "approval":
        print("Type 'YES' or 'NO' to proceed:")
        user_input = input("Your decision: ").strip().upper()
        return user_input

    elif scenario == "multiple_choice":
        if not choices:
            return "No choices were provided."
        print("\nAvailable options:")
        for i, choice in enumerate(choices, start=1):
            print(f"{i}. {choice}")
        user_input = input("\nEnter the number of your chosen option: ")
        return user_input


@tool
def collect_financial_metrics(
    symbol: Annotated[str, "Company symbol to analyze"],
) -> dict:
    """Collect key financial metrics using FinanceToolkit.

    Args:
        symbol: The company symbol to analyze.

    Returns:
        dict: Dictionary containing financial metrics and status message.
    """
    try:
        company = Toolkit(symbol, api_key=os.getenv("FMP_API_KEY"))

        income_statement = company.get_income_statement()
        balance_sheet = company.get_balance_sheet_statement()
        cash_flow = company.get_cash_flow_statement()
        metrics = company.ratios.collect_profitability_ratios()

        report = f"""# Financial Metrics Report for {symbol}
        
## Income Statement
{income_statement.to_markdown()}

## Balance Sheet
{balance_sheet.to_markdown()}

## Cash Flow Statement
{cash_flow.to_markdown()}

## Profitability Metrics
{metrics.to_markdown()}
"""

        output_path = Path("outputs/fmp_data") / f"{symbol}_metrics.md"
        with open(output_path, "w") as f:
            f.write(report)

        return {"status": "success", "message": f"Metrics saved to {output_path}"}
    except Exception as e:
        return {"status": "error", "message": f"Error collecting metrics: {str(e)}"}


@tool
def perform_valuation_analysis(
    symbol: Annotated[str, "Company symbol to analyze"],
) -> dict:
    """Perform comprehensive valuation analysis using FinanceToolkit.

    Args:
        symbol: The company symbol to analyze.

    Returns:
        dict: Dictionary containing valuation results and status message.
    """
    try:
        company = Toolkit(symbol, api_key=os.getenv("FMP_API_KEY"))
        fin_models = FinModelsTools(os.getenv("FMP_API_KEY"))

        try:
            ev = company.models.get_enterprise_value_breakdown()
            ev_dict = ev.to_dict("records")
        except Exception:
            ev_dict = [{"Error": "EV calculation failed"}]

        try:
            profile = company.get_profile()
            first_col = profile.columns[0]
            dcf_value = profile.loc["DCF", first_col]
            dcf_difference = profile.loc["DCF Difference", first_col]

            dcf_dict = [
                {
                    "symbol": symbol,
                    "dcf_value": dcf_value,
                    "dcf_difference": dcf_difference,
                    "current_price": profile.loc["Price", first_col],
                    "valuation_status": (
                        "undervalued"
                        if dcf_value > profile.loc["Price", first_col]
                        else "overvalued"
                    ),
                }
            ]
        except Exception:
            dcf_dict = [{"Error": "DCF calculation failed"}]

        try:
            wacc = company.models.get_weighted_average_cost_of_capital()
            wacc_dict = wacc.to_dict("records")
        except Exception:
            wacc_dict = [{"Error": "WACC calculation failed"}]

        try:
            dupont = company.models.get_dupont_analysis()
            dupont_dict = dupont.to_dict("records")
        except Exception:
            dupont_dict = [{"Error": "Dupont analysis calculation failed"}]

        try:
            ext_dupont = company.models.get_extended_dupont_analysis()
            ext_dupont_dict = ext_dupont.to_dict("records")
        except Exception:
            ext_dupont_dict = [{"Error": "Extended Dupont analysis calculation failed"}]

        try:
            cap = company.performance.get_capital_asset_pricing_model()
            cap_dict = cap.to_dict("records")
        except Exception:
            cap_dict = [{"Error": "CAPM calculation failed"}]

        try:
            all_perf = company.performance.collect_all_metrics()
            all_perf_dict = all_perf.to_dict("records")
        except Exception:
            all_perf_dict = [{"Error": "Performance metrics calculation failed"}]

        try:
            alt_zscore = company.models.get_altman_z_score()
            alt_zscore_dict = alt_zscore.to_dict("records")
        except Exception:
            alt_zscore_dict = [{"Error": "Altman Z-Score calculation failed"}]

        try:
            intrinsic_value = company.models.get_intrinsic_valuation(15, 0.04, 0.094)
            intrinsic_value_dict = intrinsic_value.to_dict("records")
        except Exception:
            intrinsic_value_dict = [{"Error": "Intrinsic Value calculation failed"}]
        try:
            piotroski = company.models.get_piotroski_score()
            piotroski_dict = piotroski.to_dict("records")
        except Exception:
            piotroski_dict = [{"Error": "Piotroski score calculation failed"}]

        try:
            gorden = company.models.get_gorden_growth_model(0.15, 0.04)
            gorden_dict = gorden.to_dict("records")
        except Exception:
            gorden_dict = [{"Error": "Gorden growth model calculation failed"}]

        try:
            lbo_metrics = fin_models.calculate_lbo_metrics(symbol)
            lbo_dict = (
                lbo_metrics
                if lbo_metrics
                else {"Error": "LBO metrics calculation failed"}
            )
        except Exception:
            lbo_dict = {"Error": "LBO metrics calculation failed"}

        try:
            ipo_valuation = fin_models.calculate_ipo_valuation(symbol)
            ipo_dict = (
                ipo_valuation
                if ipo_valuation
                else {"Error": "IPO valuation calculation failed"}
            )
        except Exception:
            ipo_dict = {"Error": "IPO valuation calculation failed"}

        try:
            lbo_sensitivity = fin_models.perform_lbo_sensitivity_analysis(symbol)
            lbo_sensitivity_dict = (
                lbo_sensitivity
                if lbo_sensitivity
                else {"Error": "LBO sensitivity analysis failed"}
            )
        except Exception:
            lbo_sensitivity_dict = {"Error": "LBO sensitivity analysis failed"}

        try:
            ipo_sensitivity = fin_models.perform_ipo_sensitivity_analysis(symbol)
            ipo_sensitivity_dict = (
                ipo_sensitivity
                if ipo_sensitivity
                else {"Error": "IPO sensitivity analysis failed"}
            )
        except Exception:
            ipo_sensitivity_dict = {"Error": "IPO sensitivity analysis failed"}

        output_path = Path("outputs/fmp_data") / f"{symbol}_valuation.md"
        report = f"""# Valuation Analysis for {symbol}

## Enterprise Value Breakdown
{ev.to_markdown() if not isinstance(ev_dict[0].get("Error"), str) else "Error calculating Enterprise Value"}

## DCF Valuation
- Current Price: {profile.loc['Price', first_col]}
- DCF Value: {dcf_value}
- DCF Difference: {dcf_difference}
- Valuation Status: {'Undervalued' if dcf_value > profile.loc['Price', first_col] else 'Overvalued'}
 
## Weighted Average Cost of Capital
{wacc.to_markdown() if not isinstance(wacc_dict[0].get("Error"), str) else "Error calculating WACC"}

## Dupont Analysis
{dupont.to_markdown() if not isinstance(dupont_dict[0].get("Error"), str) else "Error calculating Dupont Analysis"}

## Extended Dupont Analysis
{ext_dupont.to_markdown() if not isinstance(ext_dupont_dict[0].get("Error"), str) else "Error calculating Extended Dupont Analysis"}

## Altman Z-Score
{alt_zscore.to_markdown() if not isinstance(alt_zscore_dict[0].get("Error"), str) else "Error calculating Altman Z-Score"}

## Intrinsic Value
{intrinsic_value.to_markdown() if not isinstance(intrinsic_value_dict[0].get("Error"), str) else "Error calculating Intrinsic Value"}

## Capital Asset Pricing Model
{cap.to_markdown() if not isinstance(cap_dict[0].get("Error"), str) else "Error calculating CAPM"}

## Performance Metrics
{all_perf.to_markdown() if not isinstance(all_perf_dict[0].get("Error"), str) else "Error calculating Performance Metrics"}

## Piotroski Score
{piotroski.to_markdown() if not isinstance(piotroski_dict[0].get("Error"), str) else "Error calculating Piotroski Score"}

## Gorden Growth Model
{gorden.to_markdown() if not isinstance(gorden_dict[0].get("Error"), str) else "Error calculating Gorden Growth Model"}

## LBO Analysis
{format_lbo_metrics(lbo_dict) if not isinstance(lbo_dict.get("Error"), str) else "Error calculating LBO Metrics"}

## IPO Valuation
{format_ipo_valuation(ipo_dict) if not isinstance(ipo_dict.get("Error"), str) else "Error calculating IPO Valuation"}

## LBO Sensitivity Analysis
{format_lbo_sensitivity(lbo_sensitivity_dict) if not isinstance(lbo_sensitivity_dict.get("Error"), str) else "Error calculating LBO Sensitivity Analysis"}

## IPO Sensitivity Analysis
{format_ipo_sensitivity(ipo_sensitivity_dict) if not isinstance(ipo_sensitivity_dict.get("Error"), str) else "Error calculating IPO Sensitivity Analysis"}
"""

        with open(output_path, "w") as f:
            f.write(report)

        return {
            "status": "success",
            "message": f"Valuation saved to {output_path}",
        }
    except Exception as e:
        return {"status": "error", "message": f"Error performing valuation: {str(e)}"}


def format_lbo_metrics(metrics: Dict) -> str:
    """Format LBO metrics for markdown report."""

    irr = metrics["returns"]["irr"]
    if isinstance(irr, str):
        irr_display = irr
    else:
        irr_display = f"{irr:.2%}"

    return f"""### Purchase Price
- Enterprise Value: ${metrics['purchase_price']['enterprise_value']:,.2f}
- Equity Contribution: ${metrics['purchase_price']['equity_contribution']:,.2f}
- Debt Financing: ${metrics['purchase_price']['debt_financing']:,.2f}
- Debt Ratio: {metrics['purchase_price']['debt_ratio']:.2%}

### Projections
- Exit Value: ${metrics['projections']['exit_value']:,.2f}
- Remaining Debt: ${metrics['projections']['remaining_debt']:,.2f}
- Projected Free Cash Flow:
  {', '.join(f'Year {i+1}: ${fcf:,.2f}' for i, fcf in enumerate(metrics['projections']['projected_fcf']))}

### Returns
- IRR: {irr_display}
- MOIC: {metrics['returns']['moic']:.2f}x
- Exit Equity: ${metrics['returns']['exit_equity']:,.2f}"""


def format_ipo_valuation(valuation: Dict) -> str:
    """Format IPO valuation for markdown report."""

    def format_multiple(value):
        return f"{value:.2f}x" if value != "N/A" else "N/A"

    return f"""### Valuation Metrics
- Enterprise Value: ${valuation['valuation']['enterprise_value']:,.2f}
- Equity Value: ${valuation['valuation']['equity_value']:,.2f}
- EV/Revenue Multiple: {format_multiple(valuation['valuation']['ev_revenue_multiple'])}
- EV/EBITDA Multiple: {format_multiple(valuation['valuation']['ev_ebitda_multiple'])}

### Offering Details
- Shares Outstanding: {valuation['offering']['shares_outstanding']:,.0f}
- Float Shares: {valuation['offering']['float_shares']:,.0f}
- Price Range:
  - Low: ${valuation['offering']['price_range']['low']:,.2f}
  - Base: ${valuation['offering']['price_range']['base']:,.2f}
  - High: ${valuation['offering']['price_range']['high']:,.2f}

### Comparable Analysis
- Sector P/E: {valuation['comparables']['sector_pe']:.2f}x
- Revenue Growth: {valuation['comparables']['revenue_growth']:.2%}
- EBIT Growth: {valuation['comparables']['ebit_growth']:.2%}
- Peer Companies: {', '.join(valuation['comparables']['peer_companies'])}"""


def format_lbo_sensitivity(sensitivity: Dict) -> str:
    """Format LBO sensitivity analysis for markdown report."""
    unique_growth_rates = sorted(set(sensitivity["fcf_growth_rate"]))
    unique_exit_multiples = sorted(set(sensitivity["exit_multiple"]))

    table = "\n\n"
    table += "Selected scenarios showing IRR variation by FCF Growth Rate and Exit Multiple (at 8% interest rate):\n\n"
    table += "| FCF Growth | Exit Multiple | IRR | MOIC |\n"
    table += "|------------|---------------|-----|------|\n"

    base_rate_indices = [
        i for i, rate in enumerate(sensitivity["interest_rate"]) if rate == 0.08
    ]

    for growth_rate in unique_growth_rates:
        for multiple in unique_exit_multiples:
            for idx in base_rate_indices:
                if (
                    sensitivity["fcf_growth_rate"][idx] == growth_rate
                    and sensitivity["exit_multiple"][idx] == multiple
                ):
                    irr = sensitivity["irr"][idx]
                    moic = sensitivity["moic"][idx]

                    if isinstance(irr, float) and not np.isnan(irr):
                        irr_display = f"{irr:.1%}"
                    else:
                        irr_display = "N/A"

                    table += f"| {growth_rate:.1%} | {multiple:.1f}x | {irr_display} | {moic:.2f}x |\n"
                    break

    return table


def format_ipo_sensitivity(sensitivity: Dict) -> str:
    """Format IPO sensitivity analysis for markdown report."""
    table = "\n\n"
    table += "Price Range and Float Analysis:\n\n"
    table += "| Target Float | Price Buffer | Price Range | Float Shares |\n"
    table += "|--------------|--------------|-------------|-------------|\n"

    unique_floats = sorted(set(sensitivity["target_float"]))

    for float_pct in unique_floats:
        idx = sensitivity["target_float"].index(float_pct)

        price_range = f"${sensitivity['price_low'][idx]:,.0f} - ${sensitivity['price_high'][idx]:,.0f}"
        table += f"| {float_pct:.1%} | {sensitivity['price_range_buffer'][idx]:.1%} | {price_range} | {sensitivity['float_shares'][idx]:,.0f} |\n"

    return table


@tool
def get_company_profile(symbol: Annotated[str, "Company symbol"]) -> dict:
    """Get company profile information using FinanceToolkit.

    Args:
        symbol: The company symbol to analyze.

    Returns:
        dict: Company profile information.
    """
    try:
        company = Toolkit(symbol, api_key=os.getenv("FMP_API_KEY"))
        profile = company.get_profile()

        return {"status": "success", "data": profile.to_dict()}
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error getting company profile: {str(e)}",
        }


@tool
def google_search(query: Annotated[str, "Query to search on Google"]) -> str:
    """Perform a Google search using the GenerativeAI API.

    Args:
        query: The search query.

    Returns:
        str: The search results.
    """

    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(contents=query, tools="google_search_retrieval")

    return response.text


@tool
def save_to_markdown(
    content: Annotated[str, "Content to save"],
    file_path: Annotated[str, "Path to save"],
) -> str:
    """Save the input content to a markdown file at the specified path.

    Args:
        content: The content to be saved. Must be a string.
        file_path: The path to the file where the content will be saved.
    """
    with open(file_path, "w") as file:
        file.write(content)

    return f"Content saved to {file_path}"


@tool
def read_from_markdown(
    filepath: Annotated[str, "Path of Strategy Report"],
) -> Annotated[str, "Content of Strategy Report"]:
    """Read the content from a markdown file.

    Args:
        filepath: Path to the markdown file.

    Returns:
        str: Content of the markdown file.
    """
    with open(filepath, "r") as file:
        content = file.read()
    return content


@tool
def read_from_json(file_path: Annotated[str, "Path to JSON file"]) -> dict:
    """Read the content from a JSON file.

    Args:
        file_path: Path to the JSON file to read.

    Returns:
        dict: Content of the JSON file.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        if isinstance(data, list):
            return {"data": data}
        return data


@tool
def save_to_json(
    string: Annotated[str, "String to save as JSON"],
    path: Annotated[str, "Path to save JSON file to"],
) -> None:
    """Save the given string as a JSON file. Conversion is automatically handled.

    Args:
        string: The string to save as JSON.
        path: The path to the file where the JSON string will be saved.
    """
    data = json.loads(string)
    with open(path, "w") as file:
        json.dump(data, file, indent=4)

    return f"Data saved to {path}"


@tool
def get_options(parameter: Annotated[str, "Parameter you want options for"]) -> dict:
    """Retrieve options for a given parameter.

    Args:
        parameter: The parameter to retrieve options for.

    Returns:
        dict: Options for the specified parameter.
    """
    options = fd.obtain_options("equities")
    return {"options": convert_ndarray_to_list(options[parameter])}


def convert_ndarray_to_list(data):
    """Recursively convert numpy ndarrays to lists in a dictionary.

    Args:
        data: The data to convert.

    Returns:
        The converted data with ndarrays as lists.
    """
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: convert_ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [convert_ndarray_to_list(element) for element in data]
    else:
        return data


import json
import os
from typing import Annotated

# Define valid options for each argument
CURRENCY_OPTIONS = [
    "ARS",
    "AUD",
    "BRL",
    "CAD",
    "CHF",
    "CLP",
    "CNY",
    "COP",
    "CZK",
    "DKK",
    "EUR",
    "GBP",
    "HKD",
    "HUF",
    "IDR",
    "ILA",
    "ILS",
    "INR",
    "ISK",
    "JPY",
    "KES",
    "KRW",
    "LKR",
    "MXN",
    "MYR",
    "NOK",
    "NZD",
    "PEN",
    "PHP",
    "PLN",
    "QAR",
    "RUB",
    "SAR",
    "SEK",
    "SGD",
    "THB",
    "TRY",
    "TWD",
    "USD",
    "ZAR",
    "ZAc",
]

SECTOR_OPTIONS = [
    "Communication Services",
    "Consumer Discretionary",
    "Consumer Staples",
    "Energy",
    "Financials",
    "Health Care",
    "Industrials",
    "Information Technology",
    "Materials",
    "Real Estate",
    "Utilities",
]

INDUSTRY_GROUP_OPTIONS = [
    "Automobiles & Components",
    "Banks",
    "Capital Goods",
    "Commercial & Professional Services",
    "Consumer Durables & Apparel",
    "Consumer Services",
    "Diversified Financials",
    "Energy",
    "Food & Staples Retailing",
    "Food, Beverage & Tobacco",
    "Health Care Equipment & Services",
    "Household & Personal Products",
    "Insurance",
    "Materials",
    "Media & Entertainment",
    "Pharmaceuticals, Biotechnology & Life Sciences",
    "Real Estate",
    "Retailing",
    "Semiconductors & Semiconductor Equipment",
    "Software & Services",
    "Technology Hardware & Equipment",
    "Telecommunication Services",
    "Transportation",
    "Utilities",
]

INDUSTRY_OPTIONS = [
    "Aerospace & Defense",
    "Air Freight & Logistics",
    "Airlines",
    "Auto Components",
    "Automobiles",
    "Banks",
    "Beverages",
    "Biotechnology",
    "Building Products",
    "Capital Markets",
    "Chemicals",
    "Commercial Services & Supplies",
    "Communications Equipment",
    "Construction & Engineering",
    "Construction Materials",
    "Consumer Finance",
    "Distributors",
    "Diversified Consumer Services",
    "Diversified Financial Services",
    "Diversified Telecommunication Services",
    "Electric Utilities",
    "Electrical Equipment",
    "Electronic Equipment, Instruments & Components",
    "Energy Equipment & Services",
    "Entertainment",
    "Equity Real Estate Investment Trusts (REITs)",
    "Food & Staples Retailing",
    "Food Products",
    "Gas Utilities",
    "Health Care Equipment & Supplies",
    "Health Care Providers & Services",
    "Health Care Technology",
    "Hotels, Restaurants & Leisure",
    "Household Durables",
    "Household Products",
    "IT Services",
    "Independent Power and Renewable Electricity Producers",
    "Industrial Conglomerates",
    "Insurance",
    "Interactive Media & Services",
    "Internet & Direct Marketing Retail",
    "Machinery",
    "Marine",
    "Media",
    "Metals & Mining",
    "Multi-Utilities",
    "Oil, Gas & Consumable Fuels",
    "Paper & Forest Products",
    "Pharmaceuticals",
    "Professional Services",
    "Real Estate Management & Development",
    "Road & Rail",
    "Semiconductors & Semiconductor Equipment",
    "Software",
    "Specialty Retail",
    "Technology Hardware, Storage & Peripherals",
    "Textiles, Apparel & Luxury Goods",
    "Thrifts & Mortgage Finance",
    "Tobacco",
    "Trading Companies & Distributors",
    "Transportation Infrastructure",
    "Water Utilities",
]

COUNTRY_OPTIONS = [
    "Afghanistan",
    "Anguilla",
    "Argentina",
    "Australia",
    "Austria",
    "Azerbaijan",
    "Bahamas",
    "Bangladesh",
    "Barbados",
    "Belgium",
    "Belize",
    "Bermuda",
    "Botswana",
    "Brazil",
    "British Virgin Islands",
    "Cambodia",
    "Canada",
    "Cayman Islands",
    "Chile",
    "China",
    "Colombia",
    "Costa Rica",
    "Cyprus",
    "Czech Republic",
    "Denmark",
    "Dominican Republic",
    "Egypt",
    "Estonia",
    "Falkland Islands",
    "Finland",
    "France",
    "French Guiana",
    "Gabon",
    "Georgia",
    "Germany",
    "Ghana",
    "Gibraltar",
    "Greece",
    "Greenland",
    "Guernsey",
    "Hong Kong",
    "Hungary",
    "Iceland",
    "India",
    "Indonesia",
    "Ireland",
    "Isle of Man",
    "Israel",
    "Italy",
    "Ivory Coast",
    "Japan",
    "Jersey",
    "Jordan",
    "Kazakhstan",
    "Kenya",
    "Kyrgyzstan",
    "Latvia",
    "Liechtenstein",
    "Lithuania",
    "Luxembourg",
    "Macau",
    "Macedonia",
    "Malaysia",
    "Malta",
    "Mauritius",
    "Mexico",
    "Monaco",
    "Mongolia",
    "Montenegro",
    "Morocco",
    "Mozambique",
    "Myanmar",
    "Namibia",
    "Netherlands",
    "Netherlands Antilles",
    "New Zealand",
    "Nigeria",
    "Norway",
    "Panama",
    "Papua New Guinea",
    "Peru",
    "Philippines",
    "Poland",
    "Portugal",
    "Qatar",
    "Reunion",
    "Romania",
    "Russia",
    "Saudi Arabia",
    "Senegal",
    "Singapore",
    "Slovakia",
    "Slovenia",
    "South Africa",
    "South Korea",
    "Spain",
    "Suriname",
    "Sweden",
    "Switzerland",
    "Taiwan",
    "Tanzania",
    "Thailand",
    "Turkey",
    "Ukraine",
    "United Arab Emirates",
    "United Kingdom",
    "United States",
    "Uruguay",
    "Vietnam",
    "Zambia",
]

MARKET_CAP_OPTIONS = [
    "Large Cap",
    "Mega Cap",
    "Micro Cap",
    "Mid Cap",
    "Nano Cap",
    "Small Cap",
]


@tool
def get_companies(
    path: Annotated[str, "Path to save JSON file"],
    currency: Annotated[str, "Currency"],
    sector: Annotated[str, "Sector"],
    industry_group: Annotated[str, "Industry Group"],
    industry: Annotated[str, "Industry"],
    country: Annotated[str, "Country"],
    market_cap: Annotated[str, "Market Cap"],
) -> str:
    """Filter and save a list of companies based on specified criteria.

    Args:
        path: Path to save JSON file to.
        currency: Currency of companies
        sector: Sector of companies.
        industry_group: Industry group.
        industry: Industry.
        country: Country.
        market_cap: Market capitalization.

    Returns:
        str: Completion message
    """
    if currency not in CURRENCY_OPTIONS:
        raise ValueError(
            f"Invalid currency. Available options: {', '.join(CURRENCY_OPTIONS)}"
        )
    if sector not in SECTOR_OPTIONS:
        raise ValueError(
            f"Invalid sector. Available options: {', '.join(SECTOR_OPTIONS)}"
        )
    if industry_group not in INDUSTRY_GROUP_OPTIONS:
        raise ValueError(
            f"Invalid industry group. Available options: {', '.join(INDUSTRY_GROUP_OPTIONS)}"
        )
    if industry not in INDUSTRY_OPTIONS:
        raise ValueError(
            f"Invalid industry. Available options: {', '.join(INDUSTRY_OPTIONS)}"
        )
    if country not in COUNTRY_OPTIONS:
        raise ValueError(
            f"Invalid country. Available options: {', '.join(COUNTRY_OPTIONS)}"
        )
    if market_cap not in MARKET_CAP_OPTIONS:
        raise ValueError(
            f"Invalid market cap. Available options: {', '.join(MARKET_CAP_OPTIONS)}"
        )

    equities = fd.Equities()
    companies = equities.select()

    filtered_companies = companies.loc[
        (companies["currency"] == currency)
        & (companies["sector"] == sector)
        & (companies["industry_group"] == industry_group)
        & (companies["industry"] == industry)
        & (companies["country"] == country)
        & (companies["market_cap"] == market_cap)
    ]

    filtered_companies = filtered_companies.dropna(subset=["summary"])
    if len(filtered_companies) > 5:
        filtered_companies = filtered_companies.head(5)

    with open(path, "w") as file:
        json.dump(
            json.loads(filtered_companies.reset_index().to_json(orient="records")),
            file,
            indent=4,
        )

    return f"Companies saved to {path}"


@tool
def get_names_and_summaries(path: Annotated[str, "Path to JSON file"]) -> str:
    """Get symbols, names, and summaries of companies from the JSON file.

    Args:
        path: Path to the JSON file containing company data.

    Returns:
        str: JSON string with symbols, names, and summaries.
    """
    try:
        data = read_from_json(path)
        companies = data.get("data", [])
        df = pd.DataFrame(companies, columns=["symbol", "name", "summary"])
        df = df.reset_index(drop=True)
        return df.to_json(orient="records", indent=4)
    except Exception as e:
        return f"Error processing data: {str(e)}"


def get_sp500_tickers():
    """
    Fetches the list of S&P 500 tickers from Wikipedia.

    Returns:
    list: List of S&P 500 tickers.
    """
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, header=0)
    df = table[0]
    return df["Symbol"].tolist()


def get_nyse_tickers():
    """
    Fetches the list of NYSE tickers.

    Returns:
    list: List of NYSE tickers.
    """
    nyse_tickers_url = "https://datahub.io/core/nyse-other-listings/r/nyse-listed.csv"
    df = pd.read_csv(nyse_tickers_url)
    return df["ACT Symbol"].tolist()


def find_company_tickers(nyse_tickers):
    """
    Finds company tickers based on market capitalization and sector and saves all the ticker info in a CSV file.

    Parameters:
    nyse_tickers (list): List of NYSE tickers.


    Returns:
    list: List of company tickers.
    """
    companies = []

    for ticker in nyse_tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info

            companies.append(info)

        except Exception as e:
            print(f"Error fetching data for ticker {ticker}: {e}")

    df = pd.DataFrame(companies)
    df.to_csv("company_tickers.csv", index=False)

    return [company["symbol"] for company in companies], len(companies)


def companies_db_info():
    path = Path("company_db.csv")
    df = pd.read_csv(path)

    result = {}
    for column in ["sectorKey", "industryKey"]:
        result[column] = df[column].unique().tolist()
    result_path = Path("company_db_info.json")
    with result_path.open("w") as f:
        json.dump(result, f, indent=4)
    return result


def clean_db():
    path = Path("company_db.csv")
    df = pd.read_csv(path)

    df["marketCap"] = df["marketCap"].apply(lambda x: x / 1e9 if pd.notnull(x) else x)

    df.dropna(how="all", inplace=True)

    cleaned_path = Path("company_db.csv")
    df.to_csv(cleaned_path, index=False)

    return cleaned_path


@tool
def shortlist_companies(
    sector: Annotated[str, "Company Sector"],
    industry: Annotated[str, "Industry"],
    market_cap_min: Annotated[float, "Minimum Market Cap"],
    market_cap_max: Annotated[float, "Maximum Market Cap"],
) -> pd.DataFrame:
    """
    Shortlists companies from company_db.csv based on the given criteria.

    Args:
        sector: Sector of the companies
        industry: Industry of the companies
        market_cap_min: Minimum market capitalization in billion USD
        market_cap_max: Maximum market capitalization in billion USD

    Returns:
        pd.DataFrame: DataFrame containing the shortlisted companies.
    """
    df = pd.read_csv("tool_data/company_db.csv")

    if sector:
        df = df[df["sectorKey"] == sector]
    if industry:
        df = df[df["industryKey"] == industry]
    if market_cap_min:
        df = df[df["marketCap"] >= market_cap_min]
    if market_cap_max:
        df = df[df["marketCap"] <= market_cap_max]

    columns_to_save = [
        "longName",
        "longBusinessSummary",
        "symbol",
        "marketCap",
        "address1",
        "city",
        "state",
        "zip",
        "country",
        "industryKey",
        "sectorKey",
    ]
    df_selected = df[columns_to_save]
    os.makedirs("outputs", exist_ok=True)

    df_selected.to_csv("outputs/companies.csv", index=False)
    df_selected.to_json("outputs/companies.json", orient="records")

    return df_selected
