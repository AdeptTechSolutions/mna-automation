import json
import os
from pathlib import Path
from typing import Dict

import financedatabase as fd
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from financetoolkit import Toolkit
from typing_extensions import Annotated

from finmodels_tools import FinModelsTools

load_dotenv()


def collect_financial_metrics(
    symbol: Annotated[str, "Company symbol to analyze"],
    output_dir: Annotated[str, "Directory to save the output"] = "outputs/fmp_data",
) -> dict:
    """Collect key financial metrics using FinanceToolkit.

    Args:
        symbol: The company symbol to analyze.
        output_dir: Directory to save the output files.

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

        output_path = Path(output_dir) / f"{symbol}_metrics.md"
        with open(output_path, "w") as f:
            f.write(report)

        return {"status": "success", "message": f"Metrics saved to {output_path}"}
    except Exception as e:
        return {"status": "error", "message": f"Error collecting metrics: {str(e)}"}


def perform_valuation_analysis(
    symbol: Annotated[str, "Company symbol to analyze"]
) -> dict:
    """Perform comprehensive valuation analysis using FinanceToolkit."""
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

    table = "### LBO Sensitivity Analysis\n\n"
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
    table = "### IPO Sensitivity Analysis\n\n"
    table += "Price Range and Float Analysis:\n\n"
    table += "| Target Float | Price Buffer | Price Range | Float Shares |\n"
    table += "|--------------|--------------|-------------|-------------|\n"

    unique_floats = sorted(set(sensitivity["target_float"]))

    for float_pct in unique_floats:
        idx = sensitivity["target_float"].index(float_pct)

        price_range = f"${sensitivity['price_low'][idx]:,.0f} - ${sensitivity['price_high'][idx]:,.0f}"
        table += f"| {float_pct:.1%} | {sensitivity['price_range_buffer'][idx]:.1%} | {price_range} | {sensitivity['float_shares'][idx]:,.0f} |\n"

    return table


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


def read_companies_list(path: Annotated[str, "Path to companies JSON"]) -> list:
    """Read list of companies from JSON file.

    Args:
        path (str): Path to JSON file

    Returns:
        list: List of company data
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_comparison_table(valuations: list) -> str:
    """Generate markdown comparison table from valuations.

    Args:
        valuations (list): List of valuation summaries

    Returns:
        str: Markdown formatted table
    """
    try:
        df = pd.DataFrame(valuations)
        df = df[df.columns[~df.columns.isin(["status"])]]  # Remove status column
        return df.to_markdown(index=False, floatfmt=".2f")
    except Exception as e:
        return f"Error generating table: {str(e)}"


def read_companies_list(path: Annotated[str, "Path to companies JSON"]) -> list:
    """Read list of companies from a JSON file.

    Args:
        path (str): Path to the companies JSON file.

    Returns:
        list: List of companies or an error message.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data
    except Exception as e:
        return {"status": "error", "message": str(e)}


def generate_comparison_table(valuations: list) -> str:
    """Generate a markdown comparison table from valuations.

    Args:
        valuations (list): List of valuation dictionaries.

    Returns:
        str: Markdown table as a string or an error message.
    """
    try:
        df = pd.DataFrame(valuations)
        markdown_table = df.to_markdown(index=False)
        return markdown_table
    except Exception as e:
        return f"Error generating table: {str(e)}"


def save_response_json(
    response_json: Annotated[str, "JSON String to save"],
    path: Annotated[str, "Path to save"],
) -> None:
    """Save the given JSON string to a file.

    Args:
        response_json (str): The JSON string to be saved.
        path (str): The path to the file where the JSON string will be saved.
    """
    data = json.loads(response_json)
    with open(path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"JSON response saved to {path}")

    return f"JSON response saved to {path}"


def google_search(query: Annotated[str, "Query to search on Google"]) -> str:
    """Perform a Google search using the GenerativeAI API.

    Args:
        query (str): The search query.

    Returns:
        str: The search results.
    """

    import google.generativeai as genai

    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

    model = genai.GenerativeModel("gemini-1.5-pro")
    response = model.generate_content(contents=query, tools="google_search_retrieval")

    return response.text


def save_to_markdown(
    content: Annotated[str, "Content to save"],
    file_path: Annotated[str, "Path to save markdown file"] = "outputs/output.md",
) -> str:
    """Save the given content to a markdown file, overwriting if it exists.

    Args:
        content (str): The content to be saved.
        file_path (str, optional): Path to the markdown file. Defaults to "outputs/output.md".
    """
    with open(file_path, "w") as file:
        file.write(content)

    return f"Content saved to {file_path}"


def read_from_markdown(
    filepath: Annotated[str, "Path of Strategy Report"]
) -> Annotated[str, "Content of Strategy Report"]:
    """Read the content from a markdown file.

    Args:
        filepath (str): Path to the markdown file.

    Returns:
        str: Content of the markdown file.
    """
    with open(filepath, "r") as file:
        content = file.read()
    return content


def save_json_to_disk(
    data: dict, file_path: Annotated[str, "Path to save JSON file"] = "data.json"
) -> None:
    """Save the given dictionary as a JSON file.

    Args:
        data (dict): The dictionary to be saved.
        file_path (str, optional): Path to the JSON file. Defaults to "data.json".
    """
    with open(file_path, "w") as file:
        json.dump(data, file, indent=4)
    print(f"JSON data written to {file_path}")


def read_json_from_disk(file_path: Annotated[str, "Path to JSON file"]) -> dict:
    """Read the content from a JSON file.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Content of the JSON file.
    """
    with open(file_path, "r") as file:
        data = json.load(file)
        if isinstance(data, list):
            return {"data": data}
        return data


def get_options(parameter: Annotated[str, "Parameter you want options for"]) -> dict:
    """Retrieve options for a given parameter.

    Args:
        parameter (str): The parameter to retrieve options for.

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


def get_companies(
    currency: Annotated[str, "Currency"] = "USD",
    sector: Annotated[str, "Sector"] = "Information Technology",
    industry_group: Annotated[str, "Industry Group"] = "Software & Services",
    industry: Annotated[str, "Industry"] = "Software",
    exchange: Annotated[str, "Exchange"] = "NASDAQ",
    market: Annotated[str, "Market"] = "us_market",
    country: Annotated[str, "Country"] = "United States",
    market_cap: Annotated[str, "Market Cap"] = "Small Cap",
    path: Annotated[str, "Path to save JSON file"] = "companies.json",
) -> str:
    """Filter and save a list of companies based on specified criteria.

    Args:
        currency (str, optional): Currency of companies. Defaults to "USD".
        sector (str, optional): Sector of companies. Defaults to "Information Technology".
        industry_group (str, optional): Industry group. Defaults to "Software & Services".
        industry (str, optional): Industry. Defaults to "Software".
        exchange (str, optional): Stock exchange. Defaults to "NASDAQ".
        market (str, optional): Market. Defaults to "us_market".
        country (str, optional): Country. Defaults to "United States".
        market_cap (str, optional): Market capitalization. Defaults to "Small Cap".
        path (str, optional): Path to save JSON file. Defaults to "companies.json".

    Returns:
        str: Completion message.
    """
    equities = fd.Equities()
    companies = equities.select()

    # Filter the DataFrame based on specific values of multiple columns using .loc
    filtered_companies = companies.loc[
        (companies["currency"] == currency)
        & (companies["sector"] == sector)
        & (companies["industry_group"] == industry_group)
        & (companies["industry"] == industry)
        & (companies["country"] == country)
        & (companies["market_cap"] == market_cap)
    ]

    filtered_companies = filtered_companies.dropna(subset=["summary"])

    with open(path, "w") as file:
        json.dump(
            json.loads(filtered_companies.reset_index().to_json(orient="records")),
            file,
            indent=4,
        )

    return f"Companies saved to {path}"


def get_number_of_companies(path: Annotated[str, "Path to JSON file"]) -> int:
    """Get the number of companies in the JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        int: Number of companies.
    """
    companies = read_json_from_disk(path)
    return len(companies)


def get_names_and_summaries(path: Annotated[str, "Path to JSON file"]) -> str:
    """Get symbols, names, and summaries of companies from the JSON file.

    Args:
        path (str): Path to the JSON file.

    Returns:
        str: JSON string with symbols, names, and summaries.
    """
    try:
        data = read_json_from_disk(path)
        companies = data.get("data", [])
        df = pd.DataFrame(companies, columns=["symbol", "name", "summary"])
        df = df.reset_index(drop=True)
        return df.to_json(orient="records", indent=4)
    except Exception as e:
        return f"Error processing data: {str(e)}"


def collect_and_save_fmp_data(
    symbol: Annotated[str, "Symbol for which data needs to be collected"],
    path: Annotated[str, "Path to save collected data"],
) -> str:
    """Collect FMP data and save it, returning a status message.

    Args:
        symbol (str): Symbol to collect data for.
        path (str): Path to save the collected data.

    Returns:
        str: Status message indicating success or error.
    """
    api_key = os.getenv("FMP_API_KEY")
    if not api_key:
        return (
            "FMP API key not found. Please set the 'FMP_API_KEY' environment variable."
        )
    base_url = "https://financialmodelingprep.com/api/v3"

    endpoints = {
        "financials": f"/financials/income-statement/{symbol}",
        "balance_sheet": f"/financials/balance-sheet-statement/{symbol}",
        "cash_flow": f"/financials/cash-flow-statement/{symbol}",
        "ratios": f"/key-ratios/{symbol}",
        "quote": f"/quote/{symbol}",
        "sector_performance": "/sector-performance",
        "macro_data": "/economic-indicator",
    }

    data = {}
    try:
        for key, endpoint in endpoints.items():
            url = f"{base_url}{endpoint}?apikey={api_key}"
            response = requests.get(url)
            response.raise_for_status()
            data[key] = response.json()

        base_path = str(Path(path))
        os.makedirs(base_path, exist_ok=True)
        filepath = Path(base_path) / f"{symbol}_fmp.json"

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        return f"Successfully collected and saved FMP data for {symbol}"
    except Exception as e:
        return f"Error processing {symbol}: {str(e)}"


if __name__ == "__main__":
    result = read_json_from_disk(
        "/home/amadgakkhar/code/mna-multi-agent/autogen/outputs/critic_companies.json"
    )
    print(result)
