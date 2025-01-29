import logging
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np
import numpy_financial as npf

from fmp_tools import FMPTools

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FinModelsTools:
    """Financial modeling tools for LBO and IPO analysis."""

    def __init__(self, fmp_api_key: str):
        """Initialize with FMP API key.

        Args:
            fmp_api_key (str): Financial Modeling Prep API key
        """
        self.fmp = FMPTools(fmp_api_key)

    def calculate_lbo_metrics(
        self,
        symbol: str,
        holding_period: int = 5,
        target_irr: float = 0.20,
        debt_ratio: float = 0.7,
        interest_rate: float = 0.08,
        fcf_growth_rate: float = 0.10,
        debt_repayment_pct: float = 0.30,
        exit_multiple: float = 10.0,
        exit_ebitda_growth: float = 0.08,
    ) -> Dict[str, Any]:
        """Calculate LBO analysis metrics."""
        data = self.fmp.get_complete_lbo_data(symbol)
        if not data:
            logger.error(f"Failed to get LBO data for {symbol}")
            return {}

        try:
            ev = data["capital_structure"]["enterprise_value"]
            ebitda = data["financials"]["income_statement"]["ebitda"][0]
            fcf = data["financials"]["cash_flow"]["free_cash_flow"][0]

            debt = ev * debt_ratio
            equity = ev - debt

            projected_fcf = []
            remaining_debt = debt
            for year in range(holding_period):
                fcf_growth = fcf * (1 + fcf_growth_rate) ** year
                debt_repayment = min(fcf_growth * debt_repayment_pct, remaining_debt)
                remaining_debt -= debt_repayment
                interest_expense = remaining_debt * interest_rate
                free_cash_after_debt = fcf_growth - interest_expense - debt_repayment
                projected_fcf.append(free_cash_after_debt)

            exit_ebitda = ebitda * (1 + exit_ebitda_growth) ** holding_period
            exit_value = exit_ebitda * exit_multiple
            exit_equity = exit_value - remaining_debt

            cash_flows = [-equity] + projected_fcf + [exit_equity]

            print(f"Cash Flow Stream for IRR: {cash_flows}")
            print(f"Total Investment: {equity}")
            print(f"Projected FCF: {projected_fcf}")
            print(f"Exit Equity: {exit_equity}")

            try:
                irr = npf.irr(cash_flows)
                if np.isnan(irr):
                    irr = "Unable to calculate (nan)"
                    print(
                        "Warning: IRR calculation resulted in 'nan'. This may be due to invalid cash flows."
                    )
                    print("Cash flow signs:", [np.sign(cf) for cf in cash_flows])
                    print(
                        "Are all cash flows negative?",
                        all(cf <= 0 for cf in cash_flows),
                    )
                else:
                    irr = float(irr)
            except Exception as e:
                irr = f"Error: {str(e)}"
                print(f"Error in IRR calculation: {str(e)}")

            moic = exit_equity / equity if equity != 0 else "N/A (division by zero)"

            return {
                "purchase_price": {
                    "enterprise_value": ev,
                    "equity_contribution": equity,
                    "debt_financing": debt,
                    "debt_ratio": debt_ratio,
                },
                "projections": {
                    "projected_fcf": projected_fcf,
                    "exit_value": exit_value,
                    "remaining_debt": remaining_debt,
                },
                "returns": {"irr": irr, "moic": moic, "exit_equity": exit_equity},
            }
        except Exception as e:
            logger.error(f"Error in LBO calculation: {str(e)}")
            return {}

    def calculate_ipo_valuation(
        self, symbol: str, target_float: float = 0.20, price_range_buffer: float = 0.15
    ) -> Dict[str, Any]:
        """Calculate IPO valuation metrics.

        Args:
            symbol (str): Company stock symbol
            target_float (float): Target percentage of shares to float
            price_range_buffer (float): Buffer for price range calculation

        Returns:
            Dict[str, Any]: IPO valuation results
        """
        # Get company data
        data = self.fmp.get_complete_ipo_data(symbol)
        if not data:
            logger.error(f"Failed to get IPO data for {symbol}")
            return {}

        try:
            metrics = data["metrics"]
            peer_data = data["peer_analysis"]

            # Get latest metrics
            revenue = metrics["valuation_metrics"]["ev_to_sales"][0] or 0
            ebitda = metrics["valuation_metrics"]["ev_to_ebitda"][0] or 0
            pe_ratio = metrics["valuation_metrics"]["pe_ratio"][0]

            # Get growth metrics
            revenue_growth = metrics["growth_metrics"]["revenue_growth"][0]
            ebit_growth = metrics["growth_metrics"]["ebit_growth"][0]

            # Calculate peer multiples
            peer_companies = peer_data["peer_companies"]
            sector_pe = peer_data["sector_multiples"].get(
                metrics["company_profile"]["sector"]
            )
            if sector_pe is None or sector_pe == 0:
                sector_pe = (
                    pe_ratio if pe_ratio and pe_ratio != 0 else 15
                )  # Default to 15 if no valid PE ratio

            # Calculate enterprise value using multiple methods
            ev_revenue = (revenue * sector_pe) if revenue and sector_pe else 0
            ev_ebitda = (ebitda * 10) if ebitda else 0  # Standard EBITDA multiple

            print(
                f"Debug - sector_pe: {sector_pe}, ev_revenue: {ev_revenue}, ev_ebitda: {ev_ebitda}"
            )

            if ev_revenue and ev_ebitda:
                # Take weighted average based on growth
                if revenue_growth > 0.3:  # High growth company
                    enterprise_value = (ev_revenue * 0.7) + (ev_ebitda * 0.3)
                else:  # Mature company
                    enterprise_value = (ev_revenue * 0.3) + (ev_ebitda * 0.7)
            elif ev_revenue:
                enterprise_value = ev_revenue
            else:
                enterprise_value = ev_ebitda

            # Calculate equity value
            debt = metrics["valuation_metrics"]["debt_to_equity"][0]
            equity_value = enterprise_value - debt

            # Calculate share price range
            shares_outstanding = equity_value / metrics["company_profile"]["market_cap"]
            float_shares = shares_outstanding * target_float

            base_price = equity_value / shares_outstanding
            price_low = base_price * (1 - price_range_buffer)
            price_high = base_price * (1 + price_range_buffer)

            # Calculate multiples, handling potential division by zero
            ev_revenue_multiple = (
                ev_revenue / revenue if revenue and revenue != 0 else "N/A"
            )
            ev_ebitda_multiple = ev_ebitda / ebitda if ebitda and ebitda != 0 else "N/A"

            return {
                "valuation": {
                    "enterprise_value": enterprise_value,
                    "equity_value": equity_value,
                    "ev_revenue_multiple": ev_revenue_multiple,
                    "ev_ebitda_multiple": ev_ebitda_multiple,
                },
                "offering": {
                    "shares_outstanding": shares_outstanding,
                    "float_shares": float_shares,
                    "price_range": {
                        "low": price_low,
                        "base": base_price,
                        "high": price_high,
                    },
                },
                "comparables": {
                    "sector_pe": sector_pe,
                    "peer_companies": peer_companies,
                    "revenue_growth": revenue_growth,
                    "ebit_growth": ebit_growth,
                },
            }
        except Exception as e:
            logger.error(f"Error in IPO valuation: {str(e)}")
            return {}

    def perform_lbo_sensitivity_analysis(
        self,
        symbol: str,
        fcf_growth_rates: List[float] = [0.05, 0.08, 0.10, 0.12, 0.15],
        exit_multiples: List[float] = [8.0, 10.0, 12.0, 14.0, 16.0],
        interest_rates: List[float] = [0.06, 0.07, 0.08, 0.09, 0.10],
        holding_period: int = 5,
        target_irr: float = 0.20,
        debt_ratio: float = 0.7,
        debt_repayment_pct: float = 0.30,
        exit_ebitda_growth: float = 0.08,
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis for LBO analysis.

        Args:
            symbol (str): Company stock symbol
            fcf_growth_rates (List[float]): Range of FCF growth rates to analyze
            exit_multiples (List[float]): Range of exit multiples to analyze
            interest_rates (List[float]): Range of interest rates to analyze
            holding_period (int): Investment holding period in years
            target_irr (float): Target internal rate of return
            debt_ratio (float): Initial debt to enterprise value ratio
            debt_repayment_pct (float): Debt repayment percentage of FCF
            exit_ebitda_growth (float): Exit EBITDA growth rate

        Returns:
            Dict[str, Any]: Sensitivity analysis results
        """
        # Get company data
        data = self.fmp.get_complete_lbo_data(symbol)
        if not data:
            logger.error(f"Failed to get LBO data for {symbol}")
            return {}

        try:
            # Extract key metrics
            ev = data["capital_structure"]["enterprise_value"]
            ebitda = data["financials"]["income_statement"]["ebitda"][
                0
            ]  # Most recent year
            fcf = data["financials"]["cash_flow"]["free_cash_flow"][0]

            # Initialize results dictionary
            results = {
                "fcf_growth_rate": [],
                "exit_multiple": [],
                "interest_rate": [],
                "irr": [],
                "moic": [],
            }

            # Perform sensitivity analysis
            for growth_rate in fcf_growth_rates:
                for multiple in exit_multiples:
                    for rate in interest_rates:
                        # Calculate purchase price components
                        debt = ev * debt_ratio
                        equity = ev - debt

                        # Project cash flows
                        projected_fcf = []
                        remaining_debt = debt
                        for year in range(holding_period):
                            # Apply specified FCF growth rate
                            fcf_growth = fcf * (1 + growth_rate) ** year

                            # Cap debt repayment at specified percentage of FCF
                            debt_repayment = min(
                                fcf_growth * debt_repayment_pct, remaining_debt
                            )
                            remaining_debt -= debt_repayment

                            # Calculate interest expense
                            interest_expense = remaining_debt * rate

                            # Calculate free cash after debt service
                            # Allow for negative values up to available cash balance
                            free_cash_after_debt = (
                                fcf_growth - interest_expense - debt_repayment
                            )
                            available_cash = data["financials"]["balance_sheet"][
                                "cash"
                            ][0]
                            free_cash_after_debt = max(
                                free_cash_after_debt, -available_cash
                            )

                            projected_fcf.append(free_cash_after_debt)

                        # Calculate exit value using specified multiple and growth rate
                        exit_ebitda = (
                            ebitda * (1 + exit_ebitda_growth) ** holding_period
                        )
                        exit_value = exit_ebitda * multiple

                        # Calculate returns
                        total_investment = equity
                        exit_equity = exit_value - remaining_debt

                        # Calculate IRR
                        cash_flows = [-total_investment] + projected_fcf + [exit_equity]
                        irr = npf.irr(cash_flows)

                        # Calculate MOIC (Multiple of Invested Capital)
                        moic = exit_equity / total_investment

                        # Store results
                        results["fcf_growth_rate"].append(growth_rate)
                        results["exit_multiple"].append(multiple)
                        results["interest_rate"].append(rate)
                        results["irr"].append(irr)
                        results["moic"].append(moic)

            return results
        except Exception as e:
            logger.error(f"Error in LBO sensitivity analysis: {str(e)}")
            return {}

    def perform_ipo_sensitivity_analysis(
        self,
        symbol: str,
        target_floats: List[float] = [0.15, 0.20, 0.25, 0.30, 0.35],
        price_range_buffers: List[float] = [0.10, 0.15, 0.20, 0.25, 0.30],
    ) -> Dict[str, Any]:
        """Perform sensitivity analysis for IPO analysis.

        Args:
            symbol (str): Company stock symbol
            target_floats (List[float]): Range of target float percentages to analyze
            price_range_buffers (List[float]): Range of price range buffers to analyze

        Returns:
            Dict[str, Any]: Sensitivity analysis results
        """
        # Get company data
        data = self.fmp.get_complete_ipo_data(symbol)
        if not data:
            logger.error(f"Failed to get IPO data for {symbol}")
            return {}

        try:
            metrics = data["metrics"]
            peer_data = data["peer_analysis"]

            # Get latest metrics
            revenue = metrics["valuation_metrics"]["ev_to_sales"][0] or 0
            ebitda = metrics["valuation_metrics"]["ev_to_ebitda"][0] or 0
            pe_ratio = metrics["valuation_metrics"]["pe_ratio"][0]

            # Get growth metrics
            revenue_growth = metrics["growth_metrics"]["revenue_growth"][0]
            ebit_growth = metrics["growth_metrics"]["ebit_growth"][0]

            # Calculate peer multiples
            peer_companies = peer_data["peer_companies"]
            sector_pe = peer_data["sector_multiples"].get(
                metrics["company_profile"]["sector"]
            )
            if sector_pe is None:
                sector_pe = pe_ratio if pe_ratio else 15  # Default to 15 if no PE ratio

            # Calculate enterprise value using multiple methods
            ev_revenue = (revenue * sector_pe) if revenue else 0
            ev_ebitda = (ebitda * 10) if ebitda else 0  # Standard EBITDA multiple

            if ev_revenue and ev_ebitda:
                # Take weighted average based on growth
                if revenue_growth > 0.3:  # High growth company
                    enterprise_value = (ev_revenue * 0.7) + (ev_ebitda * 0.3)
                else:  # Mature company
                    enterprise_value = (ev_revenue * 0.3) + (ev_ebitda * 0.7)
            elif ev_revenue:
                enterprise_value = ev_revenue
            else:
                enterprise_value = ev_ebitda

            # Initialize results dictionary
            results = {
                "target_float": [],
                "price_range_buffer": [],
                "equity_value": [],
                "price_low": [],
                "price_base": [],
                "price_high": [],
                "float_shares": [],
            }

            # Perform sensitivity analysis
            for target_float in target_floats:
                for buffer in price_range_buffers:
                    # Calculate equity value
                    debt = metrics["valuation_metrics"]["debt_to_equity"][0]
                    equity_value = enterprise_value - debt

                    # Calculate share price range
                    shares_outstanding = (
                        equity_value / metrics["company_profile"]["market_cap"]
                    )
                    float_shares = shares_outstanding * target_float

                    base_price = equity_value / shares_outstanding
                    price_low = base_price * (1 - buffer)
                    price_high = base_price * (1 + buffer)

                    # Store results
                    results["target_float"].append(target_float)
                    results["price_range_buffer"].append(buffer)
                    results["equity_value"].append(equity_value)
                    results["price_low"].append(price_low)
                    results["price_base"].append(base_price)
                    results["price_high"].append(price_high)
                    results["float_shares"].append(float_shares)

            return results
        except Exception as e:
            logger.error(f"Error in IPO sensitivity analysis: {str(e)}")
            return {}
