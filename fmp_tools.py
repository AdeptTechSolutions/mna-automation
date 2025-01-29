import logging
from typing import Any, Dict, List, Optional

import fmpsdk

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FMPTools:
    """Financial Modeling Prep (FMP) API tools focused on LBO and IPO analysis."""

    def __init__(self, api_key: str):
        """Initialize FMP Tools with API key.

        Args:
            api_key (str): FMP API key
        """
        self.api_key = api_key

    def get_lbo_financials(self, symbol: str) -> Dict[str, Any]:
        """Get financial data required for LBO analysis.

        Args:
            symbol (str): Company stock symbol

        Returns:
            Dict[str, Any]: LBO financial metrics
        """
        try:
            # Get financial statements
            income_stmt = fmpsdk.income_statement(self.api_key, symbol, limit=5)
            balance_sheet = fmpsdk.balance_sheet_statement(
                self.api_key, symbol, limit=5
            )
            cash_flow = fmpsdk.cash_flow_statement(self.api_key, symbol, limit=5)

            if not all([income_stmt, balance_sheet, cash_flow]):
                logger.error(f"Failed to get complete financial data for {symbol}")
                return {}

            # Helper function to safely get positive values
            def safe_positive(value):
                return max(0, value) if value is not None else 0

            financials = {
                "income_statement": {
                    "revenue": [
                        safe_positive(stmt.get("revenue")) for stmt in income_stmt
                    ],
                    "ebitda": [
                        safe_positive(stmt.get("ebitda")) for stmt in income_stmt
                    ],
                    "operating_income": [
                        safe_positive(stmt.get("operatingIncome"))
                        for stmt in income_stmt
                    ],
                    "net_income": [stmt.get("netIncome", 0) for stmt in income_stmt],
                    "income_tax": [
                        stmt.get("incomeTaxExpense", 0) for stmt in income_stmt
                    ],
                },
                "balance_sheet": {
                    "total_debt": [
                        safe_positive(stmt.get("totalDebt")) for stmt in balance_sheet
                    ],
                    "cash": [
                        safe_positive(stmt.get("cashAndCashEquivalents"))
                        for stmt in balance_sheet
                    ],
                    "current_assets": [
                        safe_positive(stmt.get("totalCurrentAssets"))
                        for stmt in balance_sheet
                    ],
                    "current_liabilities": [
                        safe_positive(stmt.get("totalCurrentLiabilities"))
                        for stmt in balance_sheet
                    ],
                    "working_capital": [
                        safe_positive(
                            stmt.get("totalCurrentAssets", 0)
                            - stmt.get("totalCurrentLiabilities", 0)
                        )
                        for stmt in balance_sheet
                    ],
                },
                "cash_flow": {
                    "operating_cash_flow": [
                        stmt.get("operatingCashFlow", 0) for stmt in cash_flow
                    ],
                    "capex": [stmt.get("capitalExpenditure", 0) for stmt in cash_flow],
                    "free_cash_flow": [
                        stmt.get("freeCashFlow", 0) for stmt in cash_flow
                    ],
                },
            }

            # Ensure we have at least some positive cash flows
            if all(fcf <= 0 for fcf in financials["cash_flow"]["free_cash_flow"]):
                logger.warning(
                    f"All free cash flows are non-positive for {symbol}. Using operating cash flow instead."
                )
                financials["cash_flow"]["free_cash_flow"] = financials["cash_flow"][
                    "operating_cash_flow"
                ]

            return financials
        except Exception as e:
            logger.error(f"Error getting LBO financials: {str(e)}")
            return {}

    def get_capital_structure(self, symbol: str) -> Dict[str, Any]:
        """Get capital structure data for LBO analysis.

        Args:
            symbol (str): Company stock symbol

        Returns:
            Dict[str, Any]: Capital structure metrics
        """
        try:
            # Get enterprise value and other metrics
            key_metrics = fmpsdk.key_metrics(self.api_key, symbol, limit=1)
            enterprise_values = fmpsdk.enterprise_values(self.api_key, symbol, limit=1)

            if not all([key_metrics, enterprise_values]):
                logger.error(f"Failed to get capital structure data for {symbol}")
                return {}

            ev_data = enterprise_values[0] if enterprise_values else {}
            return {
                "enterprise_value": ev_data.get("enterpriseValue"),
                "shares_outstanding": ev_data.get("numberOfShares"),
                "market_cap": ev_data.get("marketCapitalization"),
                "total_debt": ev_data.get("totalDebt"),
                "net_debt": ev_data.get("netDebt"),
            }
        except Exception as e:
            logger.error(f"Error getting capital structure: {str(e)}")
            return {}

    def get_ipo_metrics(self, symbol: str) -> Dict[str, Any]:
        """Get metrics required for IPO analysis.

        Args:
            symbol (str): Company stock symbol

        Returns:
            Dict[str, Any]: IPO relevant metrics
        """
        try:
            # Get various metrics
            profile = fmpsdk.company_profile(self.api_key, symbol)
            key_metrics = fmpsdk.key_metrics(self.api_key, symbol, limit=5)
            growth = fmpsdk.financial_growth(self.api_key, symbol, limit=5)
            ratios = fmpsdk.financial_ratios(self.api_key, symbol, limit=5)

            if not all([profile, key_metrics, growth, ratios]):
                logger.error(f"Failed to get complete IPO metrics for {symbol}")
                return {}

            profile_data = profile[0] if profile else {}
            return {
                "company_profile": {
                    "sector": profile_data.get("sector"),
                    "industry": profile_data.get("industry"),
                    "market_cap": profile_data.get("mktCap"),
                },
                "valuation_metrics": {
                    "pe_ratio": [metric.get("peRatio") for metric in key_metrics],
                    "ev_to_ebitda": [
                        metric.get("enterpriseValueOverEBITDA")
                        for metric in key_metrics
                    ],
                    "ev_to_sales": [
                        metric.get("enterpriseValueOverRevenue")
                        for metric in key_metrics
                    ],
                    "debt_to_equity": [
                        metric.get("debtToEquity") for metric in key_metrics
                    ],
                },
                "growth_metrics": {
                    "revenue_growth": [g.get("revenueGrowth") for g in growth],
                    "ebit_growth": [g.get("ebitgrowth") for g in growth],
                    "eps_growth": [g.get("epsgrowth") for g in growth],
                },
                "financial_ratios": {
                    "price_to_book": [
                        ratio.get("priceToBookRatio") for ratio in ratios
                    ],
                    "price_to_sales": [
                        ratio.get("priceToSalesRatio") for ratio in ratios
                    ],
                    "roe": [ratio.get("returnOnEquity") for ratio in ratios],
                    "roa": [ratio.get("returnOnAssets") for ratio in ratios],
                },
            }
        except Exception as e:
            logger.error(f"Error getting IPO metrics: {str(e)}")
            return {}

    def get_peer_analysis(self, symbol: str) -> Dict[str, Any]:
        """Get peer analysis data for IPO comparables.

        Args:
            symbol (str): Company stock symbol

        Returns:
            Dict[str, Any]: Peer analysis data
        """
        try:
            # Get peer data
            # Get company profile which includes industry/sector for finding peers
            profile = fmpsdk.company_profile(self.api_key, symbol)
            if not profile:
                logger.error(f"Failed to get company profile for {symbol}")
                return {}

            # Use sector/industry to get comparable companies
            sector = profile[0].get("sector", "")
            industry = profile[0].get("industry", "")

            # Get financial ratios for PE and other relevant data
            financial_ratios = fmpsdk.financial_ratios(self.api_key, symbol, limit=1)
            if not financial_ratios:
                logger.error(f"Failed to get financial ratios for {symbol}")
                return {}
            pe_ratio = financial_ratios[0].get(
                "priceEarningsRatioTTM", 0
            )  # Extract PE ratio

            return {
                "peer_companies": [symbol],  # Start with the company itself for now
                "company_sector": sector,
                "company_industry": industry,
                "sector_multiples": {
                    sector: pe_ratio
                },  # Use company's PE as proxy for sector
            }
        except Exception as e:
            logger.error(f"Error getting peer analysis: {str(e)}")
            return {}

    def get_complete_lbo_data(self, symbol: str) -> Dict[str, Any]:
        """Get all data required for LBO analysis.

        Args:
            symbol (str): Company stock symbol

        Returns:
            Dict[str, Any]: Complete LBO analysis data
        """
        return {
            "financials": self.get_lbo_financials(symbol),
            "capital_structure": self.get_capital_structure(symbol),
        }

    def get_complete_ipo_data(self, symbol: str) -> Dict[str, Any]:
        """Get all data required for IPO analysis.

        Args:
            symbol (str): Company stock symbol

        Returns:
            Dict[str, Any]: Complete IPO analysis data
        """
        return {
            "metrics": self.get_ipo_metrics(symbol),
            "peer_analysis": self.get_peer_analysis(symbol),
        }
