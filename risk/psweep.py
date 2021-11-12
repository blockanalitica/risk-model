from datetime import datetime
import logging

import numpy as np
from numpy import arange
import pandas as pd
import psweep as ps

log = logging.getLogger(__name__)


def compute_cr_distribution(liquidation_ratio, df_vaults):
    buf_range = [
        0.15,
        0.25,
        0.5,
        0.75,
        1.0,
        1.25,
        1.5,
        1.75,
        2.0,
        2.25,
        2.5,
        2.75,
        3.0,
        3.25,
        3.5,
        3.75,
    ]
    cr_range = [round(i + liquidation_ratio, 2) for i in buf_range]

    lim_threshold = buf_range[-1] + liquidation_ratio

    df_vaults["collateralization_lim"] = np.where(
        df_vaults["collateralization"] <= lim_threshold,
        df_vaults["collateralization"],
        lim_threshold,
    )

    cr_dist = (
        pd.cut(
            df_vaults["collateralization_lim"],
            bins=cr_range,
            labels=cr_range[:-1],
            right=True,
        )
        .value_counts()
        .sort_index()
    )
    cr_dist = (
        pd.DataFrame(cr_dist)
        .reset_index()
        .rename(
            columns={"index": "cr_bucket", "collateralization_lim": "vault_quantity"}
        )
    )
    cr_dist["buffer"] = buf_range[:-1]

    np_cr_range = np.array(cr_range)
    vault_cr_buckets = {}
    for vault_id, limit in zip(
        df_vaults["vault_id"], df_vaults["collateralization_lim"]
    ):
        greater_limit_indexes = np_cr_range > limit
        if any(greater_limit_indexes):
            vault_cr_buckets[vault_id] = np_cr_range[np.argmax(greater_limit_indexes)]

    df_vaults["cr_bucket"] = df_vaults["vault_id"].map(vault_cr_buckets)

    cr_dist["cr_bucket"] = cr_dist["cr_bucket"].astype("float64")

    df_vaults["cr_bucket"] = np.clip(
        df_vaults["cr_bucket"], a_max=5, a_min=None
    ).fillna(5)

    grouped_debt = df_vaults.groupby("cr_bucket")["total_debt_dai"].sum().to_dict()
    cr_dist["total_debt_dai"] = cr_dist["cr_bucket"].map(grouped_debt)
    cr_dist["total_debt_dai"] = cr_dist["total_debt_dai"].astype("float64")
    cr_dist["total_debt_dai_pdf"] = (
        cr_dist["total_debt_dai"] / cr_dist["total_debt_dai"].sum()
    )
    return cr_dist


class Precompute:
    def __init__(self, max_slippage=0.8, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scenario_cr_dists_cache = {}
        self.max_slippage = max_slippage

        # setting parameter sets
        self.jump_severity_list = list(np.around(arange(-0.25, -0.7 - 0.05, -0.05), 2))
        self.jump_frequency_list = list(range(1, 5 + 1, 1))
        self.keeper_profit_list = [0.01, 0.025, 0.05, 0.075, 0.1]

        jump_frequency = ps.plist(
            "jump_frequency", [i for i in self.jump_frequency_list]
        )
        jump_severity = ps.plist(
            "jump_severity", [round(i, 2) for i in self.jump_severity_list]
        )
        keeper_profit = ps.plist("keeper_profit", self.keeper_profit_list)
        self.psets = ps.pgrid([jump_severity, jump_frequency, keeper_profit])

    def compute_scenario_cr_distribution_psweep_asset(
        self, simulation_params, is_base_case, df_vaults, params
    ):
        if is_base_case:
            cache_key = hash(str(simulation_params["liquidation_ratio"]))
            if cache_key in self.scenario_cr_dists_cache:
                # DataFrames by default are mutable!
                scenario_cr_dist = self.scenario_cr_dists_cache[cache_key].copy()
            else:
                # compute the CR distribution
                scenario_cr_dist = compute_cr_distribution(
                    simulation_params["liquidation_ratio"],
                    df_vaults=df_vaults,
                )
                self.scenario_cr_dists_cache[cache_key] = scenario_cr_dist.copy()

        else:
            cache_key = hash(
                "".join(
                    [
                        str(simulation_params["liquidation_ratio"]),
                        str(params["cr_distribution"]),
                    ]
                )
            )
            if cache_key in self.scenario_cr_dists_cache:
                # DataFrames by default are mutable!
                scenario_cr_dist = self.scenario_cr_dists_cache[cache_key].copy()
            else:
                # take the CR distribution from the dict
                scenario_cr_dist = (
                    pd.DataFrame.from_dict(params["cr_distribution"], orient="index")
                    .reset_index()
                    .rename(columns={"index": "buffer", 0: "total_debt_dai_pdf"})
                )
                scenario_cr_dist["cr_bucket"] = (
                    scenario_cr_dist["buffer"] + simulation_params["liquidation_ratio"]
                )
                scenario_cr_dist["total_debt_dai_pdf"] = (
                    scenario_cr_dist["total_debt_dai_pdf"] / 100
                )
                self.scenario_cr_dists_cache[cache_key] = scenario_cr_dist.copy()

        scenario_cr_dist["liquidation"] = np.where(
            (scenario_cr_dist["cr_bucket"] * (1 + params["jump_severity"]))
            <= simulation_params["liquidation_ratio"],
            1,
            0,
        )

        scenario_cr_dist["liquidated_debt"] = (
            scenario_cr_dist["liquidation"]
            * scenario_cr_dist["total_debt_dai_pdf"]
            * simulation_params["simulate_de"]
            * (1 - params["share_vaults_protected"])
        ).round(2)

        return scenario_cr_dist

    def compute_scenario_cr_distribution_psweep(
        self, simulation_params, scenario_params, asset_vault_types_dict
    ):

        # compute scenario_cr_dist per asset for each of the vault types
        scenario_cr_dists = {}
        for asset_vault_type in asset_vault_types_dict.keys():
            scenario_cr_dists[
                asset_vault_type
            ] = self.compute_scenario_cr_distribution_psweep_asset(
                simulation_params[asset_vault_type],
                scenario_params["scenario_name"] == "base_case",
                asset_vault_types_dict[asset_vault_type]["df_vaults"],
                scenario_params["params"],
            )
        return scenario_cr_dists

    def get_simulation_spec(
        self,
        vault_type,
        asset_vault_types_dict,
    ):
        simulation_params = dict()
        for asset_vault_type, data in asset_vault_types_dict.items():
            simulate_de = "To simulate"
            if asset_vault_type != vault_type:
                simulate_de = int(data["df_vaults"]["total_debt_dai"].sum())

            simulation_params[asset_vault_type] = {
                "liquidation_ratio": float(data["liquidation_ratio"]),
                "simulate_de": simulate_de,
            }
        return simulation_params

    def compute_scenario_params_psweep(self, parameter_set, share_vaults_protected):
        scenario_params = [
            {
                "scenario_name": "base_case",
                "params": {
                    "jump_frequency": parameter_set["jump_frequency"],
                    "jump_severity": parameter_set["jump_severity"],
                    "keeper_profit": parameter_set["keeper_profit"],
                    "share_vaults_protected": share_vaults_protected[
                        parameter_set["jump_severity"]
                    ],
                },
            },
            {
                "scenario_name": "downside_case",
                "params": {
                    # move up (jump frequency, jump severity) or down (keeper profit,
                    # share vaults protected) in the parameter space list
                    "jump_frequency": self.jump_frequency_list[
                        min(
                            self.jump_frequency_list.index(
                                parameter_set["jump_frequency"]
                            )
                            + 1,
                            4,
                        )
                    ],
                    "jump_severity": self.jump_severity_list[
                        min(
                            self.jump_severity_list.index(
                                parameter_set["jump_severity"]
                            )
                            + 2,
                            9,
                        )
                    ],
                    "keeper_profit": self.keeper_profit_list[
                        max(
                            self.keeper_profit_list.index(
                                parameter_set["keeper_profit"]
                            )
                            - 1,
                            0,
                        )
                    ],
                    "share_vaults_protected": share_vaults_protected[
                        self.jump_severity_list[
                            min(
                                self.jump_severity_list.index(
                                    parameter_set["jump_severity"]
                                )
                                + 1,
                                4,
                            )
                        ]
                    ],
                    "cr_distribution": {
                        0.15: 2.00,
                        0.25: 5.00,
                        0.5: 10.00,
                        0.75: 15.00,
                        1: 15.00,
                        1.25: 10.00,
                        1.50: 10.00,
                        1.75: 5.00,
                        2.00: 5.00,
                        2.25: 5.00,
                        2.50: 5.00,
                        2.75: 5.00,
                        3.00: 5.00,
                        3.25: 3.00,
                        3.5: 0.00,
                    },
                },
            },
            {
                "scenario_name": "downside_case",
                "params": {
                    # move down (jump frequency, jump severity) or up (keeper profit,
                    # share vaults protected) in the parameter space list
                    "jump_frequency": self.jump_frequency_list[
                        max(
                            self.jump_frequency_list.index(
                                parameter_set["jump_frequency"]
                            )
                            - 1,
                            0,
                        )
                    ],
                    "jump_severity": self.jump_severity_list[
                        max(
                            self.jump_severity_list.index(
                                parameter_set["jump_severity"]
                            )
                            - 2,
                            0,
                        )
                    ],
                    "keeper_profit": self.keeper_profit_list[
                        min(
                            self.keeper_profit_list.index(
                                parameter_set["keeper_profit"]
                            )
                            + 1,
                            4,
                        )
                    ],
                    "share_vaults_protected": share_vaults_protected[
                        self.jump_severity_list[
                            max(
                                self.jump_severity_list.index(
                                    parameter_set["jump_severity"]
                                )
                                - 2,
                                0,
                            )
                        ]
                    ],
                    "cr_distribution": {
                        0.15: 0.00,
                        0.25: 0.00,
                        0.5: 5.00,
                        0.75: 5.00,
                        1: 5.00,
                        1.25: 10.00,
                        1.50: 10.00,
                        1.75: 10.00,
                        2.00: 5.00,
                        2.25: 5.00,
                        2.50: 10.00,
                        2.75: 10.00,
                        3.00: 10.00,
                        3.25: 10.00,
                        3.5: 5.00,
                    },
                },
            },
        ]

        return scenario_params

    def compute_expected_loss_perc(
        self,
        sim_params,
        params,
        scenario_cr_dists,
        vault_type,
        df_slippage,
    ):
        total_asset_liquidated_debt = sum(
            [value["liquidated_debt"].sum() for value in scenario_cr_dists.values()]
        )

        # compute the rest of the vault metrics
        scenario_cr_dist = scenario_cr_dists[vault_type]

        # liquidated collateral
        scenario_cr_dist["liquidated_collateral"] = (
            scenario_cr_dist["liquidated_debt"]
            * scenario_cr_dist["cr_bucket"]
            * (1 + params["jump_severity"])
        ).round(2)

        # on-chain slippage
        df_asset_slippage = df_slippage.sort_values(by="usd_amount")
        if total_asset_liquidated_debt < df_asset_slippage["usd_amount"].max():
            onchain_slippage = round(
                df_asset_slippage[
                    df_asset_slippage["usd_amount"] > total_asset_liquidated_debt
                ]["slippage_percent"].iloc[0],
                4,
            )
        else:
            onchain_slippage = self.max_slippage

        # debt repaid
        scenario_cr_dist["debt_repaid"] = np.where(
            (
                scenario_cr_dist["liquidated_collateral"]
                * (1 - onchain_slippage - params["keeper_profit"])
            )
            <= scenario_cr_dist["liquidated_debt"],
            scenario_cr_dist["liquidated_collateral"]
            * (1 - onchain_slippage - params["keeper_profit"]),
            scenario_cr_dist["liquidated_debt"],
        ).round(2)

        # loss (bad debt)
        scenario_cr_dist["loss_bad_debt"] = (
            scenario_cr_dist["debt_repaid"] - scenario_cr_dist["liquidated_debt"]
        ).round(3)
        total_loss_bad_debt = scenario_cr_dist["loss_bad_debt"].sum()

        # expected loss (risk premium)
        expected_loss = total_loss_bad_debt * params["jump_frequency"]
        expected_loss_perc = round(
            (expected_loss / sim_params["simulate_de"]) * 100,
            2,
        )
        return expected_loss_perc

    def run_simulation_psweep(
        self,
        vault_type,
        simulation_params,
        scenario_params,
        debt_ranges,
        asset_vault_types_dict,
        df_slippage,
    ):
        # iterate over debt ceiling simulation values
        de_rp_dict = {}
        for debt_range in debt_ranges:
            scenario_list = []
            simulation_params[vault_type]["simulate_de"] = debt_range
            # iterate over scenario values
            for params_iter in scenario_params:
                # compute scenario cr distribution (if base case, use cr distribution)
                # for each vault type
                scenario_cr_dists = self.compute_scenario_cr_distribution_psweep(
                    simulation_params=simulation_params,
                    scenario_params=params_iter,
                    asset_vault_types_dict=asset_vault_types_dict,
                )

                expected_loss_perc = self.compute_expected_loss_perc(
                    sim_params=simulation_params[vault_type],
                    params=params_iter["params"],
                    scenario_cr_dists=scenario_cr_dists,
                    vault_type=vault_type,
                    df_slippage=df_slippage,
                )
                scenario_list.append(expected_loss_perc)
            de_rp_dict[debt_range] = abs(round(np.mean(scenario_list), 1))
        df = (
            pd.DataFrame.from_dict(de_rp_dict, orient="index")
            .reset_index()
            .rename(columns={"index": "simulate_de", 0: "risk_premium"})
        )
        return df

    def compute_for_vault_type(
        self,
        vault_type,
        df_slippage,
        debt_ranges,
        asset_vault_types_dict,
        share_vaults_protected,
    ):
        # Reset cache!
        self.scenario_cr_dists_cache = {}

        vault_start = datetime.now()
        log.info(f"Computing simulation results for: {vault_type}")

        # get the simulation parameters and vaults per vault type from the same asset
        # vault asset (eg WETH)
        simulation_params = self.get_simulation_spec(vault_type, asset_vault_types_dict)
        vault_type_total_debt_dai = int(
            asset_vault_types_dict[vault_type]["df_vaults"]["total_debt_dai"].sum()
        )

        results = []
        # iterate over parameter sets
        for index, pset in enumerate(self.psets):
            log.info(f"Computing for {vault_type}, pset #{index}")
            scenario_params = self.compute_scenario_params_psweep(
                pset, share_vaults_protected
            )
            # compute the simulation results
            simulation_results = self.run_simulation_psweep(
                vault_type,
                simulation_params,
                scenario_params,
                debt_ranges,
                asset_vault_types_dict,
                df_slippage,
            )
            df_results = pd.DataFrame(
                {
                    # time
                    "report_date": datetime.today().strftime("%d-%m-%Y"),
                    "meta_timestamp": datetime.now().timestamp(),
                    # simulation spec
                    "vault_type": vault_type,
                    "simulate_de": simulation_results["simulate_de"],
                    "risk_premium": simulation_results["risk_premium"],
                    "vault_type_total_debt_dai": vault_type_total_debt_dai,
                    "liquidation_ratio": simulation_params[vault_type][
                        "liquidation_ratio"
                    ],
                    # base scenario params
                    "jump_severity_base": scenario_params[0]["params"]["jump_severity"],
                    "jump_frequency_base": scenario_params[0]["params"][
                        "jump_frequency"
                    ],
                    "share_vaults_protected_base": share_vaults_protected[
                        scenario_params[0]["params"]["jump_severity"]
                    ],
                    "keeper_profit_base": scenario_params[0]["params"]["keeper_profit"],
                    # downside scenario params
                    "jump_severity_downside": scenario_params[1]["params"][
                        "jump_severity"
                    ],
                    "jump_frequency_downside": scenario_params[1]["params"][
                        "jump_frequency"
                    ],
                    "share_vaults_protected_downside": share_vaults_protected[
                        scenario_params[1]["params"]["jump_severity"]
                    ],
                    "keeper_profit_downside": scenario_params[1]["params"][
                        "keeper_profit"
                    ],
                    # upside scenario params
                    "jump_severity_upside": scenario_params[2]["params"][
                        "jump_severity"
                    ],
                    "jump_frequency_upside": scenario_params[2]["params"][
                        "jump_frequency"
                    ],
                    "share_vaults_protected_upside": share_vaults_protected[
                        scenario_params[2]["params"]["jump_severity"]
                    ],
                    "keeper_profit_upside": scenario_params[2]["params"][
                        "keeper_profit"
                    ],
                }
            )
            results.append(
                (
                    pset,
                    df_results,
                )
            )
        log.info(
            "Simulation results computed for: "
            f"{vault_type}. Took {datetime.now() - vault_start}"
        )
        return results
