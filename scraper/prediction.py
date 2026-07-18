from typing import Tuple
from .crawler import TeamTactics
import numpy as np


class PredictionModel:
    @staticmethod
    def generate_prediction(
        t1: TeamTactics, t2: TeamTactics
    ) -> Tuple[TeamTactics, TeamTactics]:
        t1_passes = max(t1.metrics["Total Passes"], 1)
        t2_passes = max(t2.metrics["Total Passes"], 1)
        total_passes = t1_passes + t2_passes

        t1_share = t1_passes / total_passes
        t2_share = t2_passes / total_passes

        match_avg_passes = 900
        pred_passes_t1 = match_avg_passes * t1_share
        pred_passes_t2 = match_avg_passes * t2_share

        t1_dampener = 0.8 + (t1_share * 0.4)
        t2_dampener = 0.8 + (t2_share * 0.4)

        pred_metrics_t1 = {
            "Expected Goals (xG)": t1.metrics["Expected Goals (xG)"]
            * t1_dampener,
            "Touches in Opp Box": t1.metrics["Touches in Opp Box"]
            * t1_dampener,
            "Total Passes": pred_passes_t1,
            "High Press / Recoveries": t1.metrics["High Press / Recoveries"]
            * 1.05,
        }

        pred_metrics_t2 = {
            "Expected Goals (xG)": t2.metrics["Expected Goals (xG)"]
            * t2_dampener,
            "Touches in Opp Box": t2.metrics["Touches in Opp Box"]
            * t2_dampener,
            "Total Passes": pred_passes_t2,
            "High Press / Recoveries": t2.metrics["High Press / Recoveries"]
            * 1.05,
        }

        pred_mom_t1 = t1.momentum * t1_dampener
        pred_mom_t2 = t2.momentum * t2_dampener

        return (
            TeamTactics(t1.name, pred_mom_t1, pred_metrics_t1),
            TeamTactics(t2.name, pred_mom_t2, pred_metrics_t2),
        )

    @staticmethod
    def bivariate_poisson_mc(
        t1: TeamTactics, t2: TeamTactics, num_simulations=10000
    ):
        lambda_1 = t1.metrics["Expected Goals (xG)"]
        lambda_2 = t2.metrics["Expected Goals (xG)"]

        covar = 0.1 * min(lambda_1, lambda_2)

        lam_1_indep = max(lambda_1 - covar, 0.01)
        lam_2_indep = max(lambda_2 - covar, 0.01)

        x1 = np.random.poisson(lam_1_indep, num_simulations)
        x2 = np.random.poisson(lam_2_indep, num_simulations)
        common = np.random.poisson(covar, num_simulations)

        goals_t1 = x1 + common
        goals_t2 = x2 + common

        t1_wins = np.sum(goals_t1 > goals_t2) / num_simulations
        t2_wins = np.sum(goals_t2 > goals_t1) / num_simulations
        draws = np.sum(goals_t1 == goals_t2) / num_simulations

        return {
            t1.name: t1_wins * 100,
            t2.name: t2_wins * 100,
            "Draw": draws * 100,
        }
