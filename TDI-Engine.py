# -*- coding: utf-8 -*-
"""
Created on Tue Jul 14 19:31:48 2026

@author: The Normal One
"""

import argparse
import json
import logging
import re
import sys
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cloudscraper
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from bs4 import BeautifulSoup

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
logger = logging.getLogger(__name__)

@dataclass
class TeamTactics:
    name: str
    momentum: np.ndarray = field(repr=False)
    metrics: Dict[str, float]

class FotMobExtractor:
    def __init__(self):
        logger.info("Initializing cloudscraper client.")
        self.scraper = cloudscraper.create_scraper(
            browser={'browser': 'chrome', 'platform': 'windows', 'desktop': True}
        )

    def fetch_match_data(self, url: str, output_filename: str) -> bool:
        logger.info(f"Initiating request to: {url}")
        try:
            response = self.scraper.get(url, timeout=15)
            if response.status_code != 200:
                logger.error(f"HTTP Request failed with status code {response.status_code}.")
                return False
                
            soup = BeautifulSoup(response.text, 'html.parser')
            next_data = soup.find('script', id='__NEXT_DATA__')
            
            if next_data:
                payload = json.loads(next_data.string)
                with open(output_filename, 'w', encoding='utf-8') as f:
                    json.dump(payload, f, indent=4, ensure_ascii=False)
                logger.info(f"Payload successfully persisted to {output_filename}")
                return True
            
            logger.error("Tag __NEXT_DATA__ not found in the DOM.")
            return False
            
        except Exception as e:
            logger.error(f"Network or parsing exception: {str(e)}")
            return False

class TacticalParser:
    @staticmethod
    def _get_nested(d: dict, keys: list, default=None):
        for k in keys:
            if isinstance(d, dict) and k in d:
                d = d[k]
            else:
                return default
        return d

    @classmethod
    def parse_team(cls, filepath: str, team_name: str) -> Optional[TeamTactics]:
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except FileNotFoundError:
            logger.error(f"Target file not found: {filepath}")
            return None
        except json.JSONDecodeError:
            logger.error(f"Invalid JSON format in file: {filepath}")
            return None

        general_info = cls._get_nested(data, ['props', 'pageProps', 'general'], {})
        home_team_name = cls._get_nested(general_info, ['homeTeam', 'name'], '').lower()
        
        is_home = team_name.lower() in home_team_name
        val_idx = 0 if is_home else 1
        logger.info(f"Team '{team_name}' registered as {'Home' if is_home else 'Away'} status.")
            
        content = cls._get_nested(data, ['props', 'pageProps', 'content'])
        if not content:
            logger.warning(f"Key 'content' is missing or null in {filepath}. Structure may have changed.")
            return None

        momentum_array = np.zeros(150)
        momentum_data = cls._get_nested(content, ['matchFacts', 'momentum', 'main', 'data'], [])
        
        if not momentum_data:
            logger.warning(f"Momentum data array is empty for {team_name}.")

        for point in momentum_data:
            try:
                raw_min = str(point.get('minute', 0))
                minute = sum(int(float(x)) for x in raw_min.split('+')) if '+' in raw_min else int(float(raw_min))
                val = float(point.get('value', 0))
                if 0 <= minute < 150:
                    momentum_array[minute] = val if val > 0 else 0 if is_home else abs(val) if val < 0 else 0
            except (ValueError, TypeError):
                continue

        metrics = {
            'Expected Goals (xG)': 0.0,
            'Touches in Opp Box': 0.0,
            'Total Passes': 0.0,
            'High Press / Recoveries': 0.0
        }
        
        stats_groups = cls._get_nested(content, ['stats', 'Periods', 'All', 'stats'], [])
        for group in stats_groups:
            for stat in group.get('stats', []):
                title = stat.get('title', '').lower()
                vals = stat.get('stats', [])
                
                if len(vals) > val_idx:
                    raw_str = str(vals[val_idx])
                    match = re.search(r'\d+(\.\d+)?', raw_str)
                    if not match:
                        continue
                    clean_val = float(match.group(0))
                    
                    if title in ['expected goals (xg)', 'xg']:
                        metrics['Expected Goals (xG)'] = clean_val
                    elif 'touches in opposition box' in title or 'touches in attacking' in title:
                        metrics['Touches in Opp Box'] = clean_val
                    elif title in ['passes', 'total passes', 'passes completed']:
                        metrics['Total Passes'] = max(metrics['Total Passes'], clean_val)
                    elif any(k in title for k in ['possession won in final 3rd', 'balls recovered', 'interceptions']):
                        metrics['High Press / Recoveries'] = max(metrics['High Press / Recoveries'], clean_val)

        return TeamTactics(name=team_name, momentum=momentum_array, metrics=metrics)

class PredictionModel:
    @staticmethod
    def generate_prediction(t1: TeamTactics, t2: TeamTactics) -> Tuple[TeamTactics, TeamTactics]:
        t1_passes = max(t1.metrics['Total Passes'], 1)
        t2_passes = max(t2.metrics['Total Passes'], 1)
        total_passes = t1_passes + t2_passes
        
        t1_share = t1_passes / total_passes
        t2_share = t2_passes / total_passes
        
        match_avg_passes = 900
        pred_passes_t1 = match_avg_passes * t1_share
        pred_passes_t2 = match_avg_passes * t2_share

        t1_dampener = 0.8 + (t1_share * 0.4)
        t2_dampener = 0.8 + (t2_share * 0.4)

        pred_metrics_t1 = {
            'Expected Goals (xG)': t1.metrics['Expected Goals (xG)'] * t1_dampener,
            'Touches in Opp Box': t1.metrics['Touches in Opp Box'] * t1_dampener,
            'Total Passes': pred_passes_t1,
            'High Press / Recoveries': t1.metrics['High Press / Recoveries'] * 1.05 
        }
        
        pred_metrics_t2 = {
            'Expected Goals (xG)': t2.metrics['Expected Goals (xG)'] * t2_dampener,
            'Touches in Opp Box': t2.metrics['Touches in Opp Box'] * t2_dampener,
            'Total Passes': pred_passes_t2,
            'High Press / Recoveries': t2.metrics['High Press / Recoveries'] * 1.05
        }
        
        pred_mom_t1 = t1.momentum * t1_dampener
        pred_mom_t2 = t2.momentum * t2_dampener
        
        return (
            TeamTactics(t1.name, pred_mom_t1, pred_metrics_t1),
            TeamTactics(t2.name, pred_mom_t2, pred_metrics_t2)
        )

class Dashboard:
    @staticmethod
    def render(t1: TeamTactics, t2: TeamTactics, output_name: str = "prediction_dashboard.png"):
        plt.style.use('dark_background')
        fig = plt.figure(figsize=(15, 10), dpi=300)
        gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.2], hspace=0.3)
        
        max_min = max(100, max(np.max(np.nonzero(t1.momentum)), np.max(np.nonzero(t2.momentum))) + 5)
        minutes = np.arange(max_min)
        
        net_momentum = t1.momentum[:max_min] - t2.momentum[:max_min]
        smoothed_net = gaussian_filter1d(net_momentum, sigma=1.5)
        
        t1_color, t2_color = '#00e5ff', '#ff1744' 
        
        ax_mom = fig.add_subplot(gs[0])
        bar_colors = np.where(smoothed_net > 0, t1_color, t2_color)
        ax_mom.bar(minutes, smoothed_net, color=bar_colors, width=1.0, alpha=0.85)
        ax_mom.axhline(0, color='#ffffff', linewidth=1.5, alpha=0.5)
        
        ax_mom.set_title("Predicted Match Momentum", fontsize=16, fontweight='bold', pad=15)
        ax_mom.set_xlim(0, max_min)
        ax_mom.set_xticks([0, 45, 90, max_min if max_min > 100 else 90])
        ax_mom.set_xticklabels(['0', 'HT', '90', 'FT' if max_min > 100 else ''], fontsize=12)
        ax_mom.set_yticks([])
        
        for spine in ['top', 'right', 'left']:
            ax_mom.spines[spine].set_visible(False)
        ax_mom.spines['bottom'].set_color('#333333')
        
        ax_mom.text(0, np.max(smoothed_net)*1.1, t1.name, color=t1_color, fontsize=14, fontweight='bold')
        ax_mom.text(0, np.min(smoothed_net)*1.1, t2.name, color=t2_color, fontsize=14, fontweight='bold')

        ax_stats = fig.add_subplot(gs[1])
        keys = list(t1.metrics.keys())
        y_pos = np.arange(len(keys))
        
        v1 = np.array([t1.metrics[k] for k in keys])
        v2 = np.array([t2.metrics[k] for k in keys])
        
        max_vals = np.maximum(v1 + v2, 1)
        norm1 = (v1 / max_vals) * 100
        norm2 = (v2 / max_vals) * 100
        
        ax_stats.barh(y_pos, -norm1, color=t1_color, height=0.35, alpha=0.9)
        ax_stats.barh(y_pos, norm2, color=t2_color, height=0.35, alpha=0.9)
        
        for i, key in enumerate(keys):
            ax_stats.text(0, i + 0.4, key, ha='center', va='center', color='white', fontsize=12, fontweight='bold')
            fmt = "{:.0f}" if 'Passes' in key or 'Recoveries' in key else "{:.2f}"
            
            ax_stats.text(-norm1[i] - 3, i, fmt.format(v1[i]), ha='right', va='center', color=t1_color, fontsize=13, fontweight='bold')
            ax_stats.text(norm2[i] + 3, i, fmt.format(v2[i]), ha='left', va='center', color=t2_color, fontsize=13, fontweight='bold')

        ax_stats.set_xlim(-120, 120)
        ax_stats.axis('off')
        ax_stats.text(-50, -0.8, t1.name, ha='center', color=t1_color, fontsize=16, fontweight='bold')
        ax_stats.text(50, -0.8, t2.name, ha='center', color=t2_color, fontsize=16, fontweight='bold')
        ax_stats.invert_yaxis()

        plt.savefig(output_name, format="png", bbox_inches='tight', transparent=False)
        logger.info(f"Render completed. Output generated at: {output_name}")
        plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="TDI Engine - Match Prediction Pipeline")
    parser.add_argument("--match1", type=str, required=True, help="URL for Team 1 historical match data")
    parser.add_argument("--match2", type=str, required=True, help="URL for Team 2 historical match data")
    parser.add_argument("--team1", type=str, required=True, help="Name of Team 1")
    parser.add_argument("--team2", type=str, required=True, help="Name of Team 2")
    args = parser.parse_args()

    f1_json = 'match_1_payload.json'
    f2_json = 'match_2_payload.json'
    
    extractor = FotMobExtractor()
    logger.info("Starting data extraction phase.")
    
    ext1 = extractor.fetch_match_data(args.match1, f1_json)
    ext2 = extractor.fetch_match_data(args.match2, f2_json)
    
    if not (ext1 and ext2):
        logger.error("Extraction phase failed. Halting execution.")
        sys.exit(1)
        
    logger.info("Starting parsing phase.")
    team1_base = TacticalParser.parse_team(f1_json, args.team1) 
    team2_base = TacticalParser.parse_team(f2_json, args.team2)   
    
    if team1_base and team2_base:
        logger.info("Generating predictive models.")
        team1_pred, team2_pred = PredictionModel.generate_prediction(team1_base, team2_base)
        
        Dashboard.render(team1_pred, team2_pred)
        logger.info("Pipeline execution completed successfully.")
    else:
        logger.error("Data parsing returned null objects. Execution halted.")
        sys.exit(1)

if __name__ == "__main__":
    main()