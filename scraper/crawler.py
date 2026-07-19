import json
import logging
import re
from typing import Dict, List, Optional
from dataclasses import dataclass, field
import numpy as np
from bs4 import BeautifulSoup
from curl_cffi.requests import AsyncSession

logger = logging.getLogger(__name__)


@dataclass
class TeamTactics:
    name: str
    momentum: np.ndarray = field(repr=False)
    metrics: Dict[str, float]


class FotMobScraper:
    def __init__(self):
        self.session_kwargs = {"impersonate": "chrome"}

    async def get_team_id(self, team_name: str) -> Optional[str]:
        try:
            async with AsyncSession(**self.session_kwargs) as session:
                url = f"https://search.yahoo.com/search?p=site:fotmob.com/teams+overview+{team_name.replace(' ', '+')}"
                response = await session.get(url, timeout=15)
                match = re.search(
                    r"RU=https%3a%2f%2fwww\.fotmob\.com%2fteams%2f(\d+)",
                    response.text,
                )
                if match:
                    team_id = match.group(1)
                    logger.info(f"Resolved '{team_name}' to team ID {team_id}")
                    return team_id

                match = re.search(
                    r"fotmob\.com(?:%2F|/)teams(?:%2F|/)(\d+)", response.text
                )
                if match:
                    team_id = match.group(1)
                    logger.info(
                        f"Resolved '{team_name}' to team ID {team_id} (fallback)"
                    )
                    return team_id

                logger.warning(f"Could not resolve team ID for '{team_name}'")
                return None
        except Exception as e:
            logger.error(f"Error resolving team ID for {team_name}: {e}")
            return None

    async def get_last_5_matches(self, team_id: str, tournament_id: str = "77") -> List[str]:
        """
        Fetches the last 5 match IDs for a team.
        Attempts to fetch from the team overview page first (which works for clubs).
        If that fails (which it does for national teams missing the data), it falls back
        to fetching fixtures from a tournament endpoint (defaulting to World Cup ID 77).
        """
        match_ids = []
        try:
            async with AsyncSession(**self.session_kwargs) as session:
                # Primary logic for Clubs
                url = f"https://www.fotmob.com/teams/{team_id}/overview/"
                response = await session.get(url, timeout=15)
                soup = BeautifulSoup(response.text, "html.parser")
                next_data_script = soup.find("script", id="__NEXT_DATA__")

                if next_data_script:
                    data = json.loads(next_data_script.string)
                    fallback = (
                        data.get("props", {})
                        .get("pageProps", {})
                        .get("fallback", {})
                    )
                    team_data = fallback.get(f"team-{team_id}", {})

                    if team_data:
                        all_fixtures = (
                            team_data.get("fixtures", {})
                            .get("allFixtures", {})
                            .get("fixtures", [])
                        )

                        for fixture in all_fixtures:
                            if fixture.get("notStarted") is False and fixture.get("result"):
                                match_ids.append(str(fixture["id"]))

                        if match_ids:
                            return match_ids[-5:] if len(match_ids) >= 5 else match_ids

                # Fallback logic for National Teams (Tournament-Centric)
                logger.info(f"Team data not found in fallback for team {team_id}. Using tournament fallback ({tournament_id}).")

                # We will check both ID 77 (World Cup) and 73 (Copa America) and 50 (Euros) just to be safe if 77 has no recent matches for them.
                # Actually, the user says we can default to 77. But checking a few common ones for national teams is more robust.
                tournaments_to_check = [tournament_id, "77", "73", "50", "111", "130"]

                # Deduplicate list while preserving order
                tournaments_to_check = list(dict.fromkeys(tournaments_to_check))

                all_found_matches = []
                for t_id in tournaments_to_check:
                    tournament_url = f"https://www.fotmob.com/api/data/leagues?id={t_id}"
                    tour_response = await session.get(tournament_url, timeout=15)
                    if tour_response.status_code == 200:
                        tour_data = tour_response.json()

                        found_matches = []
                        def extract_matches(obj):
                            if isinstance(obj, dict):
                                if 'home' in obj and 'away' in obj and 'id' in obj and 'notStarted' in obj:
                                    found_matches.append(obj)
                                for k, v in obj.items():
                                    extract_matches(v)
                            elif isinstance(obj, list):
                                for v in obj:
                                    extract_matches(v)

                        extract_matches(tour_data)

                        team_matches = []
                        for m in found_matches:
                            home_id = str(m.get('home', {}).get('id', ''))
                            away_id = str(m.get('away', {}).get('id', ''))
                            if home_id == str(team_id) or away_id == str(team_id):
                                # Ensure it's finished
                                if m.get('notStarted') is False or m.get('status', {}).get('finished') is True:
                                    team_matches.append(m)

                        all_found_matches.extend(team_matches)

                        if len(all_found_matches) >= 5:
                            break # We have enough matches

                # Deduplicate by ID and sort if possible (assuming chronological order in extraction)
                unique_matches = {}
                for m in all_found_matches:
                    unique_matches[str(m['id'])] = m

                match_ids = list(unique_matches.keys())

                # Note: if there are more than 5, we return the last 5
                # The ordering from fotmob is usually chronological per tournament
                return match_ids[-5:] if len(match_ids) >= 5 else match_ids

        except Exception as e:
            logger.error(f"Error fetching matches for team {team_id}: {e}")
            return []

        return []

    async def get_match_data(self, match_id: str) -> Optional[dict]:
        try:
            async with AsyncSession(**self.session_kwargs) as session:
                url = f"https://www.fotmob.com/match/{match_id}/"
                response = await session.get(url, timeout=15)
                soup = BeautifulSoup(response.text, "html.parser")
                next_data_script = soup.find("script", id="__NEXT_DATA__")

                if not next_data_script:
                    logger.error(
                        f"Could not find __NEXT_DATA__ for match {match_id}"
                    )
                    return None

                data = json.loads(next_data_script.string)
                content = (
                    data.get("props", {})
                    .get("pageProps", {})
                    .get("content", {})
                )
                general = (
                    data.get("props", {})
                    .get("pageProps", {})
                    .get("general", {})
                )

                return {"content": content, "general": general}
        except Exception as e:
            logger.error(f"Error fetching match data for {match_id}: {e}")
            return None


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
    def parse_team_match(
        cls, match_data: dict, team_name: str
    ) -> Optional[TeamTactics]:
        general_info = match_data.get("general", {})
        home_team_name = cls._get_nested(
            general_info, ["homeTeam", "name"], ""
        ).lower()

        is_home = team_name.lower() in home_team_name
        val_idx = 0 if is_home else 1

        content = match_data.get("content", {})
        if not content:
            logger.warning("Key 'content' is missing or null.")
            return None

        momentum_array = np.zeros(150)
        momentum_data = cls._get_nested(
            content, ["momentum", "main", "data"], []
        )

        if not momentum_data:
            momentum_data = cls._get_nested(
                content, ["matchFacts", "momentum", "main", "data"], []
            )

        for point in momentum_data:
            try:
                raw_min = str(point.get("minute", 0))
                minute = (
                    sum(int(float(x)) for x in raw_min.split("+"))
                    if "+" in raw_min
                    else int(float(raw_min))
                )
                val = float(point.get("value", 0))
                if 0 <= minute < 150:
                    momentum_array[minute] = (
                        val
                        if val > 0
                        else 0 if is_home else abs(val) if val < 0 else 0
                    )
            except (ValueError, TypeError):
                continue

        metrics = {
            "Expected Goals (xG)": 0.0,
            "Touches in Opp Box": 0.0,
            "Total Passes": 0.0,
            "High Press / Recoveries": 0.0,
        }

        stats_groups = cls._get_nested(
            content, ["stats", "Periods", "All", "stats"], []
        )
        for group in stats_groups:
            for stat in group.get("stats", []):
                title = stat.get("title", "").lower()
                vals = stat.get("stats", [])

                if len(vals) > val_idx:
                    raw_str = str(vals[val_idx])
                    match = re.search(r"\d+(\.\d+)?", raw_str)
                    if not match:
                        continue
                    clean_val = float(match.group(0))

                    if title in ["expected goals (xg)", "xg"]:
                        metrics["Expected Goals (xG)"] = clean_val
                    elif (
                        "touches in opposition box" in title
                        or "touches in attacking" in title
                    ):
                        metrics["Touches in Opp Box"] = clean_val
                    elif title in [
                        "passes",
                        "total passes",
                        "passes completed",
                    ]:
                        metrics["Total Passes"] = max(
                            metrics["Total Passes"], clean_val
                        )
                    elif any(
                        k in title
                        for k in [
                            "possession won in final 3rd",
                            "balls recovered",
                            "interceptions",
                        ]
                    ):
                        metrics["High Press / Recoveries"] = max(
                            metrics["High Press / Recoveries"], clean_val
                        )

        return TeamTactics(
            name=team_name, momentum=momentum_array, metrics=metrics
        )

    @classmethod
    async def get_aggregated_team_tactics(
        cls, team_name: str
    ) -> Optional[TeamTactics]:
        scraper = FotMobScraper()
        team_id = await scraper.get_team_id(team_name)
        if not team_id:
            logger.error(f"Failed to find team ID for {team_name}")
            return None

        match_ids = await scraper.get_last_5_matches(team_id)
        if not match_ids:
            logger.error(f"Failed to find recent matches for {team_name}")
            return None

        tactics_list = []
        for match_id in match_ids:
            data = await scraper.get_match_data(match_id)
            if data:
                tactics = cls.parse_team_match(data, team_name)
                if tactics:
                    tactics_list.append(tactics)

        if not tactics_list:
            logger.error(f"Failed to parse any matches for {team_name}")
            return None

        avg_momentum = np.mean([t.momentum for t in tactics_list], axis=0)

        avg_metrics = {
            "Expected Goals (xG)": np.mean(
                [t.metrics["Expected Goals (xG)"] for t in tactics_list]
            ),
            "Touches in Opp Box": np.mean(
                [t.metrics["Touches in Opp Box"] for t in tactics_list]
            ),
            "Total Passes": np.mean(
                [t.metrics["Total Passes"] for t in tactics_list]
            ),
            "High Press / Recoveries": np.mean(
                [t.metrics["High Press / Recoveries"] for t in tactics_list]
            ),
        }

        logger.info(
            f"Successfully aggregated data from {len(tactics_list)} matches for {team_name}"
        )
        return TeamTactics(
            name=team_name, momentum=avg_momentum, metrics=avg_metrics
        )
