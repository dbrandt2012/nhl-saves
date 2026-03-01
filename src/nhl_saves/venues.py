"""NHL arena coordinates and travel distance utilities.

All 32 current NHL teams (including UTA for the Utah Hockey Club, 2024-25 onward).
"""

import math

# (latitude, longitude) for each team's home arena
ARENA_COORDS: dict[str, tuple[float, float]] = {
    "ANA": (33.8073, -117.8768),  # Honda Center, Anaheim CA
    "BOS": (42.3662, -71.0621),  # TD Garden, Boston MA
    "BUF": (42.8750, -78.8764),  # KeyBank Center, Buffalo NY
    "CGY": (51.0374, -114.0519),  # Scotiabank Saddledome, Calgary AB
    "CAR": (35.8033, -78.7217),  # PNC Arena, Raleigh NC
    "CHI": (41.8807, -87.6742),  # United Center, Chicago IL
    "COL": (39.7484, -105.0076),  # Ball Arena, Denver CO
    "CBJ": (39.9694, -83.0064),  # Nationwide Arena, Columbus OH
    "DAL": (32.7905, -96.8102),  # American Airlines Center, Dallas TX
    "DET": (42.3411, -83.0550),  # Little Caesars Arena, Detroit MI
    "EDM": (53.5461, -113.4938),  # Rogers Place, Edmonton AB
    "FLA": (26.1584, -80.3256),  # Amerant Bank Arena, Sunrise FL
    "LAK": (34.0430, -118.2673),  # Crypto.com Arena, Los Angeles CA
    "MIN": (44.9448, -93.1010),  # Xcel Energy Center, St. Paul MN
    "MTL": (45.4960, -73.5694),  # Centre Bell, Montreal QC
    "NSH": (36.1591, -86.7785),  # Bridgestone Arena, Nashville TN
    "NJD": (40.7334, -74.1710),  # Prudential Center, Newark NJ
    "NYI": (40.7226, -73.5906),  # UBS Arena, Elmont NY
    "NYR": (40.7505, -73.9934),  # Madison Square Garden, New York NY
    "OTT": (45.2969, -75.9271),  # Canadian Tire Centre, Kanata ON
    "PHI": (39.9012, -75.1720),  # Wells Fargo Center, Philadelphia PA
    "PIT": (40.4396, -79.9893),  # PPG Paints Arena, Pittsburgh PA
    "SEA": (47.6218, -122.3543),  # Climate Pledge Arena, Seattle WA
    "SJS": (37.3328, -121.9007),  # SAP Center, San Jose CA
    "STL": (38.6266, -90.2029),  # Enterprise Center, St. Louis MO
    "TBL": (27.9428, -82.4519),  # Amalie Arena, Tampa FL
    "TOR": (43.6435, -79.3791),  # Scotiabank Arena, Toronto ON
    "VAN": (49.2778, -123.1088),  # Rogers Arena, Vancouver BC
    "VGK": (36.1027, -115.1780),  # T-Mobile Arena, Las Vegas NV
    "WSH": (38.8981, -77.0209),  # Capital One Arena, Washington DC
    "WPG": (49.8925, -97.1437),  # Canada Life Centre, Winnipeg MB
    "UTA": (40.7683, -111.9010),  # Delta Center, Salt Lake City UT
}


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in km between two lat/lon points."""
    R = 6371.0  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = (
        math.sin(dphi / 2) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    )
    return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))


def trip_km(from_team: str, to_team: str) -> float:
    """Distance in km between two NHL arenas.

    Returns 0.0 if teams are identical, NaN if either team is unknown.
    """
    if from_team == to_team:
        return 0.0
    c1 = ARENA_COORDS.get(from_team)
    c2 = ARENA_COORDS.get(to_team)
    if c1 is None or c2 is None:
        return float("nan")
    return haversine_km(c1[0], c1[1], c2[0], c2[1])
