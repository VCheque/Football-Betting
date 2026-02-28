#!/usr/bin/env python3
"""Canonical team names used across the project."""

from __future__ import annotations

import pandas as pd

# Maps source names to commonly used official club names.
TEAM_NAME_MAP: dict[str, str] = {
    # Premier League
    "Man City": "Manchester City",
    "Man United": "Manchester United",
    "Nott'm Forest": "Nottingham Forest",
    "Wolves": "Wolverhampton Wanderers",
    "Tottenham": "Tottenham Hotspur",
    "Leeds": "Leeds United",
    "Leicester": "Leicester City",
    "Newcastle": "Newcastle United",
    "West Ham": "West Ham United",
    "Ipswich": "Ipswich Town",
    # La Liga
    "Alaves": "Deportivo Alaves",
    "Ath Bilbao": "Athletic Club",
    "Ath Madrid": "Atletico de Madrid",
    "Betis": "Real Betis",
    "Celta": "Celta de Vigo",
    "Espanol": "RCD Espanyol",
    "Leganes": "CD Leganes",
    "Sociedad": "Real Sociedad",
    "Vallecano": "Rayo Vallecano",
    "Valladolid": "Real Valladolid",
    "Oviedo": "Real Oviedo",
    # Bundesliga
    "Dortmund": "Borussia Dortmund",
    "Ein Frankfurt": "Eintracht Frankfurt",
    "FC Koln": "1. FC Koln",
    "Greuther Furth": "SpVgg Greuther Furth",
    "M'gladbach": "Borussia Monchengladbach",
    "Hertha": "Hertha BSC",
    "Leverkusen": "Bayer Leverkusen",
    "Mainz": "Mainz 05",
    "St Pauli": "FC St. Pauli",
    "RB Leipzig": "RasenBallsport Leipzig",
    # Ligue 1
    "Paris SG": "Paris Saint-Germain",
    "St Etienne": "AS Saint-Etienne",
    "Lyon": "Olympique Lyonnais",
    "Marseille": "Olympique de Marseille",
    "Lille": "Lille OSC",
    "Nice": "OGC Nice",
    "Reims": "Stade de Reims",
    "Rennes": "Stade Rennais",
    "Toulouse": "Toulouse FC",
    "Nantes": "FC Nantes",
    "Lens": "RC Lens",
    "Angers": "Angers SCO",
    "Auxerre": "AJ Auxerre",
    "Brest": "Stade Brestois",
    "Le Havre": "Le Havre AC",
    # Serie A
    "Inter": "Inter Milan",
    "Milan": "AC Milan",
    "Roma": "AS Roma",
    "Lazio": "SS Lazio",
    "Verona": "Hellas Verona",
    "Como": "Como 1907",
    "Pisa": "Pisa SC",
    # Eredivisie
    "For Sittard": "Fortuna Sittard",
    "Nijmegen": "NEC Nijmegen",
    "Waalwijk": "RKC Waalwijk",
    "Zwolle": "PEC Zwolle",
    "Twente": "FC Twente",
    "Utrecht": "FC Utrecht",
    # Primeira Liga
    "Sp Lisbon": "Sporting CP",
    "Sp. Lisbon": "Sporting CP",
    "Sp Braga": "SC Braga",
    "Guimaraes": "Vitoria de Guimaraes",
    "Pacos Ferreira": "Pacos de Ferreira",
    "Estoril": "Estoril Praia",
    "Estrela": "Estrela da Amadora",
    "Maritimo": "CS Maritimo",
    "Porto": "FC Porto",
    "Benfica": "SL Benfica",
    "Rio Ave": "Rio Ave FC",
    "Gil Vicente": "Gil Vicente FC",
    "Santa Clara": "CD Santa Clara",
    "Farense": "SC Farense",
    "Nacional": "CD Nacional",
    "Arouca": "FC Arouca",
    "Famalicao": "FC Famalicao",
    "Moreirense": "Moreirense FC",
    "Portimonense": "Portimonense SC",
    "Casa Pia": "Casa Pia AC",
    "Tondela": "CD Tondela",
    "Vizela": "FC Vizela",
    "Chaves": "GD Chaves",
    "AVS": "AVS Futebol SAD",
    "Alverca": "FC Alverca",
}


def canonical_team_name(name: object) -> object:
    if pd.isna(name):
        return name
    text = str(name).strip()
    return TEAM_NAME_MAP.get(text, text)

