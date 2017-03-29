import csv
from collections import defaultdict

class Teams(object):
    """ The teams in MLB.
    Teams have the following properties:

    Attributes:
        wins = dictionary of dictionaries, wins[season][team] = team_wins
        abbr = dictionary of (3-letter abbrev, full team name) pairs
        road_att = dictionary of dictionaries, road_att[season][team] = avg_road_attendance

    Methods:
        get_wins(season, team): Total wins for team in season.
        get_name(3-letter abbreviation): Full name of team
        get_road_att(season, team): Avg. road attendance for team in season.
        is_al_west(season, team): Returns boolean for whether or not team was in AL West that season.
    """

    def __init__(self):
        self.wins = wins()
        self.abbr = team_dict()
        self.road_att = avg_road_attendance()

    def get_wins(self, season, team):
        team_wins = self.wins[season][team]

        return team_wins

    def get_name(self, abbrev):
        full_name = self.abbr[abbrev]

        return full_name

    def get_road_att(self, season, team):
        avg_att = self.road_att[season][team]

        return avg_att

    def is_al_west(self, season, team):
        div_old = ["OAK", "Oakland Athletics", "ANA", "Los Angeles Angels", "SEA", "Seattle Mariners",
                   "TEX", "Texas Rangers"]

        div_new = ["OAK", "Oakland Athletics", "ANA", "Los Angeles Angels", "SEA", "Seattle Mariners",
                   "TEX", "Texas Rangers", "HOU", "Houston Astros"]

        div = div_old if int(season) < 2013 else div_new

        return team in div


def team_dict():
    """
    This dictionary can be used to help make outside data sources more compatible.

    :return: a dictionary of team abbreviations to full team names
             and the ESPN city qualifier to full team name.
    """
    teams = dict()

    teams["BLA"] = "Baltimore Orioles"
    teams["BOS"] = "Boston Red Sox"
    teams["NYY"] = "New York Yankees"
    teams["ARI"] = "Arizona Diamondbacks"
    teams["ATL"] = "Atlanta Braves"
    teams["BAL"] = "Baltimore Orioles"
    teams["CHC"] = "Chicago Cubs"
    teams["CHW"] = "Chicago White Sox"
    teams["CIN"] = "Cincinnati Reds"
    teams["CLE"] = "Cleveland Indians"
    teams["COL"] = "Colorado Rockies"
    teams["DET"] = "Detroit Tigers"
    teams["FLA"] = "Miami Marlins"
    teams["HOU"] = "Houston Astros"
    teams["KCR"] = "Kansas City Royals"
    teams["ANA"] = "Los Angeles Angels"
    teams["LAD"] = "Los Angeles Dodgers"
    teams["MIA"] = "Miami Marlins"
    teams["MIL"] = "Milwaukee Brewers"
    teams["MIN"] = "Minnesota Twins"
    teams["NYM"] = "New York Mets"
    teams["OAK"] = "Oakland Athletics"
    teams["PHI"] = "Philadelphia Phillies"
    teams["PIT"] = "Pittsburgh Pirates"
    teams["SDP"] = "San Diego Padres"
    teams["SFG"] = "San Francisco Giants"
    teams["SEA"] = "Seattle Mariners"
    teams["STL"] = "St. Louis Cardinals"
    teams["TBD"] = "Tampa Bay Rays"
    teams["TEX"] = "Texas Rangers"
    teams["TOR"] = "Toronto Blue Jays"
    teams["WSN"] = "Washington Nationals"

    # ESPN Team Names from road attendance
    teams["Baltimore"] = "Baltimore Orioles"
    teams["Boston"] = "Boston Red Sox"
    teams["NY Yankees"] = "New York Yankees"
    teams["Arizona"] = "Arizona Diamondbacks"
    teams["Atlanta"] = "Atlanta Braves"
    teams["Chicago Cubs"] = "Chicago Cubs"
    teams["Chicago White Sox"] = "Chicago White Sox"
    teams["Cincinnati"] = "Cincinnati Reds"
    teams["Cleveland"] = "Cleveland Indians"
    teams["Colorado"] = "Colorado Rockies"
    teams["Detroit"] = "Detroit Tigers"
    teams["Florida"] = "Miami Marlins"  # Not worth separating
    teams["Miami"] = "Miami Marlins"
    teams["Houston"] = "Houston Astros"
    teams["Kansas City"] = "Kansas City Royals"
    teams["LA Angels"] = "Los Angeles Angels"
    teams["LA Dodgers"] = "Los Angeles Dodgers"
    teams["Milwaukee"] = "Milwaukee Brewers"
    teams["Minnesota"] = "Minnesota Twins"
    teams["NY Mets"] = "New York Mets"
    teams["Oakland"] = "Oakland Athletics"
    teams["Philadelphia"] = "Philadelphia Phillies"
    teams["Pittsburgh"] = "Pittsburgh Pirates"
    teams["San Diego"] = "San Diego Padres"
    teams["San Francisco"] = "San Francisco Giants"
    teams["Seattle"] = "Seattle Mariners"
    teams["St. Louis"] = "St. Louis Cardinals"
    teams["Tampa Bay"] = "Tampa Bay Rays"
    teams["Texas"] = "Texas Rangers"
    teams["Toronto"] = "Toronto Blue Jays"
    teams["Washington"] = "Washington Nationals"

    return teams


def wins():
    """

    :return: Dictionary that will be an attribute of a Teams object.
    """

    # Team wins file saved from baseball reference url:
    # http://www.baseball-reference.com/leagues/MLB/#teams_team_wins3000::none
    file_name = 'data/team_wins.csv'

    # dictionary of dictionaries
    d = defaultdict(lambda: defaultdict(int))

    # Abbreviations: Team Name pairs make it easier to deal with baseball reference csv file.
    team_abbrevs = team_dict()

    with open(file_name, 'r') as csv_file:
        reader = csv.reader(csv_file)
        headers = next(reader)
        for row in reader:
            year = int(row[0])
            for i in range(2, len(row)):
                team_abbrev = headers[i]
                full_name = team_abbrevs[team_abbrev]
                team_wins = int(row[i]) if row[i] else 0

                d[year][team_abbrev] = team_wins
                d[year][full_name] = team_wins

    return d


def avg_road_attendance():
    """

    :return: Dictionary that will be an attribute of a Teams object.
    """
    # average road attendance file scraped from ESPN.com
    file_name = 'data/avg_road_attendance.csv'

    d = defaultdict(lambda: defaultdict(int))

    with open(file_name, 'r') as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            yr = int(row[0])
            team = row[1]
            attendance = int(row[2])

            d[yr][team] = attendance

            if team == 'Florida Marlins':
                d[yr]['Miami Marlins'] = attendance

        return d
