import csv
import locale
import urllib2
from bs4 import BeautifulSoup

def team_dict():
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
    teams["FLA"] = "Florida Marlins"
    teams["HOU"] = "Houston Astros"
    teams["KCR"] = "Kansas City Royals"
    teams["ANA"] = "Los Angeles Angels"
    teams["LAD"] = "Los Angeles Dodgers"
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
    teams["Anaheim"] = "Los Angeles Angels"  # Before name change
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
    teams["Florida"] = "Florida Marlins"
    teams["Miami"] = "Florida Marlins"  # Not worth separating
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
    teams["Montreal"] = "Washington Nationals"  # Same franchise

    return teams


def save_avg_rd_attendance():
    """
    Writes a csv file in the format [year, team, avg_road_attendance]
    For the years 2008 to 2016, as scraped from "espn.com/mlb/attendance/_/year/"
    """
    locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

    espn_teams = team_dict()

    BASE_URL = "http://www.espn.com/mlb/attendance/_/year/%s/sort/awayAvg"

    import csv
    with open('data/avg_road_attendance.csv', 'w') as csvfile:
        writer = csv.writer(csvfile)

        for yr in range(2003, 2017):
            url = BASE_URL % yr

            # Parse html into a beautiful soup object
            request = urllib2.Request(url)
            response = urllib2.urlopen(request)
            soup = BeautifulSoup(response, 'html.parser')

            # All html table_rows with avg_row attendance
            attend = soup.find_all(class_='sortcell')

            for row in attend:
                try:
                    espn_team = row.parent.contents[01].contents[0].contents[0]
                except AttributeError:
                    espn_team = row.parent.contents[01].contents[0]

                team = espn_teams[espn_team]
                attendance = locale.atoi(row.contents[0])

                # Write row to csv file
                writer.writerow([yr, team, attendance])

if __name__ == '__main__':
    save_avg_rd_attendance()
