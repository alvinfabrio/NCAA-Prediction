import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

def preprocess_team_name(team_name):
    # Preprocess the name to replace spaces and punctuation with respective URL counterpart
    team_name = team_name.replace(" ", "%20")
    team_name = team_name.replace("&", "%26")
    team_name = team_name.replace("'", "%27")
    return team_name

years = range(2013, 2022)

# Initialize a Pandas dataframe
df = pd.DataFrame(columns=["year",
                           "date",
                           "team1",
                           "team2",
                           "team1_venue",
                           "team1_outcome",
                           "team1_score",
                           "team2_score"])

# Initialize Selenium driver
options = Options()
options.headless = True
driver = webdriver.Chrome(options=options)

for year in years:

    # Get the corresponding cbb dataset for the year
    cbb_df = pd.read_csv(f"dataset/cbb{year % 2000}.csv")

    # Get 200 random teams from the CBB dataframe without replacement
    teams = np.random.choice(cbb_df["TEAM"].unique(), 250, replace=False)

    # Iterate through the teams and get the games for each team
    for team in teams:

        # Preprocess the team name
        team = preprocess_team_name(team)

        # Use headless Selenium to get the Javascript rendered table on the page
        driver.get(
            f"http://barttorvik.com/gamestat.php?sIndex=0&year={year}&tvalue={team}")
        try:
            elem = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(
                    (By.CLASS_NAME, "mobileout"))
            )
        except:
            # Skip the team if there is an error
            print(f"Error with team {team} in year {year} - skipping team.")
            continue
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')

        # Extract games from the table in the page
        games = soup.find_all("tr", style="white-space:nowrap")

        # Extract all relevant info from the games
        for game in games:
            date = game.find("td", id="_0").find("a").text
            team1_info = game.find("td", id="_2")
            team2_info = game.find("td", id="_4")
            team1_venue = game.find("td", id="_5").text
            score = game.find("td", id="_6")
            team1_name = team1_info.find("a").text
            team2_name = team2_info.text

            # Score is in the format "W/L, score1-score2" where score1 >= score2
            # Separate the score from the W/L
            win_or_loss = score.find("a").text.split(",")[0]
            score = score.find("a").text.split(",")[1]

            # Separate the scores based on if it is a W or L
            if win_or_loss[0] == "W":
                team1_score = score.split("-")[0].strip()
                team2_score = score.split("-")[1].strip()
            else:
                team1_score = score.split("-")[1].strip()
                team2_score = score.split("-")[0].strip()

            # Append the game to the dataframe
            df = df.append({"year": year,
                            "date": date,
                            "team1": team1_name,
                            "team2": team2_name,
                            "team1_venue": team1_venue,
                            "team1_outcome": win_or_loss,
                            "team1_score": team1_score,
                            "team2_score": team2_score},
                            ignore_index=True)
    
    print(f"Finished scraping for year {year}.")

# Close the Selenium driver
driver.quit()

# Save the dataframe to a CSV file
df.to_csv("dataset/games_updated.csv", index=False)
