from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import pandas as pd
import pickle

def main():
    urls = [("https://ultrasignup.com/results_event.aspx?did=77199", 2021),
        ("https://ultrasignup.com/results_event.aspx?did=67039", 2020),
        ("https://ultrasignup.com/results_event.aspx?did=57827", 2019),
        ("https://ultrasignup.com/results_event.aspx?did=48278", 2018),
        ("https://ultrasignup.com/results_event.aspx?did=38965", 2017),
        ("https://ultrasignup.com/results_event.aspx?did=34087", 2016),
        ("https://ultrasignup.com/results_event.aspx?did=29244", 2015),
        ("https://ultrasignup.com/results_event.aspx?did=24355", 2014)]

    #NOTE: Chromedriver is in /Lab/CitationNetworks
    #driver = webdriver.Chrome("../../chromedriver", options=options)
    #NOTE: driver setup from: https://stackoverflow.com/questions/60296873/sessionnotcreatedexception-message-session-not-created-this-version-of-chrome
    driver = webdriver.Chrome(ChromeDriverManager().install())

    for url, year in urls:
        driver.get(url)

        #Results from: https://medium.com/@elizabeth.guy86/gender-differences-in-ultra-running-f0880047b9ed
        sel = "gbox_list"
        results = driver.find_element_by_id(sel)
        rows = results.text.split('\n')
        runner_rows = [row.split() for row in rows]
        cols = [c[0] for c in runner_rows[0:9]]
        cols.insert(0, "Description")
        cols.insert(5, "State")
        # #10th element is number of finishers"['Finishers', '-', '363']"
        #
        content = runner_rows[10:]
        content = [c[-6:] for c in content]
        df = pd.DataFrame(content, columns = cols[-6:])
        print(year)
        print(df.head())
        print()
        pickle.dump(df, open("BC" + str(year) + "_100k.p", "wb"))


if __name__ == "__main__":
    main()
