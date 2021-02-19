from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import pandas as pd
import pickle

def main():
    url = "https://ultrasignup.com/results_event.aspx?did=77199"
    #NOTE: Chromedriver is in /Lab/CitationNetworks
    #driver = webdriver.Chrome("../../chromedriver", options=options)

    #NOTE: driver setup from: https://stackoverflow.com/questions/60296873/sessionnotcreatedexception-message-session-not-created-this-version-of-chrome
    driver = webdriver.Chrome(ChromeDriverManager().install())
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
    print(content[0])
    print(content[0][-6:])
    content = [c[-6:] for c in content]
    for c in content:
        print(c)
    df = pd.DataFrame(content, columns = cols[-6:])
    print(df.head())
    pickle.dump(df, open("BC2021_100k.p", "wb"))

    ##TODO: add first finisher, remove "Did Not Finish" line, make

if __name__ == "__main__":
    main()
