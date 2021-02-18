from selenium import webdriver
from selenium.webdriver.support.ui import Select
from selenium.webdriver.chrome.options import Options
from webdriver_manager.chrome import ChromeDriverManager
import os
import time
import pandas as pd

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
    cols = runner_rows[0:10]
    content = runner_rows[11:]
    print(content)

if __name__ == "__main__":
    main()
