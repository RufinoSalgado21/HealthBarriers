import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import selenium as se
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.support import expected_conditions as EC

def find_nearest_hospital(driver, zipcode):
    zip = zipcode
    print('finding nearest hospital...')
    if zip == 'None' or zip is None:
        return 'None'

    url = "http://www.ushospitalfinder.com/hospitals/search?search_query=" + zip
    driver.get(url)
    #time.sleep(10)
    delay = 2
    #driver.implicitly_wait(10)
    #elem = driver.find_element_by_id('list')
    myElem = WebDriverWait(driver,delay).until(EC.presence_of_element_located((By.ID, 'list')))

    #time.sleep(5)
    #print("Chrome Browser Initialized in Headless Mode")
    a = None
    html = driver.page_source
    soup = BeautifulSoup(html, features="html5lib")
    try:
        a = soup.find("div", {"id": "list"})
        a = a.find('a')
    except:
        a = 'None'
        return 'None'

    '''
    f = open('website.txt','w')
    f.write(html)
    f.close()
    a = str(a)
    '''
    if a is None:
        print('n')
        return 'None'

    a = lstrip_until(a,'>')
    a = a.rstrip('</a>')

    return a

def close_driver(driver):
    driver.quit()

def load_chrome_webdriver():
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('window-size=1200x600')
    driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)
    return driver


def lstrip_until(string, character):
    flag = '>'
    a = string
    for c in list(a):
        a = a.lstrip(c)
        if c == flag:
            break
    return a

def main():
    web = load_chrome_webdriver()
    find_nearest_hospital(web, '60639')
    close_driver(web)

if __name__ == '__main__':
    main()

