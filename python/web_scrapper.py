import requests
import urllib.request
import time
from bs4 import BeautifulSoup
import selenium as se
from selenium import webdriver

'''
url = "http://www.ushospitalfinder.com/hospitals/search?search_query=60641"
req = requests.get(url, timeout=(3.0,300))
soup = BeautifulSoup(req.text, 'html.parser')
print(req)
for td in soup.find_all('a'):
    print(td.get('href'))
'''

from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options

options = webdriver.ChromeOptions()
options.add_argument('--headless')
options.add_argument('window-size=1200x600')
driver = webdriver.Chrome(ChromeDriverManager().install(), options=options)

url = "http://www.ushospitalfinder.com/hospitals/search?search_query=60641"
driver.get(url)
time.sleep(10)
print("Chrome Browser Initialized in Headless Mode")
a = None
html = driver.page_source
soup = BeautifulSoup(html)
a= soup.find('table').find('a')
driver.quit()
print("Driver Exited")
f = open('website.txt','w')
f.write(html)
f.close()
print(a)

