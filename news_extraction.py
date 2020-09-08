from bs4 import BeautifulSoup
from selenium import webdriver
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.chrome.options import Options
import csv
options = Options()
options.add_argument('--headless')
options.add_argument('--disable-gpu')
options.add_argument('--log-level=3')
driver=webdriver.Chrome(ChromeDriverManager().install(),options=options)

with open('stock.csv', mode='w',newline='') as file:
    writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    comp=[]
    driver.get("https://www.moneycontrol.com/india/stockpricequote/")
    html= driver.execute_script("return document.documentElement.outerHTML")
    soup=BeautifulSoup(html,'html.parser')
    h=soup.find_all('tr',{'bgcolor':"#f6f6f6"})
    for ele in h:
        f=ele.find_all('td')
        for el in f:
    ##       print(el.a['href'])
    ##        print(el.a['href'].split('/'))
            a=el.a['href'].split('/')[6::]
            if len(a)==2:
                print(a)
                comp.append(a)
    
    print(len(comp))
    
    for ele in comp:
        link="http://www.moneycontrol.com/company-article/"+ele[0]+"/news/"+ele[1]
        for i in range(2):
            driver.get(link)
            html= driver.execute_script("return document.documentElement.outerHTML")
            soup=BeautifulSoup(html,'html.parser')
            try:
                h=soup.find_all('div',{'style':'width:550px'},class_='FL')
                for el in h:
                    print(el.a.strong.text)
                    writer.writerow([el.a.strong.text])
                f=soup.find('div',class_="pages MR10 MT15")
                if i==0:
                    print(f.a['href'])
                    link="https://www.moneycontrol.com"+f.a['href']
            except:
                continue
            
##print(comp)                  
driver.quit()        
