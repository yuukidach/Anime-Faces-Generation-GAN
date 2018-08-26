import requests
import os
import traceback
from bs4 import BeautifulSoup
from selenium import webdriver  
from selenium.common.exceptions import NoSuchElementException  

def download_image(url, filename):
    if os.path.exists(filename):
        return
    try:
        r = requests.get(url, proxies=PROXIES, stream=True, timeout=60)
        if r.status_code == 200:
            open(filename, 'wb').write(r.content)
    except KeyboardInterrupt:
        if os.path.exists(filename):
            os.remove(filename)
    except Exception:
        traceback.print_exc()
        if os.path.exists(filename):
            os.remove(filename)

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Ubuntu Chromium/68.0.3440.106 Chrome/68.0.3440.106 Safari/537.36",
    'host': 'www.getchu.com',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Upgrade-Insecure-Requests': '1',
    'Connection': 'close'
}


PROXIES = {
    'http': 'socks5://127.0.0.1:1080',
    'https': 'socks5://127.0.0.1:1080',
}

if __name__ == "__main__":
    if os.path.exists('crypko_sample') is False:
        os.mkdir('crypko_sample')
    
    for i in range(1, 10):
        print(i)
        url = 'https://crypko.ai/#/card/1%d' % i
        try:
            html = requests.get(url, proxies=PROXIES, timeout=20).text
            print(html)
            soup = BeautifulSoup(html, 'html.parser')
            img = soup.find('img', class_='progressive-image-main')
            target_url = img['src']
            print(target_url)
            filename = os.path.join('getchu_sample', target_url.split('/')[-1])
            download_image(target_url, filename) 
        except:
            continue
         
    