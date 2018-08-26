import requests
import os
import traceback
from bs4 import BeautifulSoup

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
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.11; rv:47.0) Gecko/20100101 Firefox/47.0",
    'host': 'www.getchu.com',
    'Accept-Encoding': 'gzip, deflate, sdch',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Upgrade-Insecure-Requests': '1',
    'Connection': 'keep-alive'
}


PROXIES = {
    'http': 'socks5://127.0.0.1:1080',
    'https': 'socks5://127.0.0.1:1080',
}

if __name__ == "__main__":
    if os.path.exists('getchu_sample') is False:
        os.mkdir('getchu_sample')
    
    start  = 965532
    end =  1110200
    for i in range (start, end+1):
        url = 'http://www.getchu.com/soft.phtml?id=%d' % i
        try:
            html = requests.get(url, proxies=PROXIES, timeout=20).text
        except:
            continue
        soup = BeautifulSoup(html, 'html.parser')
        for img in soup.find_all('a', class_='highslide'):
            target_url = 'http://www.getchu.com' + img['href'][1:]
            print(target_url)
            filename = os.path.join('getchu_sample', target_url.split('/')[-1])
            download_image(target_url, filename)
        print('%d / %d' % (i, end))   
    