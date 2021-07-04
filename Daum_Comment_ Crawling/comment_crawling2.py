#댓글을 작성할 시각을 모방하기 위한 학습 데이터

import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver

import pandas as pd
import time

url = 'https://news.daum.net/ranking/bestreply'
html = urllib.request.urlopen(url).read()
soup = BeautifulSoup(html,'html.parser')

title = soup.find_all(class_='tit_thumb')

num=1
print('------다음 뉴스 : 댓글 많은 뉴스------')
print()

news_url_list=[]
news_title=[]

for i in title:
    if (num>50):
        break

    news_title.append(i.select_one('.link_txt').text)
    news_url_list.append(i.select_one('.link_txt').attrs['href']) #url 리스트에 정리
    num+=1

#driver 경로 변경해서 사용할 것
driver = webdriver.Chrome(executable_path='C:/Users/82109/source/chromedriver.exe')

commentTime_list = []

for i in range(10):
    driver.get(news_url_list[i])
    print(driver.current_url)

    time.sleep(3)