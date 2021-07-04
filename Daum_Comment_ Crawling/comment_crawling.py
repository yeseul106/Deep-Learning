#다음 뉴스 '댓글많이 뉴스' 검색결과 가져오기
import urllib.request
from bs4 import BeautifulSoup
from selenium import webdriver

import pandas as pd
import time

Dictionary = {}

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

Dictionary['Title'] = news_title
Dictionary['URL'] = news_url_list

#driver 경로 변경해서 사용할 것
driver = webdriver.Chrome(executable_path='C:/Users/82109/source/chromedriver.exe')

contents_list = []

for i in range(10):
    driver.get(news_url_list[i])
    print(driver.current_url)

    time.sleep(3)
    try:
        #댓글 더보기 클릭
       while driver.find_element_by_xpath('/html/body/div[2]/div[4]/div[2]/div[1]/div[2]/div[4]/div[2]/div/div/div/div[3]/ul[2]/li[1]/div/p').text!='':
           driver.find_element_by_xpath('/html/body/div[2]/div[4]/div[2]/div[1]/div[2]/div[4]/div[2]/div/div/div/div[3]/div[3]/button').click()
           time.sleep(3)
    except:
        pass
    time.sleep(2)

    sub_list = []
    for k in range(40):
        try:
            # 댓글 추출
            contents = driver.find_element_by_xpath(
                '/html/body/div[2]/div[4]/div[2]/div[1]/div[2]/div[4]/div[2]/div/div/div/div[3]/ul[2]/li[' + str(k) + ']/div/p').text
            contents = contents.replace("\n", " ")
            sub_list.append(contents)
        except:
            pass
    time.sleep(3)

    contents_list.append(sub_list)

Dictionary['Contents'] = contents_list

print(news_title)
print(contents_list)

# f = open('write.csv','w', newline='')
# wr = csv.writer(f)
#
# df = pd.DataFrame(data= [news_title, contents_list], columns=['news_title','contents'])
# print(df)

df = pd.DataFrame(Dictionary)
print(df)




