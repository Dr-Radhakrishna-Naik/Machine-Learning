# -*- coding: utf-8 -*-
"""
Created on Mon Mar  7 16:08:49 2022

@author: Dell
"""


from bs4 import BeautifulSoup
soup=BeautifulSoup(open("c:/360DG/Datasets/sample_doc.html"),'html.parser')
print(soup)
soup.text
soup.contents
soup.find('address')
soup.find_all('address')
soup.find_all('q')
soup.find_all('b')
table=soup.find('table')
table
for row in table.find_all('tr'):
    columns=row.find_all('td')
    print(columns)

#It will show all the rows except first row
#Now we want to display M.Tech which is located in third row and second column
    #I need to give [3][2]
    table.find_all('tr')[3].find_all('td')[2]
