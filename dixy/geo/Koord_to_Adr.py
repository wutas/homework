#!/usr/bin/env python
# coding: utf-8


import requests
import json
import pyodbc
import pandas as pd
import numpy as np
from tqdm import tqdm_notebook
from time import sleep

def dup(k):
    newlist = []
    k1 = []
    k = list(k)
    for a in k:
        k1.append(a + '1')
    k = k1
    for i, v in enumerate(k):
        totalcount = k.count(v)
        count = k[:i].count(v)
        newlist.append(v[:-1] + str(count + 1) if totalcount > 1 else v)
    return np.array(newlist)


def main():
    conn = pyodbc.connect('Driver={SQL Server};'
                              'Server=srv-dwh-lsn,15025;'
                              'Database=dwh;'
                              'Trusted_Connection=yes;')

    sql =   'SELECT SiteID as SiteExtID,[Latitude] as Shir,[Longitude] as Dolg, [SiteAddress] as Adres'\
            'FROM [Presentation].[dim].[Shops_new] '         'WHERE SiteID !=0 and Latitude !=0'
    df = (pd.read_sql(sql, conn))
    conn.close()


    print('1 step')


    n=0
    name = []
    kind = []
    Store = []
    StorePROBLEM = []
    dfDOLG = df[~pd.isnull(df.Dolg)]
    dfDOLG.reset_index(drop = True, inplace =True)
    print(dfDOLG.shape[0])
    for i in range(0,len(dfDOLG.Shir)):
        try:
            n+=1
            if n == 100:
                print(f'{dfDOLG.shape[0] - i} left')
                sleep(60)
                n=0
            url = 'https://geocode-maps.yandex.ru/1.x/?apikey=21b909b3-7ef8-400d-9a15-892f5ebd692a&geocode=' \
                    + str(dfDOLG.Dolg[i]) +','+ str(dfDOLG.Shir[i])+'&format=json'
            r = requests.get(url)
            response = json.loads(r.text)
            features = response['response']['GeoObjectCollection']['featureMember']
            components = features[0]['GeoObject']['metaDataProperty']['GeocoderMetaData']['Address']['Components']
            for j in range(0,len(components)):
                name.append(components[j]['name'])
                kind.append(components[j]['kind'])
                Store.append(dfDOLG.SiteExtID[i])
        except:
            print(f'!!!!! {dfDOLG.SiteExtID[i]}')
            StorePROBLEM.append(dfDOLG.SiteExtID[i])
            sleep(120)
            print('Try again')

    df1=pd.DataFrame(columns=['Store','kind','name'])
    df1["Store"] = Store 
    df1["kind"] = kind 
    df1["name"] = pd.Series(name).replace('Нан',np.nan).replace('Nan',np.nan)


    print('2 step')


    name1 = []
    kind1 = []
    Store1 = []
    dfADR = df[(pd.isnull(df.Dolg)) & ~(pd.isnull(df.Adres))]
    dfADR.reset_index(drop = True,inplace =True)
    print(dfADR.shape[0])
    for i in range(0,len(dfADR.Shir)):
        try:
            n+=1
            if n == 100:
                print(f'{dfADR.shape[0] - i} left')
                sleep(60)
                n=0
            url = 'https://geocode-maps.yandex.ru/1.x/?apikey=21b909b3-7ef8-400d-9a15-892f5ebd692a&geocode=' \
                    + str(dfADR.Adres[i]) +'&format=json'
            r = requests.get(url)
            response = json.loads(r.text)
            features = response['response']['GeoObjectCollection']['featureMember']
            components = features[0]['GeoObject']['metaDataProperty']['GeocoderMetaData']['Address']['Components']
            for j in range(0,len(components)):
                name1.append(components[j]['name'])
                kind1.append(components[j]['kind'])
                Store1.append(dfADR.SiteExtID[i])
        except: 
            print(f'!!!!! {dfADR.SiteExtID[i]}')
            StorePROBLEM.append(dfADR.SiteExtID[i])
            sleep(120)
            print('Try again')



    df[df['SiteExtID'].isin(StorePROBLEM)]


    df3=pd.DataFrame(columns=['Store','kind','name'])
    df3["Store"] = Store1 
    df3["kind"] = kind1 
    df3["name"] = pd.Series(name1).replace('Нан',np.nan).replace('Nan',np.nan)



    print('3 step')


    name2 = []
    kind2 = []
    Store2 = []
    StorePROBLEM2 = []
    dfprobl = df[df['SiteExtID'].isin(StorePROBLEM)]
    dfprobl.reset_index(drop = True,inplace =True)
    for i in range(0,len(dfprobl.Shir)):
        try:
            n+=1
            if n == 100:
                print('ಠ_ಠ')
                sleep(60)
                n=0
            url = 'https://geocode-maps.yandex.ru/1.x/?apikey=21b909b3-7ef8-400d-9a15-892f5ebd692a&geocode=' \
                    + str(dfprobl.Dolg[i]) +','+ str(dfprobl.Shir[i])+'&format=json'
            r = requests.get(url)
            response = json.loads(r.text)
            features = response['response']['GeoObjectCollection']['featureMember']
            components = features[0]['GeoObject']['metaDataProperty']['GeocoderMetaData']['Address']['Components']
            for j in range(0,len(components)):
                name2.append(components[j]['name'])
                kind2.append(components[j]['kind'])
                Store2.append(dfprobl.SiteExtID[i])
        except:
            print(f'!!!!! {dfprobl.SiteExtID[i]}')
            StorePROBLEM2.append(dfprobl.SiteExtID[i])
            sleep(120)
            print('Try again')


    df5=pd.DataFrame(columns=['Store','kind','name'])
    df5["Store"] = Store2 
    df5["kind"] = kind2 
    df5["name"] = pd.Series(name2).replace('Нан',np.nan).replace('Nan',np.nan)



    df4 = pd.concat([df1, df3,df5])

    df4.dropna(inplace = True)
    df4.reset_index(drop = True,inplace =True)


    print('4 step')


    st = df1.Store.unique()
    df2 = pd.DataFrame()
    for i in range(0,len(st)):
        k = dup(df1.loc[df1.Store == st[i],'kind'])
        n = np.array(df1.loc[df1.Store == st[i],'name'])
        df2.loc[i,'Store'] = st[i]
        for j in range(0,len(k)):
            df2.loc[i,k[j]] = n[j]
            


    np.savetxt('StorePROBLEM2.txt', StorePROBLEM2, delimiter=',', fmt="%s")

    df_melt = pd.melt(df2, id_vars='Store', value_vars=df2.columns[1:])
    df_melt.dropna(inplace=True)
    df_melt.reset_index(inplace=True,drop=True)
    df_melt['Store'] = df_melt['Store'].apply(int)
    df_melt.to_csv("D:\\Adres_from_Yandex\\Stores_melt.csv",encoding='utf-8-sig',index=False)

if __name__ == '__main__':
    main()
