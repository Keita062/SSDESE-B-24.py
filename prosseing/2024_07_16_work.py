#ライブラリを読み込む
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels as sm
import scipy.stats as stats
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from scipy import stats
#データを読み込む
data = pd.read_csv('SSDSE-B-2024 - English.csv', encoding='shift-jis',header=1)
data= data.drop(columns='area code')
#移動数=転入数-転出数
data["net_migration"]=data["Number of new arrivals (Japanese migrants)"]-data["Number of out-migrants (Japanese moving out)"]
data["Number of Institutions"]=data["Number of kindergartens"]+data["Number of Elementary Schools"]+data["Number of secondary schools"]+data["Number of high schools"]+data["Number of junior colleges"]+data["Number of universities"]+data["Number of Specialized Training Schools"]+data["Number of schools"]
#必要なカラムを選択
columns_needed = ['year', 'prefecture', 'Number of new arrivals (Japanese migrants)', 'Number of transferees (Japanese migrants) (Male)','Number of transferees (Japanese migrants) (Female)', 'Number of out-migrants (Japanese moving out)', 'Number of out-migrants (Japanese moving out)(Male)', 'Number of out-migrants (Japanese moving out)(Female)']
#データとして表示
data_comn1 = data[columns_needed]
# yearとprefectureを消して新しい変数に格納
data_comn2 = data_comn1.drop(columns=(['year',"prefecture"]))
# 相関行列の計算
correlation_matrix = data_comn2.corr()
# ヒートマップで相関行列を表示
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Heatmap of correlation matrix')
plt.show()
#それぞれは散布図。対応点は棒グラフ
plt.figure(figsize=(40, 30))
sns.pairplot(data_comn2)
plt.show()
#転入数と転出数の対応散布図とy=xのグラフの描画

data_ave=data_comn1.groupby('prefecture').mean().reset_index().sort_values('Number of out-migrants (Japanese moving out)(Female)', ascending=True)


y = data_ave["Number of new arrivals (Japanese migrants)"] #転入数
x = data_ave["Number of out-migrants (Japanese moving out)"] #転出数
prefectures = data_ave["prefecture"] # 都道府県名

# グラフの描画
plt.figure(figsize=(10, 6))
colors = plt.cm.tab20(range(len(prefectures)))  # 色の設定
for i, (xi, yi, prefecture) in enumerate(zip(x, y, prefectures)):
    plt.scatter(xi, yi, color=colors[i], label=prefecture)
    plt.text(xi, yi, prefecture, fontsize=9)

plt.ylim(0, 420000)
plt.xlim(0, 420000)
plt.xlabel("Number of new arrivals (Japanese migrants)")#転出数
plt.ylabel("Number of out-migrants (Japanese moving out)")#転入数

plt.title("Scatter plot of new arrivals and out-migrants by prefecture")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

x_line = np.linspace(0, 420000, 100)
y_line = x_line
plt.plot(x_line, y_line, color='red', linestyle='-', label='y=x')
plt.show()



# 転入数が転出数より多い都道府県を出力
in_migrants_more_than_out_migrants = prefectures[y > x]
print("転入数が転出数より上回っている都道府県:")
print(in_migrants_more_than_out_migrants)

in_migrants_more_than_out_migrants= prefectures[x > y]
print("転出数が転入数より上回っている都道府県:")
print(in_migrants_more_than_out_migrants)

# 教育機関数と純移動の散布図と回帰直線を描画
plt.figure(figsize=(10, 6))
sns.regplot(x='Number of Institutions', y='net_migration', data=data, ci=None,line_kws={'color': 'red'})
plt.xlabel('Number of Institutions')
plt.ylabel('Net Migration')
plt.title('Regression Analysis: Net Migration vs Number of Institutions')
plt.show()

positive_net_migration = data[data['net_migration'] > 0]
print("純移動が正の都道府県:")
print(positive_net_migration[['prefecture', 'net_migration', 'Number of Institutions']])

negative_net_migration = data[data['net_migration'] < 0]
print("純移動が負の都道府県:")
print(negative_net_migration[['prefecture', 'net_migration', 'Number of Institutions']])

columns_needed3 = ['year', 'prefecture', 'Number of new arrivals (Japanese migrants)', 'Number of transferees (Japanese migrants) (Male)','Number of transferees (Japanese migrants) (Female)', 'Number of out-migrants (Japanese moving out)', 'Number of out-migrants (Japanese moving out)(Male)', 'Number of out-migrants (Japanese moving out)(Female)',"Number of Institutions","net_migration"]
data_comn3 = data[columns_needed3]

data_2010 = data_comn3[data_comn1['year'] == 2010]
data_2011 = data_comn3[data_comn1['year'] == 2011]
data_2012 = data_comn3[data_comn1['year'] == 2012]
data_2013 = data_comn3[data_comn1['year'] == 2013]
data_2014 = data_comn3[data_comn1['year'] == 2014]
data_2015 = data_comn3[data_comn1['year'] == 2015]
data_2016 = data_comn3[data_comn1['year'] == 2016]
data_2017 = data_comn3[data_comn1['year'] == 2017]
data_2018 = data_comn3[data_comn1['year'] == 2018]
data_2019 = data_comn3[data_comn1['year'] == 2019]
data_2020 = data_comn3[data_comn1['year'] == 2020]
data_2021 = data_comn3[data_comn1['year'] == 2021]

#2010年の教育機関数と純移動の散布図と回帰直線を描画

# 2010年のデータをフィルタリング
data_2010 = data[data['year'] == 2010]

# 散布図と回帰直線を描画
plt.figure(figsize=(10, 6))
sns.regplot(x='Number of Institutions', y='net_migration', data=data_2010, ci=None, line_kws={'color': 'red'})
plt.xlabel('Number of Institutions')
plt.ylabel('Net Migration')
plt.title('Regression Analysis: Net Migration vs Number of Institutions (2010)')
plt.show()
positive_net_migration = data_2010[data_2010['net_migration'] > 0]
print("純移動が正の都道府県:")
print(positive_net_migration[['prefecture', 'net_migration', 'Number of Institutions']])

negative_net_migration = data_2010[data_2010['net_migration'] < 0]
print("純移動が負の都道府県:")
print(negative_net_migration[['prefecture', 'net_migration', 'Number of Institutions']])

#2010年から2021年までの教育機関数と純移動の散布図と回帰直線を描画

# 2010年から2021年までの各年について処理を実行
for year in range(2010, 2022):
    # 各年のデータをフィルタリング
    data_year = data[data['year'] == year]

    # 散布図と回帰直線を描画
    plt.figure(figsize=(10, 6))
    sns.regplot(x='Number of Institutions', y='net_migration', data=data_year, ci=None, line_kws={'color': 'red'})
    plt.xlabel('Number of Institutions')
    plt.ylabel('Net Migration')
    plt.title(f'Regression Analysis: Net Migration vs Number of Institutions ({year})')
    plt.show()

    # 純移動が正の都道府県を抽出
    positive_net_migration = data_year[data_year['net_migration'] > 0]
    print(f"{year}年: 純移動が正の都道府県:")
    print(positive_net_migration[['prefecture', 'net_migration', 'Number of Institutions']])

    # 純移動が負の都道府県を抽出
    negative_net_migration = data_year[data_year['net_migration'] < 0]
    print(f"{year}年: 純移動が負の都道府県:")
    print(negative_net_migration[['prefecture', 'net_migration', 'Number of Institutions']])

#2010年から2021年までの教育機関数と純移動の散布図と回帰直線を描画

# 2010年から2021年までの各年について処理を実行
for year in range(2010, 2022):
    # 各年のデータをフィルタリング
    data_year = data[data['year'] == year]

    # 純移動が正の都道府県を抽出
    positive_net_migration = data_year[data_year['net_migration'] > 0]
    print(f"{year}年: 純移動が正の都道府県:")
    print(positive_net_migration[['prefecture', 'net_migration', 'Number of Institutions']])

#2010年から2021年までの教育機関数と純移動の都道府県別の折れ線グラフ
prefectures = data['prefecture'].unique()

# 各都道府県ごとにプロットを作成
for prefecture in prefectures:
    plt.figure(figsize=(10, 6))

    prefecture_data = data[data['prefecture'] == prefecture]
    plt.plot(prefecture_data['Number of Institutions'], prefecture_data['net_migration'], marker='o', label=prefecture)

    for i, row in prefecture_data.iterrows():
        plt.text(row['Number of Institutions'], row['net_migration'], str(row['year']), fontsize=8)

    plt.xlabel('Number of Institutions')
    plt.ylabel('Net Migration')
    plt.title(f'Net Migration vs Number of Institutions in {prefecture} (2010-2021)')
    plt.legend(loc='upper left')
    plt.show()

# 純移動が正の年度を出力し、その年度の移動数と教育機関数を数値で出力
for prefecture in prefectures:
    prefecture_data = data[data['prefecture'] == prefecture]
    positive_net_migration_years = prefecture_data[prefecture_data['net_migration'] > 0]

    if not positive_net_migration_years.empty:
        print(f"都道府県: {prefecture}")
        for idx, row in positive_net_migration_years.iterrows():
            print(f"年度: {row['year']}, 純移動: {row['net_migration']}, 教育機関数: {row['Number of Institutions']}")

columns_needed4 = ["Number of Institutions","net_migration"]
data_comn4 = data[columns_needed4]
correlation_matrix = data_comn4.corr()
print(correlation_matrix)
