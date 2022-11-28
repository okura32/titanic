# %%
import numpy as np
import pandas as pd
from IPython.display import display


train = pd.read_csv("train.csv")  # 学習データ
test = pd.read_csv("test.csv")  # 検証用データ
display(train.head())
display(test.head())

# %%
'''
PassengerId – 乗客識別ユニークID
Survived – 生存フラグ（0 = 死亡、1 = 生存）
Pclass – チケットクラス
Name – 乗客の名前
Sex – 性別（male = 男性、female＝女性）
Age – 年齢
SibSp – タイタニックに同乗している兄弟 / 配偶者の数
parch – タイタニックに同乗している親 / 子供の数
ticket – チケット番号
fare – 料金
cabin – 客室番号
Embarked – 出港地（タイタニックへ乗った港）

pclass = チケットクラス

1 = 上層クラス（お金持ち）
2 = 中級クラス（一般階級）
3 = 下層クラス（労働階級）

Embarked = 各変数の定義は下記の通り

C = Cherbourg
Q = Queenstown
S = Southampton
'''
# %%
display(test.shape)
display(train.shape)
# %%
display(test.describe())
display(train.describe())

# %%


def kesson_table(df):
    null_val = df.isnull().sum()
    percent = 100 * df.isnull().sum() / len(df)
    kesson_table = pd.concat([null_val, percent], axis=1)
    kesson_table_ren_columns = kesson_table.rename(columns={0: "欠損数", 1: "%"})
    return kesson_table_ren_columns


display(kesson_table(train))
display(kesson_table(test))


# %%
# データセットの事前処理

# (1) 欠損データを代理データに入れ替える

# まず「Age」ですが、シンプルに train の全データの中央値（Median）を代理として使いましょう。
train["Age"] = train["Age"].fillna(train["Age"].median())

# 次に「Embarked」（出港地）ですが、こちらも2つだけ欠損データが train に含まれています。
# 他のデータを確認すると「S」が一番多い値でしたので、代理データとして「S」を使いましょう。
train["Embarked"] = train["Embarked"].fillna("S")

display(kesson_table(train))


# %%
# (2) 文字列カテゴリカルデータを数字へ変換
train["Sex"][train["Sex"] == "male"] = 0
train["Sex"][train["Sex"] == "female"] = 1
train["Embarked"][train["Embarked"] == "S"] = 0
train["Embarked"][train["Embarked"] == "C"] = 1
train["Embarked"][train["Embarked"] == "Q"] = 2

display(train.head(10))

# %%
test["Age"] = test["Age"].fillna(test["Age"].median())
test["Embarked"] = test["Embarked"].fillna("S")
display(kesson_table(test))

test["Sex"][test["Sex"] == "male"] = 0
test["Sex"][test["Sex"] == "female"] = 1
test["Embarked"][test["Embarked"] == "S"] = 0
test["Embarked"][test["Embarked"] == "C"] = 1
test["Embarked"][test["Embarked"] == "Q"] = 2
display(test.head(10))

test.Fare[152] = test.Fare.median()
display(kesson_table(test))

# %%
