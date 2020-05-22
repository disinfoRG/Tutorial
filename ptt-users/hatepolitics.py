import numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
import itertools
from datetime import *

data_dir = Path(".")


def load_data(*path_globs):
    df_all = pd.concat(
        [
            pd.read_json(json_file, lines=True, encoding="utf-8")
            for json_file in itertools.chain(
                *[data_dir.glob(path_glob) for path_glob in path_globs]
            )
        ],
        ignore_index=True,
    )
    # take the lastest version of each publication in the dataset
    return (
        df_all.sort_values("version").groupby("id").last().sort_values("published_at")
    )


hate_politics = load_data("2019-12-*.jsonl", "2020-01-*.jsonl")


def convert(df):
    for f in ["published_at", "first_seen_at", "last_updated_at"]:
        df[f] = pd.to_datetime(df[f])
    df["published_date"] = df.published_at.map(lambda d: d.date())


convert(hate_politics)

df = hate_politics


def counts_by_day(df):
    return df.groupby("published_date").size()


plt.figure(figsize=(12.8, 9.6))
plt.plot(counts_by_day(df))
plt.savefig("usage.png")

print(counts_by_day(df[df.published_date < date(2020, 1, 1)]).describe())
print(counts_by_day(df[df.published_date > date(2020, 1, 13)]).describe())
authors = df.author.value_counts()
print(authors.describe())

top_authors = authors[authors >= 8.0]
top_author_stats = top_authors.to_frame().apply(
    lambda author: counts_by_day(df[df.author == author.name]).describe(), axis=1
)
print(top_author_stats)

consistent_authors = top_author_stats[
    (top_author_stats["std"] < 1.2) & (top_author_stats["50%"] >= 5.0)
]
print(consistent_authors)

print(counts_by_day(df[df.author == "nawabonga"]))

lookup_table = (
    df[["author", "connect_from"]]
    .groupby(["author", "connect_from"])
    .size()
    .to_frame("count")
    .reset_index()
    .pivot("author", "connect_from", "count")
)


def ianalyseur(table, username):
    """
    List the IP addresses ever used by user `username`, with all of the users that have ever used each of these IPs and the number of times they have used it.
    """
    connect_from_user = table.loc[username]
    data = {}
    for ip in connect_from_user[connect_from_user > 0].index:
        connect_from_ip = table[ip]
        for user in connect_from_ip[connect_from_ip > 0].index:
            data[(ip, user)] = int(connect_from_ip.loc[user])
    return pd.DataFrame(
        data.values(), index=pd.MultiIndex.from_tuples(data.keys()), columns=["count"]
    )


print(ianalyseur(lookup_table, "nawabonga"))
