import os

import pandas as pd
import matplotlib.pyplot as plt
from datasets import load_dataset

REPO_ID = "petkopetkov/student-math-misunderstandings"
OUT_DIR = "eda_results"
os.makedirs(OUT_DIR, exist_ok=True)

def preview_df(df: pd.DataFrame, name: str, n: int = 5):
    print(f"\n===== {name} preview (n={n}) =====")
    with pd.option_context("display.max_colwidth", 160):
        print(df.head(n))

def save_bar(value_counts, title, xlabel, ylabel, out_path, rotation=0, top_n=None):
    plt.figure()
    if top_n is not None:
        value_counts = value_counts.head(top_n)
    value_counts.plot(kind="bar")
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.xticks(rotation=rotation, ha="right")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

def words(s):
    if not isinstance(s, str):
        return []
    return s.strip().split()

def norm_text(s):
    if not isinstance(s, str):
        return ""
    return " ".join(s.lower().split())

def safe_len(x):
    try:
        return len(x)
    except Exception:
        return 0

def stratify_category_parts(cat: str):
    """
    Splits e.g. 'True_Misconception' -> ('True', 'Misconception')
    Expect left in {'True','False'} and right in {'Correct','Misconception','Neither'}
    """
    if not isinstance(cat, str) or "_" not in cat:
        return None, None
    left, right = cat.split("_", 1)
    return left, right

print("Loading dataset from HF Hub:", REPO_ID)
ds = load_dataset(REPO_ID)

# Expect splits train and test (train has labels, test not)
train_hf = ds.get("train")
test_hf = ds.get("test")

if train_hf is None:
    raise RuntimeError("No 'train' split found in the dataset.")
if test_hf is None:
    print("Warning: no 'test' split found. Proceeding with train-only EDA.")

train = train_hf.to_pandas()
test = test_hf.to_pandas() if test_hf is not None else None

print("\n===== Shapes =====")
print("train:", train.shape)
if test is not None:
    print("test: ", test.shape)

print("\n===== Columns =====")
print("train:", list(train.columns))
if test is not None:
    print("test: ", list(test.columns))

preview_df(train, "train", 5)
if test is not None:
    preview_df(test, "test", 5)

print("\n===== Missing values (train) =====")
print(train.isna().sum())

if test is not None:
    print("\n===== Missing values (test) =====")
    print(test.isna().sum())

for df, name in [(train, "train")] + ([(test, "test")] if test is not None else []):
    df["QuestionText_charlen"] = df["QuestionText"].astype(str).map(len)
    df["StudentExplanation_charlen"] = df["StudentExplanation"].astype(str).map(len)
    df["QuestionText_wordlen"] = df["QuestionText"].astype(str).map(lambda s: len(words(s)))
    df["StudentExplanation_wordlen"] = df["StudentExplanation"].astype(str).map(lambda s: len(words(s)))

    print(f"\n===== {name} text length summary =====")
    print(df[[
        "QuestionText_charlen", "StudentExplanation_charlen",
        "QuestionText_wordlen", "StudentExplanation_wordlen"
    ]].describe(percentiles=[.5, .75, .9, .95, .99]))

    # Histograms
    for col, ttl in [
        ("StudentExplanation_wordlen", f"{name}: StudentExplanation word length"),
        ("QuestionText_wordlen", f"{name}: QuestionText word length"),
    ]:
        plt.figure()
        df[col].hist(bins=50)
        plt.title(ttl)
        plt.xlabel("words")
        plt.ylabel("count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f"{name}_{col}_hist.png"))
        plt.close()

if "Category" in train.columns:
    cat_vc = train["Category"].value_counts(dropna=False).sort_values(ascending=False)
    print("\n===== Category distribution (train) =====")
    print(cat_vc)
    save_bar(
        cat_vc,
        "Category distribution (train)",
        "Category",
        "Count",
        os.path.join(OUT_DIR, "train_category_distribution.png"),
        rotation=45
    )

    train["AnswerCorrectness"], train["ExplanationFlag"] = zip(*train["Category"].map(stratify_category_parts))
    print("\n===== AnswerCorrectness vs ExplanationFlag (train) =====")
    ctab = pd.crosstab(train["AnswerCorrectness"], train["ExplanationFlag"])
    print(ctab)
    plt.figure()
    ctab.plot(kind="bar", stacked=True)
    plt.title("AnswerCorrectness × ExplanationFlag (train)")
    plt.xlabel("AnswerCorrectness")
    plt.ylabel("Count")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "train_answercorrectness_explflag.png"))
    plt.close()

    grp = train.groupby("ExplanationFlag")["StudentExplanation_wordlen"].describe()
    print("\n===== StudentExplanation word length by ExplanationFlag (train) =====")
    print(grp)
    plt.figure()
    train.boxplot(column="StudentExplanation_wordlen", by="ExplanationFlag")
    plt.suptitle("")
    plt.title("Explanation length by ExplanationFlag (train)")
    plt.xlabel("ExplanationFlag")
    plt.ylabel("Word length")
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "train_expl_len_by_flag.png"))
    plt.close()

if "Misconception" in train.columns:
    mis_vc = train.loc[train["Misconception"].astype(str).str.upper() != "NA", "Misconception"] \
                   .value_counts().sort_values(ascending=False)
    print("\n===== Top misconceptions (train) =====")
    print(mis_vc.head(30))
    save_bar(
        mis_vc,
        "Top Misconceptions (train) – all",
        "Misconception",
        "Count",
        os.path.join(OUT_DIR, "train_top_misconceptions_all.png"),
        rotation=45
    )
    save_bar(
        mis_vc,
        "Top 25 Misconceptions (train)",
        "Misconception",
        "Count",
        os.path.join(OUT_DIR, "train_top_misconceptions_top25.png"),
        rotation=45,
        top_n=25
    )

    train["CatMis"] = train["Category"].astype(str) + ":" + train["Misconception"].astype(str)
    catmis_space = sorted(set(
        train.loc[train["Misconception"].astype(str).str.upper() != "NA", "CatMis"]
    ))
    print(f"\n===== Category:Misconception label space (train) =====\n(count={len(catmis_space)})")
    for x in catmis_space[:50]:
        print(x)
    if len(catmis_space) > 50:
        print("...")

for df, name in [(train, "train")] + ([(test, "test")] if test is not None else []):
    q_counts = df["QuestionId"].value_counts()
    print(f"\n===== {name}: students per QuestionId =====")
    print(q_counts.describe())
    save_bar(
        q_counts.sort_values(ascending=False).head(30),
        f"{name}: Top 30 QuestionIds by responses",
        "QuestionId",
        "Count",
        os.path.join(OUT_DIR, f"{name}_top_questionids.png"),
        rotation=45
    )

if "Category" in train.columns:
    per_q_cat_n = train.groupby("QuestionId")["Category"].nunique().sort_values(ascending=False)
    print("\n===== train: per-QuestionId #unique Category =====")
    print(per_q_cat_n.describe())
    save_bar(
        per_q_cat_n.head(30),
        "train: Top 30 QuestionIds by Category diversity",
        "QuestionId",
        "# unique Category",
        os.path.join(OUT_DIR, "train_question_category_diversity.png"),
        rotation=45
    )

if "Misconception" in train.columns:
    per_q_mis_n = train.loc[train["Misconception"].astype(str).str.upper() != "NA"] \
                     .groupby("QuestionId")["Misconception"].nunique().sort_values(ascending=False)
    print("\n===== train: per-QuestionId #unique Misconception (excluding NA) =====")
    print(per_q_mis_n.describe())
    save_bar(
        per_q_mis_n.head(30),
        "train: Top 30 QuestionIds by Misconception diversity",
        "QuestionId",
        "# unique Misconception",
        os.path.join(OUT_DIR, "train_question_misconception_diversity.png"),
        rotation=45
    )

if test is not None:
    train_qids = set(train["QuestionId"].tolist())
    test_qids = set(test["QuestionId"].tolist())
    overlap_qids = train_qids & test_qids
    print(f"\n===== Leakage check: QuestionId overlap =====")
    print(f"train unique QuestionId: {len(train_qids)}")
    print(f"test  unique QuestionId: {len(test_qids)}")
    print(f"overlap QuestionId: {len(overlap_qids)}")
    if len(overlap_qids) > 0:
        pct = 100.0 * len(overlap_qids) / max(1, len(test_qids))
        print(f"Overlap as % of test unique QuestionId: {pct:.2f}%")

    key_cols = ["QuestionId", "QuestionText", "MC_Answer", "StudentExplanation"]
    train_norm = train[key_cols].copy().applymap(norm_text)
    test_norm = test[key_cols].copy().applymap(norm_text)

    train_norm["__join_key__"] = train_norm.apply(lambda r: "||".join(map(str, r.values)), axis=1)
    test_norm["__join_key__"] = test_norm.apply(lambda r: "||".join(map(str, r.values)), axis=1)
    dup_join = set(train_norm["__join_key__"]) & set(test_norm["__join_key__"])
    print(f"\n===== Leakage check: exact normalized row duplicates across train/test =====")
    print(f"Exact duplicates: {len(dup_join)}")

    for df, name in [(train, "train"), (test, "test")]:
        de = df["StudentExplanation"].astype(str).map(norm_text)
        dup_ct = de.duplicated().sum()
        uniq_ct = de.nunique()
        print(f"{name}: {dup_ct} duplicated explanations; {uniq_ct} unique explanations out of {len(df)} rows.")

for df, name in [(train, "train")] + ([(test, "test")] if test is not None else []):
    if "MC_Answer" in df.columns:
        mc_vc = df["MC_Answer"].astype(str).value_counts()
        print(f"\n===== {name}: MC_Answer distribution =====")
        print(mc_vc.head(20))
        save_bar(
            mc_vc.head(25),
            f"{name}: Top MC_Answer values",
            "MC_Answer",
            "Count",
            os.path.join(OUT_DIR, f"{name}_mc_answer_top25.png"),
            rotation=45
        )

if "Category" in train.columns:
    maj_cat = train["Category"].value_counts().idxmax()
    maj_acc = (train["Category"] == maj_cat).mean()
    print("\n===== Majority baseline (train) =====")
    print(f"Most frequent Category: {maj_cat} (acc vs train labels = {maj_acc:.4f})")
    train["short_expl"] = train["StudentExplanation_wordlen"] <= train["StudentExplanation_wordlen"].median()
    corr_table = pd.crosstab(train["short_expl"], train["ExplanationFlag"], normalize="index")
    print("\n===== Correlation: short explanation vs ExplanationFlag (normalized by row) =====")
    print(corr_table)

if "Misconception" in train.columns:
    top_mis_tbl = mis_vc.reset_index()
    top_mis_tbl.columns = ["Misconception", "Count"]
    top_mis_tbl.to_csv(os.path.join(OUT_DIR, "train_top_misconceptions.csv"), index=False)

q_agg = train.groupby("QuestionId").agg(
    n_rows=("QuestionId", "size"),
    n_unique_students=("StudentExplanation", "nunique"),
    avg_expl_words=("StudentExplanation_wordlen", "mean"),
)
if "Category" in train.columns:
    q_agg["n_unique_category"] = train.groupby("QuestionId")["Category"].nunique()
if "Misconception" in train.columns:
    q_agg["n_unique_misconception"] = train.groupby("QuestionId")["Misconception"].nunique()
q_agg = q_agg.sort_values("n_rows", ascending=False)
q_agg.to_csv(os.path.join(OUT_DIR, "train_question_aggregates.csv"))

print(f"\nAll figures and tables saved to: {OUT_DIR}/")
print("Done.")
