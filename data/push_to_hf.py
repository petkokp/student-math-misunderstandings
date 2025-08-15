from datasets import load_dataset, Features, Value

features = Features({
    "row_id": Value("int64"),
    "QuestionId": Value("int64"),
    "QuestionText": Value("string"),
    "MC_Answer": Value("string"),
    "StudentExplanation": Value("string"),
    "Category": Value("string"),
    "Misconception": Value("string"),
})

ds = load_dataset(
    "csv",
    data_files="train.csv",
    features=features,
    encoding="utf-8"
)

print(ds)
print(ds["train"][0])

repo_id = "petkopetkov/student-math-misunderstandings"

ds.push_to_hub(repo_id)

print(f"Dataset pushed to {repo_id}")
