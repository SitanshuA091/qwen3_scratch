from datasets import load_dataset

dataset = load_dataset(
    "roneneldan/TinyStories",
    split={
        "train": "train[:50000]",
        "validation": "validation[:5000]"
    }
)

print(dataset)


for i in range(5):
    print("\n====================")
    print(f"Example {i}")
    print("====================")
    print(dataset["train"][i]["text"])