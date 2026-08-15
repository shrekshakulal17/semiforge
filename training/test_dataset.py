print("TEST STARTED")

from dataset import RestorationDataset

print("DATASET IMPORTED")

dataset = RestorationDataset("datasets/DIV2K_train_HR")

print("Number of images:", len(dataset))

bad, good = dataset[0]

print("Bad shape:", bad.shape)
print("Good shape:", good.shape)

print("TEST FINISHED")

from dataset import RestorationDataset

dataset = RestorationDataset(
    "datasets/DIV2K_train_HR"
)

print("Number of images:", len(dataset))

bad, good = dataset[0]

print("Bad shape:", bad.shape)
print("Good shape:", good.shape)