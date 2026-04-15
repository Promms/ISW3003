from torchvision.datasets import VOCSegmentation

dataset = VOCSegmentation(root="./data", year="2012", image_set="train", download=True)

sizes = [dataset[i][0].size for i in range(len(dataset))]
widths  = [s[0] for s in sizes]
heights = [s[1] for s in sizes]

print(f"전체 이미지 수: {len(sizes)}")
print(f"width  unique: {sorted(set(widths))}")
print(f"height unique: {sorted(set(heights))}")
print(f"width  max/min: {max(widths)} / {min(widths)}")
print(f"height max/min: {max(heights)} / {min(heights)}")