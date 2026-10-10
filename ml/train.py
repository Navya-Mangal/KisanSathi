"""Train crop-disease classifier (MobileNetV3 transfer learning).
Data layout (use ORIGINAL non-augmented PlantVillage, split first):
  data/train/<class>/*.jpg  data/val/<class>/*.jpg  data/test/<class>/*.jpg
Run: python ml/train.py --data data --epochs 8
"""
import argparse, json, torch, torch.nn as nn
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader

p = argparse.ArgumentParser()
p.add_argument("--data", default="data"); p.add_argument("--epochs", type=int, default=8)
a = p.parse_args()
dev = "cuda" if torch.cuda.is_available() else "cpu"
norm = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
tr = transforms.Compose([transforms.RandomResizedCrop(224, scale=(0.6,1)), transforms.RandomHorizontalFlip(),
     transforms.ColorJitter(0.3,0.3,0.3), transforms.GaussianBlur(3), transforms.ToTensor(), norm])
ev = transforms.Compose([transforms.Resize((224,224)), transforms.ToTensor(), norm])
ds = {k: datasets.ImageFolder(f"{a.data}/{k}", tr if k=="train" else ev) for k in ("train","val","test")}
dl = {k: DataLoader(v, batch_size=32, shuffle=(k=="train"), num_workers=2) for k,v in ds.items()}
classes = ds["train"].classes
json.dump(classes, open("backend/classes.json","w"), ensure_ascii=False)

m = models.mobilenet_v3_small(weights="DEFAULT")
for prm in m.features.parameters(): prm.requires_grad = False   # stage 1: head only
m.classifier[3] = nn.Linear(m.classifier[3].in_features, len(classes))
m.to(dev)
opt = torch.optim.AdamW(filter(lambda x: x.requires_grad, m.parameters()), 1e-3)
lossf = nn.CrossEntropyLoss()

def acc(loader):
    m.eval(); c = n = 0
    with torch.no_grad():
        for x,y in loader:
            c += (m(x.to(dev)).argmax(1).cpu()==y).sum().item(); n += len(y)
    return c/n

for e in range(a.epochs):
    if e == a.epochs//2:  # stage 2: fine-tune everything
        for prm in m.parameters(): prm.requires_grad = True
        opt = torch.optim.AdamW(m.parameters(), 1e-4)
    m.train()
    for x,y in dl["train"]:
        opt.zero_grad(); lossf(m(x.to(dev)), y.to(dev)).backward(); opt.step()
    print(f"epoch {e+1}: val acc {acc(dl['val']):.3f}")
print(f"TEST (lab) acc: {acc(dl['test']):.3f}")
torch.save(m.state_dict(), "backend/model.pt")
print("Now evaluate on PlantDoc / your own field photos -> report the gap!")
