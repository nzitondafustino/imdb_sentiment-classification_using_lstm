# imdb sentiment classification using lstm

This project usese LSTM for sentiment Classification


## Downlaod dataset 

```
!curl -O https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
!tar -xf aclImdb_v1.tar.gz
```

## Clean data

```

def get_file(dir):
  path = []
  for par, dirs, files in os.walk("aclImdb/" + dir):
    if "neg" in par or "pos" in par:
      path.extend([par + "/" + f for f in files])
  return path
  
train_files = get_file("train")
test_files = get_file("test")

def yield_tokens(file_paths):
  for file_path in file_paths:
    with io.open(file_path, encoding = 'utf-8') as f:
      yield f.read().strip().lower().replace("<br />", " ").translate(str.maketrans('', '', string.punctuation)).split(" ")

vocab = build_vocab_from_iterator(yield_tokens(train_files), specials=["<unk>"],min_freq=10)
vocab.set_default_index(0)

```

## Dataset and dataloader


```

class MyDataset(Dataset):


  def __init__(self, files) -> None:
    self.files = files


  def __len__(self):

    return len(self.files)


  def __getitem__(self, index):

    path = self.files[index]
    label = 1 if "pos" in path else 0
    with io.open(path, encoding = 'utf-8') as f:
      data =  f.read().strip().lower().replace("<br />", " ").translate(str.maketrans('', '', string.punctuation)).split(" ")
    return torch.LongTensor(vocab.vocab.lookup_indices(data)), label
  @staticmethod
  def collate_fun(batch):
    X = [x for x,_ in batch]
    y = [y for _,y in batch]
    X = pad_sequence(X,batch_first=True)
    return X,torch.LongTensor(y)
 
```


```
train_dataset = MyDataset(train_files)
test_dataset = MyDataset(test_files)
train_dataloader = DataLoader(dataset=train_dataset, batch_size=64,shuffle=True,collate_fn=MyDataset.collate_fun,num_workers=2)
test_dataloader = DataLoader(dataset=test_dataset, batch_size=64,shuffle=True,collate_fn=MyDataset.collate_fun, num_workers=2)
```

## Model  and Model Settings

```
VOC_SIZE = len(vocab)
EMBED_SIZE = 300
HIDDEN_SIZE = 128
NUM_LAYER = 2

class SentimentClassifier(nn.Module):

  def __init__(self) -> None:
    super(SentimentClassifier,self).__init__()
    self.emblayer = nn.Embedding(VOC_SIZE,EMBED_SIZE)
    self.lstmlayer = nn.LSTM(EMBED_SIZE, HIDDEN_SIZE,batch_first=True)
    self.linear1 = nn.Linear(HIDDEN_SIZE, 32)
    self.relu = nn.ReLU()
    self.linear2 = nn.Linear(32, 2)

  def forward(self, x):
    
    x = self.emblayer(x)
    x, x_len = self.lstmlayer(x)
    x = x[:,-1,:]
    x = self.linear1(x)
    x = self.relu(x)
    x = self.linear2(x)

    return x

```

## Training and Training settings


```
torch.manual_seed(42)
model = SentimentClassifier().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)

epoch = 10
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

for i in range(epoch):
    model.train()
    train_acc = []
    test_acc = []
    for j, (features,labels) in enumerate(train_dataloader):
        t = []
        l = []
        optimizer.zero_grad()
        features = features.cuda()
        labels = labels.cuda()
        logits = model(features)
        # print(getAccuracy(logits,labels))
        loss = criterion(logits,labels)
        loss.backward()
        optimizer.step()
        if (j+1) % 100 == 0:
              print("epoch:{}/{}".format(i+1,epoch,j+1,))
        acc = getAccuracy(logits,labels)
        train_acc.append(acc)
        t.append(loss.item())
    av_a_t = sum(train_acc)/len(train_acc)
    train_accuracies.append(av_a_t)
    av_t = sum(t)/len(t)
    print("epoch:{}/{},Train loss:{}, Training Accuracy:{}".format(i+1,epoch,av_t,av_a_t))
    train_losses.append(av_t)
    model.eval()
    for j, (features,labels) in enumerate(test_dataloader):
        with torch.no_grad(): 
            features = features.cuda()
            labels = labels.cuda()
            logits = model(features)
            loss = criterion(logits,labels)
            l.append(loss.item())
            l.append(loss.item())
            acc = getAccuracy(logits,labels)
            test_acc.append(acc)
    av_a_l = sum(test_acc)/len(test_acc)
    av_l = sum(l)/len(l)
    print("epoch:{}/{},Test loss:{}, Validation Accuracy:{}".format(i+1,epoch,av_l,av_a_l))
    test_accuracies.append(av_a_l)


```
