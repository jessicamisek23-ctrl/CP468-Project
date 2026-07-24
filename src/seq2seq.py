import torch.nn as nn

class Seq2Seq(nn.Module):

    def __init__(
        self,
        encoder,
        decoder,
        attention
    ):

        super().__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.attention = attention

train:
import torch
import torch.optim as optim

optimizer = optim.Adam(
    model.parameters(),
    lr=0.001
)

criterion = torch.nn.CrossEntropyLoss()

EPOCHS = 20

for epoch in range(EPOCHS):

    model.train()

    epoch_loss = 0

    for src, tgt in train_loader:

        optimizer.zero_grad()

        output = model(src, tgt)

        loss = criterion(
            output.reshape(-1, output.shape[-1]),
            tgt.reshape(-1)
        )

        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()

    print(epoch_loss)
