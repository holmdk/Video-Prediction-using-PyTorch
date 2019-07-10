import torch
from torch.autograd import Variable
from torchvision import transforms
from Models import ConvLSTM
from MovingMNIST import MovingMNIST

import numpy as np
import matplotlib.pyplot as plt

if __name__ == '__main__':
    EPOCHS = 1
    FILTERS = 64
    initial_runs = 500
    sample_interval = 300
    nt = 8
    batch_size = 4

    # Define dataset
    train_set = MovingMNIST(root='data/mnist', train=True, nt=nt, download=True)

    train_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=batch_size,
        shuffle=True) # maybe only shift this one? and then create a test data loader

    test_set = MovingMNIST(root='data/mnist', train=False, nt=nt, download=True)

    test_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=batch_size,
        shuffle=True) # maybe only shift this one? and then create a test data loader

    test_iter = iter(test_loader)

    # initialize model and optimizer
    model = ConvLSTM(nf=FILTERS).cuda()

    base_lr = 3e-4
    optimizer = torch.optim.Adam(model.parameters(), lr=base_lr, betas=(0.9, 0.999))  # 0.5, 0.9

    # define loss function
    criterion = torch.nn.MSELoss().cuda()  # consider using MSE instead!!

    # TRAINING
    batches_done = 0
    for epoch in range(EPOCHS):
        for i, (imgs_X, imgs_Y) in enumerate(train_loader):
            # Initialize variables to be used
            input = Variable(imgs_X.unsqueeze(2).cuda())  # .half()) # .half()
            target = Variable(imgs_Y.unsqueeze(2).cuda())

            # clear previous gradients
            optimizer.zero_grad()

             # multiple steps-ahead sequence prediction
            out = model(input, future=nt)
            loss = criterion(out[:, :, -nt:, :, :], target[:, -nt:, :, :, :].permute(0, 2, 1, 3, 4))
            loss.backward()

            # load random test set
            x_test, y_test = test_iter.next()

            with torch.no_grad():
                out_test = model(Variable(x_test.unsqueeze(2).cuda()))

                loss_test = criterion(out_test[:, :, -nt:, :, :], Variable(y_test.unsqueeze(2).cuda()[:, -nt:, :, :, :].permute(0, 2, 1, 3, 4)))

            print(
                "[Epoch %d/%d] [Batch %d/%d] [Training loss: %f] [Test loss: %f]"
                % (epoch + 1, EPOCHS, i, len(train_loader), loss.item(), loss_test.item())
            )

            # update using calculated gradients
            optimizer.step()

            # save output images every sampling interval
            # if batches_done % sample_interval == 0 and batches_done > initial_runs:
            #     with torch.no_grad():
            #         out = model(input, future=15)

            batches_done = epoch * len(train_loader) + i


torch.save(model.state_dict(), './convLSTM_state_dict.pth')