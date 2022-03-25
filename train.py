#import all the needed functions
import os.path
import argparse
import torch
import torchvision
from utils import *
import torchvision.transforms as transforms
from torch import optim
import matplotlib.pyplot as plt
from models import *
from Flows import *
from torch.distributions import MultivariateNormal
from torch.backends import cudnn
cudnn.benchmark = True
device = 'cuda'
cond_dim = 10
vector_dim = 784
K = 5
L = 1
save_rate = 500  # save every save_rate batches
batch_size = 500
directory = "./results"


def save_plot(name, plot_list):
    plt.plot(list(range(len(plot_list))), plot_list)
    plt.savefig(name)
    plt.clf()


def save(my_directory, current_action, targets, batch, epoch):
    torch.save(model.state_dict(), "./" + my_directory + "/vanillaFlowerParams.pth")
    # Save a plot for total loss
    save_plot("./" + my_directory + "/total_loss_list_" + current_action + ".png", total_loss_list[current_action])
    # Save a plot for logprob
    save_plot("./" + my_directory + "/logprob_list_" + current_action + ".png", logprob_list[current_action])
    # Save a plot for prior logprob
    save_plot("./" + my_directory + "/prior_logprob_list_" + current_action + ".png", prior_logprob_list[current_action])
    # Save a plot for log det
    save_plot("./" + my_directory + "/log_det_list_" + current_action + ".png", log_det_list[current_action])
    # Save a plot for the mean
    save_plot("./" + my_directory + "/mean_" + current_action + ".png", mean_list[current_action])
    # Save a plot for std
    save_plot("./" + my_directory + "/std_" + current_action + ".png", std_list[current_action])
    # Save a plot for bijectiviness
    save_plot("./" + my_directory + "/bijectiviness_" + current_action + ".png", bijectiviness_list[current_action])
    # Save current sample images
    save_sample_images("./" + my_directory + "/sample_images/image", targets, batch, epoch)
    print("Results and model saved successfully")


def save_sample_images(name, targets, batch, epoch, nb_images=3):
    generated_images = model.sample(nb_images, targets[:nb_images])
    for i, img in enumerate(generated_images):
        plt.subplot(1, 2, 1)
        plt.imshow(img[0].reshape(28, 28).to('cpu').detach().numpy())
        plt.savefig(name + "_epoch" + str(epoch) + "_batch" + str(batch) + "_image_ " + str(i) + ".png")
        plt.clf()


def train():
    # model.load_state_dict(torch.load("./remoteRes/results/vanillaFlowerParams.pth"))

    total_loss_average = 0
    logprob_average = 0
    prior_logprob_average = 0
    log_det_average = 0
    mean_average = 0
    std_average = 0
    bijevtiviness_average = 0
    try:
        for epoch in range(10000000):
            targets = []
            for batch, (images, targets) in enumerate(train_loader):
                optimizer.zero_grad()

                images = images.to(device).reshape(images.shape[0], 1, 784)
                targets = one_hot_MNIST(targets)
                targets = targets.to(device)[:, None, :]
                z, prior_logprob, log_det = model.forward(images, targets)
                logprob = prior_logprob + log_det
                loss = -torch.mean(prior_logprob + log_det)
                z, _ = model.inverse(z, targets)
                loss.backward()
                optimizer.step()

                # Convert the error values to float so it can be displayed and plotted
                loss = loss.item()
                logprob = logprob.mean().item()
                prior_logprob = prior_logprob.mean().item()
                log_det = log_det.mean().item()
                mean = torch.mean(z).item()
                std = torch.std(z).item()
                bijectiviness = mse_loss(z, images).item()

                # Add the errors to the average, we will divide by the total of samples afterwards
                total_loss_average += loss
                logprob_average += logprob
                prior_logprob_average += prior_logprob
                log_det_average += log_det
                mean_average += mean
                std_average += std
                bijevtiviness_average += bijectiviness

                if batch % 50 == 0:
                    print(f"Step: {epoch}\t" +
                          f"Iter: {batch}\t" +
                          f"Loss: {loss:.4f}\t" +
                          f"Logprob: {logprob:.2f}\t" +
                          f"Prior: {prior_logprob:.2f}\t" +
                          f"LogDet: {log_det:.2f}\t" +
                          f"Mean : {mean:.2f} \t" +
                          f"Std : {std:.2f} \t " +
                          f"Bijectiviness : {bijectiviness:.2f} \t"
                          )

                if (batch + epoch*len(train_loader)) % save_rate == 0:
                    total_loss_average /= save_rate
                    logprob_average /= save_rate
                    prior_logprob_average /= save_rate
                    log_det_average /= save_rate
                    mean_average /= save_rate
                    std_average /= save_rate
                    bijevtiviness_average /= save_rate

                    total_loss_list["train"].append(total_loss_average)
                    logprob_list["train"].append(logprob_average)
                    prior_logprob_list["train"].append(prior_logprob_average)
                    log_det_list["train"].append(log_det_average)
                    mean_list["train"].append(mean_average)
                    std_list["train"].append(std_average)
                    bijectiviness_list["train"].append(bijevtiviness_average)

                    save(directory, "train", targets, batch, epoch)
    except KeyboardInterrupt:
        save(directory, "train", targets, batch, epoch)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Trainer for the InvertedDrumGAN model")
    parser.add_argument("destination_directory", type=str, default="results",
                        help="Directory in which the results of the experiment will be saved")
    parser.add_argument("layer_to_test", type=str,
                        help="Directory in which the results of the experiment will be saved")
    args = parser.parse_args()
    directory = args.destination_directory
    # import the MNIST dataset from pytorch
    root = os.path.abspath(r"C:\Users\ithan\Documents\PC3\Machine_Learning\Practical part\data")
    # root = "./data"
    train_dataset = torchvision.datasets.MNIST(root=root, train=True, transform=transforms.ToTensor(), download=False)
    test_dataset = torchvision.datasets.MNIST(root=root, train=False, transform=transforms.ToTensor(), download=False)
    train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = torch.utils.data.DataLoader(dataset=test_dataset,  batch_size=100, shuffle=False)

    prior = MultivariateNormal(torch.zeros(784).to(device), torch.eye(784).to(device))
    if args.layer_to_test == "ActNorm":
        model = NormalizingFlowModelTester(prior, vector_dim, cond_dim, ActNorm(vector_dim))
    elif args.layer_to_test == "11Conv":
        model = NormalizingFlowModelTester(prior, vector_dim, cond_dim, OneByOneConv(vector_dim))
    elif args.layer_to_test == "CondC":
        model = NormalizingFlowModelTester(prior, vector_dim, cond_dim, ConditionalCoupling(vector_dim, cond_dim, False))
    else:
        raise Exception("No correct choice")

    # model = NormalizingFlowModel(prior, vector_dim, cond_dim, K, L)
    t = torch.randint(20, (10, 1, 20))
    squeezer = Squeeze()
    p = squeezer.forward(t, reverse=False)
    s = squeezer.forward(p, reverse=True)
    model = model.to(device)
    mse_loss = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Lists used for plotting the error and loss of the model
    total_loss_list = {"train": [], "validation": []}
    logprob_list = {"train": [], "validation": []}
    prior_logprob_list = {"train": [], "validation": []}
    log_det_list = {"train": [], "validation": []}
    mean_list = {"train": [], "validation": []}
    std_list = {"train": [], "validation": []}
    bijectiviness_list = {"train": [], "validation": []}

    train()

