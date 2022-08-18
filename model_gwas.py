import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()
        # self.L = 500
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            # nn.Conv2d(20, 50, kernel_size=1),
            
            #mlp as extrator
            # nn.Linear(3, 10),# note: change to mlp now for the encoding part
            nn.Linear(3, 10),
            # nn.ReLU(),
            nn.LeakyReLU(),#here changed to leakeyReLU
            # nn.MaxPool1d(2, stride=2)
            nn.MaxPool1d(2, stride=2),

            # nn.Linear(5, 20),
            nn.Linear(5, 20),
            # nn.ReLU(),
            nn.LeakyReLU(),#here changed to leakeyReLU
            # nn.MaxPool1d(2, stride=2)
            nn.MaxPool1d(2, stride=2)

            #bigger setting
            # nn.Linear(3, 30),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(2, stride=2),
            # nn.Linear(15, 60),
            # nn.LeakyReLU(),
            # nn.MaxPool1d(2, stride=2)     



            # conv1d trial
            # nn.Conv1d(1, 10, kernel_size=1),
            # nn.ReLU(),
            # nn.MaxPool1d(2, stride=1),
            # nn.Conv1d(10, 30, kernel_size=1),
            # nn.ReLU(),
            # nn.MaxPool1d(2, stride=1)
        )

        self.feature_extractor_part2 = nn.Sequential(
            # nn.Linear(30, self.L),
            nn.Linear(10, self.L),
            # nn.Linear(10, self.L),


            # nn.ReLU(),
            nn.LeakyReLU(),#here changed to leakeyReLU
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        # print(x.shape)
        x = x.squeeze(0)
        # print(x.shape)
        # x = x.type(torch.FloatTensor)#added type change
        H = self.feature_extractor_part1(x)
        print(H.shape)

        #small mlp
        # H = H.view(-1, 8)

        #cnn
        # H = H.view(-1, 30)

        #mlp
        H = H.view(-1, 10)
        # print(H.shape)

        H = self.feature_extractor_part2(H)  # NxL
        # print(H.shape)

        A = self.attention(H)  # NxK
        # print(A.shape)
        A = torch.transpose(A, 1, 0)  # KxN

        # print(A.shape)
        A = F.softmax(A, dim=1)  # softmax over N
        # print(A.shape)

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().data.item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)

       
        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli

        return neg_log_likelihood, A

class GatedAttention(nn.Module):
    def __init__(self):
        super(GatedAttention, self).__init__()
        self.L = 500
        self.D = 128
        self.K = 1

        self.feature_extractor_part1 = nn.Sequential(
            # nn.Conv2d(1, 20, kernel_size=5),
            # nn.ReLU(),
            # nn.MaxPool2d(2, stride=2),
            # nn.Conv2d(20, 50, kernel_size=5),
            # nn.ReLU(),
            # nn.MaxPool2d(2, stride=2)

            nn.Linear(3, 10),# note: change to mlp now for the encoding part
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2),
            nn.Linear(5, 20),
            nn.ReLU(),
            nn.MaxPool1d(2, stride=2)
        )

        self.feature_extractor_part2 = nn.Sequential(
            # nn.Linear(50 * 4 * 4, self.L),
            # nn.ReLU(),
            nn.Linear(10, self.L),
            nn.ReLU(),
        )

        self.attention_V = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh()
        )

        self.attention_U = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Sigmoid()
        )

        self.attention_weights = nn.Linear(self.D, self.K)

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.squeeze(0)
        # x = x.type(torch.FloatTensor)#added type change

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 10)
        # H = H.view(-1, 50 * 4 * 4)
        H = self.feature_extractor_part2(H)  # NxL

        A_V = self.attention_V(H)  # NxD
        A_U = self.attention_U(H)  # NxD
        A = self.attention_weights(A_V * A_U) # element wise multiplication # NxK
        A = torch.transpose(A, 1, 0)  # KxN
        A = F.softmax(A, dim=1)  # softmax over N

        M = torch.mm(A, H)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, A

    # AUXILIARY METHODS
    def calculate_classification_error(self, X, Y):
        Y = Y.float()
        _, Y_hat, _ = self.forward(X)
        error = 1. - Y_hat.eq(Y).cpu().float().mean().item()

        return error, Y_hat

    def calculate_objective(self, X, Y):
        Y = Y.float()
        Y_prob, _, A = self.forward(X)
        Y_prob = torch.clamp(Y_prob, min=1e-5, max=1. - 1e-5)

        neg_log_likelihood = -1. * (Y * torch.log(Y_prob) + (1. - Y) * torch.log(1. - Y_prob))  # negative log bernoulli 


        return neg_log_likelihood, A
