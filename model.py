import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class RA_MIL(nn.Module):
    def __init__(self):
        super(RA_MIL, self).__init__()
        self.L = 1000 # 500, #1000 & 500
        self.D = 500 # 128
        self.K = 1

        # resnet50
        self.feature_extractor_part1 = torch.nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])

        self.feature_extractor_part2 = nn.Sequential(
            nn.Linear(2048 * 7 * 7, self.L), #  to 512
            nn.ReLU(),
            nn.Dropout(p=0.5)
        )

        self.rank_part = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.K)
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.L*self.K, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.squeeze(0) # 删除第0维

        H = self.feature_extractor_part1(x)
        H = H.view(-1, 2048 * 7 * 7)
        H = self.feature_extractor_part2(H)  # 20*1000
        posib = self.rank_part(H)  # 2 or 1
        posib = torch.transpose(posib, 1, 0)  # use attention score to rank
        posib = F.softmax(posib, dim=1) #1
        rank = torch.sort(posib,descending=True) # # rank attention score

        feature_list = []
        A_list = []
        for i in range(10):
            feature_list.append(H[rank[1][0].tolist()[i]]) #rank attention score
            A_list.append(posib[0][rank[1][0].tolist()[i]])
        H_top = torch.stack(feature_list)
        A = torch.stack(A_list)
        A = A.unsqueeze(0)

        M = torch.mm(A, H_top)  # KxL

        Y_prob = self.classifier(M)
        Y_hat = torch.ge(Y_prob, 0.5).float()

        return Y_prob, Y_hat, posib, A # bag_prob,predict_t,ins_prob,rank_ins_prob