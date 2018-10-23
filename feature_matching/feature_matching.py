import torch
from torch import nn

# All the rights
# https://sudomake.ai/feature-matching-generative-adversarial-networks/


class Discriminator(nn.Module):

    def __init__(self, input_size, num_features):
        super().__init__()

        self.features = nn.Linear(input_size, num_features)
        self.classifier = nn.Sequential(
            nn.Linear(num_features, 2), nn.Sigmoid())

    def forward(self, x):
        # Instead of performing the classification, we return both outputs and
        # the features
        f = self.features(x)
        cls = self.classifier(f)

        return cls, f


# all nets, variables and optimizers are initialized as usual
D = ...
G = ...
x, y = ...


feature_matching_criterion = nn.MSELoss()

fake_samples = G(noise)  # perform sampling from generator
real_samples = ...  # perform sampling from real data
fake_pred, fake_feats = D(fake_samples)
real_pred, real_feats = D(real_samples)

real_feats = real_feats.detach()  # so that PyTorch will treat them as volatile

# now, calculating the new objective function
fm_loss = feature_matching_criterion(fake_feats, real_feats)
fm_loss.backward()
