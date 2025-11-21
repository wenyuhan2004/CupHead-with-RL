import torch
import torch.nn as nn
import torch.optim as optim


class DQNAgent:
    def __init__(self, q_net: nn.Module, target_net: nn.Module, lr=1e-4, gamma=0.99, device="cpu"):
        self.q_net = q_net.to(device)
        self.target_net = target_net.to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()
        self.gamma = gamma
        self.device = device
        self.optim = optim.Adam(self.q_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()

    def select_action(self, obs, epsilon, n_actions):
        if torch.rand(1).item() < epsilon:
            return torch.randint(0, n_actions, (1,)).item()
        with torch.no_grad():
            q = self.q_net(obs.to(self.device))
            return int(q.argmax(dim=1).item())

    def learn(self, batch, n_actions):
        obs = torch.as_tensor(batch.obs, dtype=torch.float32, device=self.device)
        next_obs = torch.as_tensor(batch.next_obs, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.action, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.reward, dtype=torch.float32, device=self.device).unsqueeze(1)
        dones = torch.as_tensor(batch.done, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.q_net(obs).gather(1, actions)
        with torch.no_grad():
            next_q = self.target_net(next_obs).max(dim=1, keepdim=True)[0]
            target = rewards + self.gamma * (1 - dones) * next_q

        loss = self.loss_fn(q_values, target)
        self.optim.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 5.0)
        self.optim.step()
        return loss.item()

    def update_target(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
