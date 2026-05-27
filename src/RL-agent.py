import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Categorical
import matplotlib.pyplot as plt
import random

GLOBAL_SEED = 42
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)

class DualHeadNet(nn.Module):
    def __init__(self, in_features=3, hidden_sizes=[64, 32], out_features=2):
        super().__init__()

        self.hidden_layers = nn.ModuleList()
        current_dim = in_features

        for h_size in hidden_sizes:
            self.hidden_layers.append(nn.Linear(current_dim, h_size))
            current_dim = h_size

        self.actor_head = nn.Linear(current_dim, out_features)
        self.critic_head = nn.Linear(current_dim, 1)

    def forward(self, x):
        for layer in self.hidden_layers:
            x = torch.relu(layer(x))

        action_prob = torch.softmax(self.actor_head(x), dim=-1)
        state_val = self.critic_head(x)
        return action_prob, state_val

class ACAgent:
    def __init__(self, architecture, gamma_val=0.99, learn_rate=1e-3, ent_weight=0.01):
        self.gamma = gamma_val
        self.ent_weight = ent_weight
        self.device = torch.device("cpu")

        self.net = DualHeadNet(hidden_sizes=architecture).to(self.device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=learn_rate)
        self.memory = []

    def _normalize_state(self, state):
        return [state[0] / 32.0, state[1] / 10.0, float(state[2])]

    def select_action(self, raw_state, is_train=True):
        norm_state = self._normalize_state(raw_state)
        state_t = torch.tensor(norm_state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            probs, val = self.net(state_t)

        if is_train:
            m = Categorical(probs)
            chosen_action = m.sample().item()
            self.memory.append({
                "state": norm_state,
                "log_prob": m.log_prob(torch.tensor(chosen_action)),
                "value": val.squeeze(),
                "reward": 0
            })
        else:
            chosen_action = torch.argmax(probs).item()

        return chosen_action

    def save_reward(self, r):
        self.memory[-1]["reward"] = r

    def optimize_model(self):
        if not self.memory: return
        states = torch.tensor([step["state"] for step in self.memory], dtype=torch.float32).to(self.device)
        log_probs = torch.stack([step["log_prob"] for step in self.memory])
        values = torch.stack([step["value"] for step in self.memory])
        rewards = [step["reward"] for step in self.memory]

        n_steps = len(rewards)
        returns = np.zeros(n_steps, dtype=np.float32)
        cumulative = 0
        for t in reversed(range(n_steps)):
            cumulative = rewards[t] + self.gamma * cumulative
            returns[t] = cumulative

        returns_t = torch.tensor(returns).to(self.device)

        probs, _ = self.net(states)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        advantage = returns_t - values.detach()
        loss_policy = -(log_probs * advantage).mean()
        loss_value = nn.MSELoss()(values, returns_t)

        total_loss = loss_policy + loss_value - (self.ent_weight * entropy)

        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        self.optimizer.step()

        self.memory.clear()

def run_evaluation(agent_model, num_matches=100):
    test_env = gym.make('Blackjack-v1', natural=False, sab=False)
    stats = {'W': 0, 'L': 0, 'D': 0}
    logs = []

    for idx in range(num_matches):
        obs, _ = test_env.reset()
        is_done = False
        moves = []

        while not is_done:
            act = agent_model.select_action(obs, is_train=False)
            moves.append("ВЗЯТЬ" if act == 1 else "ХВАТИТ")
            obs, rew, term, trunc, _ = test_env.step(act)
            is_done = term or trunc

        if rew > 0:
            stats['W'] += 1
            res = "ПОБЕДА"
        elif rew < 0:
            stats['L'] += 1
            res = "ПОРАЖЕНИЕ"
        else:
            stats['D'] += 1
            res = "НИЧЬЯ"

        logs.append({'id': idx + 1, 'history': moves, 'outcome': res})

    test_env.close()
    return stats['W'], stats['L'], stats['D'], logs

def execute_training(agent, total_eps=2000, check_interval=400):
    train_env = gym.make('Blackjack-v1', natural=False, sab=False)
    v_history = []
    r_history = []

    for ep in range(1, total_eps + 1):
        obs, _ = train_env.reset()
        is_done = False
        ep_reward = 0
        ep_values = []
        agent.memory.clear()

        while not is_done:
            norm_obs = agent._normalize_state(obs)
            with torch.no_grad():
                _, v = agent.net(torch.tensor(norm_obs, dtype=torch.float32).unsqueeze(0))
            ep_values.append(v.item())

            act = agent.select_action(obs, is_train=True)
            obs, rew, term, trunc, _ = train_env.step(act)
            is_done = term or trunc

            agent.save_reward(rew)
            ep_reward += rew

        agent.optimize_model()
        v_history.append(np.mean(ep_values) if ep_values else 0)
        r_history.append(ep_reward)

        if ep % check_interval == 0:
            w, l, d, _ = run_evaluation(agent, num_matches=100)
            win_pct = w / 100 * 100
            print(f"  [Эпизод {ep:4d}] ВР: {win_pct:4.1f}% | V(s) старт: {v_history[-1]:6.3f}")

    train_env.close()
    return v_history

def draw_value_plot(v_data, winrate, cfg):
    plt.figure(figsize=(10, 5))

    smooth_window = min(100, len(v_data) // 20)
    if smooth_window > 1:
        smoothed = np.convolve(v_data, np.ones(smooth_window) / smooth_window, mode='valid')
        plt.plot(range(smooth_window - 1, len(v_data)), smoothed, color='#9400D3', lw=2, label=f'Сглаженный тренд')

    plt.plot(v_data, color='#DA70D6', alpha=0.3, lw=1, label='Сырые значения V(s)')

    title_text = (f"Динамика оценки состояний V(s)\n"
                  f"WinRate: {winrate:.1f}% | Архитектура: {cfg['arch']} | "
                  f"Gamma: {cfg['g']} | LR: {cfg['lr']}")

    plt.title(title_text, fontsize=11)
    plt.xlabel('Эпизоды обучения', fontsize=10)
    plt.ylabel('Ожидаемая награда V(s)', fontsize=10)
    plt.axhline(0, color='black', ls='--', alpha=0.4)
    plt.grid(ls=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    configs = [
        {'id': 'Base', 'arch': [64, 32], 'g': 0.99, 'lr': 0.001, 'ent': 0.1},
        {'id': 'Low Ent', 'arch': [64, 32], 'g': 0.99, 'lr': 0.001, 'ent': 0.01},
        {'id': 'Fast LR', 'arch': [64, 32], 'g': 0.99, 'lr': 0.01, 'ent': 0.1},
        {'id': 'Deep Net', 'arch': [128, 64], 'g': 0.99, 'lr': 0.001, 'ent': 0.05},
        {'id': 'Low Gamma', 'arch': [128, 64], 'g': 0.5, 'lr': 0.001, 'ent': 0.05},
        {'id': 'Wide Net', 'arch': [256, 128], 'g': 0.99, 'lr': 0.0005, 'ent': 0.05},
    ]

    leaderboard = []
    top_agent = None
    top_v_hist = None
    max_winrate = -1.0
    best_cfg = None

    for c in configs:
        print(f"\n Запуск конфига: [{c['id']}] (Слои={c['arch']}, LR={c['lr']}, Gamma={c['g']})")

        current_agent = ACAgent(architecture=c['arch'], gamma_val=c['g'], learn_rate=c['lr'], ent_weight=c['ent'])
        v_hist = execute_training(current_agent, total_eps=3000, check_interval=1000)

        w, l, d, _ = run_evaluation(current_agent, num_matches=2000)
        final_wr = (w / 2000) * 100

        print(f"Итог [{c['id']}]: Винрейт {final_wr:.1f}%")
        leaderboard.append({'name': c['id'], 'wr': final_wr})

        if final_wr > max_winrate:
            max_winrate = final_wr
            top_agent = current_agent
            top_v_hist = v_hist
            best_cfg = c

    print("Рейтинг моделей:")
    leaderboard.sort(key=lambda x: x['wr'], reverse=True)
    for pos, item in enumerate(leaderboard, 1):
        print(f"{pos}. Конфиг '{item['name']:<10}' -> {item['wr']:.1f}% побед")

    print("\nОтрисовка графика для лидера...")
    draw_value_plot(top_v_hist, max_winrate, best_cfg)

    print("Детальный лог 100 игр (Лучший агент)")

    fw, fl, fd, final_logs = run_evaluation(top_agent, num_matches=100)

    for game in final_logs:
        path = " -> ".join(game['history'])
        print(f"Раунд {game['id']:03d} | Статус: {game['outcome']:<9} | Шаги: {path}")

    print(f"Итоговая статистика 100 игр:")
    print(f"Выиграно:  {fw}")
    print(f"Проиграно: {fl}")
    print(f"Ничья:     {fd}")
    print(f"Процент побед: {fw}%")
