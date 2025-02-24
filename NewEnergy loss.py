import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

###############################
# 1. 数据生成（利用 BB 方法）
###############################

def simulate_particle_system_BB(n_particles, d, outer_iter, inner_iter, tau, h, device='cpu'):
    # 随机初始化位置和速度
    x = torch.randn(n_particles, d, device=device)
    x_initial = x.clone()
    u_old = torch.randn(n_particles, d, device=device)
    
    # 存储轨迹
    X_traj = torch.zeros(outer_iter, n_particles, d, device=device)
    U_traj = torch.zeros(outer_iter, n_particles, d, device=device)
    X_traj[0] = x.clone()
    U_traj[0] = (x - x_initial) / tau
    
    grad_old = None
    x_old = None

    for i in range(1, outer_iter):
        for j in range(inner_iter):
            # 计算粒子间相互作用
            diff = x.unsqueeze(1) - x.unsqueeze(0)
            r = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-12)
            r.fill_diagonal_(1e-12)
            
            # 固定势函数
            Phi = 1.0 / torch.pow(torch.sum(diff ** 2, dim=-1) + 1, 0.5)
            
            # 计算速度场
            u = (x - x_initial) / tau
            diff_u = u.unsqueeze(1) - u.unsqueeze(0)
            
            # 更新项计算
            Akxy = (
                torch.sum(-Phi.unsqueeze(-1) * diff_u, dim=1) / n_particles 
                - (u - u_old) / tau
            )
            grad_now = Akxy.view(1, -1)

            # BB步长调整
            step_l = 1e-7
            if j > 30 and grad_old is not None:
                y_k = grad_now - grad_old
                s_k = x.view(1, -1) - x_old.view(1, -1)
                denom = torch.dot(s_k.view(-1), y_k.view(-1))
                if torch.abs(denom) > 1e-12:
                    step_l = torch.dot(s_k.view(-1), s_k.view(-1)) / denom
                    
            # 更新位置
            grad_old = grad_now.clone()
            x_old = x.clone()
            x = x - step_l * Akxy

        # 存储当前状态
        u_new = (x - x_initial) / tau
        X_traj[i] = x.clone()
        U_traj[i] = u_new.clone()
        x_initial = x.clone()
        u_old = u_new
    
    return X_traj, U_traj

def generate_simulation_data():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    return simulate_particle_system_BB(
        n_particles=50, d=2, outer_iter=1000, inner_iter=500,
        tau=0.1, h=0.2, device=device
    )

########################################
# 2. 改进的神经网络和训练流程
########################################

class PsiNet(nn.Module):
    def __init__(self, h=0.2):
        super().__init__()
        self.h = h  
        self.net = nn.Sequential(
            nn.Linear(2, 256),  
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 1),
            nn.Softplus()  
        )
        
    def forward(self, r):
        r_clamped = torch.clamp(r, 1e-3, 10.0)
        # 特征工程
        features = torch.cat([
            r_clamped, 
            1/(r_clamped**2 + self.h**2)  # 使用self.h
        ], dim=-1)
        return self.net(features).squeeze()


def train_PINN(X_traj, U_traj, n_particles, tau, num_epochs=1000, lr=1e-3, device='cpu'):
    # 初始化模型
    psi_net = PsiNet(h=0.2).to(device)  
    optimizer = optim.AdamW(psi_net.parameters(), lr=lr, weight_decay=1e-5)  # 使用实例参数
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs//2)
    
    eye_mask = ~torch.eye(n_particles, dtype=torch.bool, device=device)
    loss_history = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        total_loss = 0.0
        
        # 批量处理时间步
        time_steps = torch.randint(0, len(X_traj)-1, (32,))
        
        for t in time_steps:
            x_t = X_traj[t]
            u_t = U_traj[t]
            u_tp1 = U_traj[t+1]
            
            # Energy
            E_t = 0.5 * torch.sum(u_t**2)
            E_tp1 = 0.5 * torch.sum(u_tp1**2)
            dE_dt = (E_tp1 - E_t) / tau
            
            # 相互作用计算
            diff_x = x_t.unsqueeze(1) - x_t.unsqueeze(0)
            r = torch.sqrt(torch.sum(diff_x**2, dim=-1) + 1e-12)
            
            psi = psi_net(r[eye_mask].unsqueeze(1))
            diff_u = u_t.unsqueeze(1) - u_t.unsqueeze(0)
            v_sq = torch.sum(diff_u**2, dim=-1)[eye_mask]
            
            interaction = torch.sum(psi * v_sq) / (n_particles)
            
            # 损失计算
            loss = (dE_dt + interaction)**2
            total_loss += loss
        
        
        total_loss = total_loss/len(time_steps) + 0.001*sum(p.norm(2) for p in psi_net.parameters())
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(psi_net.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        loss_history.append(total_loss.item())
        
        if epoch % 100 == 0:
            avg_loss = np.mean(loss_history[-100:]) if epoch>100 else total_loss.item()
            print(f"Epoch {epoch:04d} | Loss: {avg_loss:.2e} | LR: {scheduler.get_last_lr()[0]:.1e}")
    
    return psi_net, loss_history

###############################
# 3. 主程序
###############################
if __name__ == "__main__":
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # 生成数据
    X_traj, U_traj = generate_simulation_data()
    print(f"Data generated: X_traj.shape={X_traj.shape}, U_traj.shape={U_traj.shape}")
    
    # 训练模型
    psi_net, losses = train_PINN(
        X_traj, U_traj,
        n_particles=50,
        tau=0.1,
        num_epochs=500,
        lr=1e-4,
        device=device
    )
    
     # 可视化
    plt.figure(figsize=(12,5))
    
    # 损失曲线
    plt.subplot(121)
    plt.semilogy(losses)
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True, alpha=0.3)
    
    # 函数对比
    plt.subplot(122)
    def psi_true(r):
        return 1.0 / torch.pow(r**2 + 1, 0.5)
    
    r_test = torch.linspace(0.1, 5, 100, device=device).unsqueeze(1)
    with torch.no_grad():
        psi_pred = psi_net(r_test).cpu().numpy()
        psi_true_vals = psi_true(r_test).cpu().numpy()
    
    plt.plot(r_test.cpu(), psi_pred, label="Learned")
    plt.plot(r_test.cpu(), psi_true_vals, 'r--', label="True")
    plt.title("Learned ψ vs True ψ")
    plt.xlabel("r")
    plt.ylabel("ψ(r)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()