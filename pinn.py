import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt

###############################
# 1. 数据生成（利用 BB 方法）
###############################

def simulate_particle_system_BB(n_particles, d, outer_iter, inner_iter, tau, h, device='cpu'):
    """
    利用 BB 方法生成粒子系统数据（位置和速度轨迹）
    
    参数：
      n_particles -- 粒子数
      d           -- 空间维数
      outer_iter  -- 外循环步数（时间步数）
      inner_iter  -- 每个时间步内循环迭代次数
      tau         -- 时间尺度参数（用于计算速度与残差）
      h           -- 高斯核参数（用于计算参考量）
      device      -- 设备（'cpu' 或 'cuda'）
      
    返回：
      X_traj -- 形状 (outer_iter, n_particles, d) 的位置轨迹
      U_traj -- 形状 (outer_iter, n_particles, d) 的速度轨迹
    """
    # 随机初始化位置 x 和初始参考速度 u_old
    x = torch.randn(n_particles, d, device=device)
    x_initial = x.clone()
    u_old = torch.randn(n_particles, d, device=device)
    
    # 用于存储轨迹
    X_traj = torch.zeros(outer_iter, n_particles, d, device=device)
    U_traj = torch.zeros(outer_iter, n_particles, d, device=device)
    X_traj[0] = x.clone()
    # 初始速度定义为： u = (x - x_initial)/tau
    u = (x - x_initial) / tau
    U_traj[0] = u.clone()
    
    
    grad_old = None
    x_old = None

    for i in range(1, outer_iter):
        # 内循环：演化粒子系统
        for j in range(inner_iter):
            # 计算两两粒子之间的差异和距离
            diff = x.unsqueeze(1) - x.unsqueeze(0)    # shape: (n_particles, n_particles, d)
            r = torch.sqrt(torch.sum(diff ** 2, dim=-1) + 1e-12)  # (n_particles, n_particles)
            # 填充对角线，避免自作用
            r.fill_diagonal_(1e-12)
            
            
            kxy = torch.exp(-torch.sum(diff ** 2, dim=-1) / (2 * h ** 2))
            norm_factor = (torch.pi * 2.0 * h * h) ** (d / 2)
            kxy = kxy / norm_factor

            # 固定势函数：1 / ((r^2 + 1)^0.01)
            Phi_xy_fixed = 1.0 / torch.pow(torch.sum(diff ** 2, dim=-1) + 1, 0.5)

            # 计算当前速度（位移/tau）
            u = (x - x_initial) / tau
            diff_u = u.unsqueeze(1) - u.unsqueeze(0)

            # 计算粒子位置更新项 Akxy
            Akxy = (torch.sum(-Phi_xy_fixed.unsqueeze(-1) * diff_u, dim=1) / n_particles
                    - (u - u_old) / tau)
            grad_now = Akxy.view(1, -1)  # 将梯度展平为 (1, N*d)

            if torch.norm(grad_now) < 1e-10:
                print(f"内循环在外循环 {i} 的第 {j} 次迭代提前停止")
                break

            # 使用 Barzilai–Borwein 规则自适应步长
            step_l = 1e-7
            if j > 30 and grad_old is not None:
                y_k = grad_now - grad_old
                s_k = x.view(1, -1) - x_old.view(1, -1)
                denom = torch.dot(s_k.view(-1), y_k.view(-1))
                if torch.abs(denom) > 1e-12:
                    step_l = torch.dot(s_k.view(-1), s_k.view(-1)) / denom

            grad_old = grad_now.clone()
            x_old = x.clone()
            # 更新粒子位置
            x = x - step_l * Akxy

        # 内循环结束，更新速度并存储当前时刻数据
        u_old = (x - x_initial) / tau
        X_traj[i] = x.clone()
        U_traj[i] = u_old.clone()
        x_initial = x.clone()
       
    
    return X_traj, U_traj

def generate_simulation_data():
    device = 'cpu'  
    n_particles = 50
    d = 2
    outer_iter = 100     # 时间步数
    inner_iter = 200    # 每步内循环次数
    tau = 0.1
    h = 0.2
    X_traj, U_traj = simulate_particle_system_BB(n_particles, d, outer_iter, inner_iter, tau, h, device)
    return X_traj, U_traj

########################################
# 2. 利用生成的数据进行 PINN 训练
########################################

# 定义用于拟合相互作用函数 psi 的神经网络
class PsiNet(nn.Module):
    def __init__(self, input_dim=1, hidden_dim=256, output_dim=1):
        super(PsiNet, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, output_dim),
            nn.Softplus()  
        )
    
    def forward(self, r):
        return self.net(r)

def train_PINN(X_traj, U_traj, n_particles, tau, num_epochs=1000, lr=1e-2, device='cpu'):
    """
    利用生成的轨迹数据（X_traj, U_traj）训练 PINN，拟合 ODE 系统：
         dx/dt = u,
         du/dt = - (1/N) sum_j psi(||x_i-x_j||) (u_i - u_j)
    """
    psi_net = PsiNet(input_dim=1, hidden_dim=256, output_dim=1).to(device)
    optimizer = optim.Adam(psi_net.parameters(), lr=lr)
    
    T = X_traj.shape[0] - 1  # 时间间隔数
    loss_history = []
    
    for epoch in range(num_epochs):
        optimizer.zero_grad()
        loss_total = 0.0
        
        # 对于每个时间间隔 t=0,...,T-1 计算残差
        for t in range(T):
            x_t = X_traj[t]       # shape: (n_particles, d)
            x_tp1 = X_traj[t+1]
            u_t = U_traj[t]       # shape: (n_particles, d)
            u_tp1 = U_traj[t+1]
            
            # 利用有限差分近似计算导数
            dx_dt_true = (x_tp1 - x_t) / tau   # 位置导数
            du_dt_true = (u_tp1 - u_t) / tau     # 速度导数
            
            # 位置方程残差：dx/dt 应该等于 u
            res_x = dx_dt_true - u_t
            
            # 速度方程残差：
            # 计算两两粒子间距离及其差
            diff = x_t.unsqueeze(1) - x_t.unsqueeze(0)  # (N, N, d)
            r = torch.sqrt(torch.sum(diff**2, dim=-1) + 1e-12)  # (N, N)
            # 将距离 r 整理为 (N*N, 1) 输入到网络中
            r_input = r.view(-1, 1)
            psi_pred = psi_net(r_input).view(n_particles, n_particles)  # (N, N)
            
            # 计算速度差
            diff_u = u_t.unsqueeze(1) - u_t.unsqueeze(0)  # (N, N, d)
            du_dt_pred = - (1.0 / n_particles) * torch.sum(psi_pred.unsqueeze(-1) * diff_u, dim=1)  # (N, d)
            
            res_u = du_dt_true - du_dt_pred
            
            loss_t = torch.mean(res_x**2) + torch.mean(res_u**2)
            loss_total += loss_t
        
        loss_total = loss_total / T
        loss_total.backward()
        optimizer.step()
        loss_history.append(loss_total.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss_total.item():.6f}")
    
    return psi_net, loss_history

###############################
# 主函数：生成数据并进行 PINN 训练
###############################
if __name__ == '__main__':
    device = 'cpu'
    # 生成粒子系统轨迹数据（真实数据）
    X_traj, U_traj = generate_simulation_data()
    print("数据生成完成。")
    print("X_traj shape:", X_traj.shape)  # (outer_iter, n_particles, d)
    print("U_traj shape:", U_traj.shape)  # 同上
    
    # 利用生成的数据训练 PINN 得到 psi_net
    n_particles = 50
    tau = 0.1
    psi_net, loss_history = train_PINN(X_traj, U_traj, n_particles, tau, num_epochs=1000, lr=1e-3, device=device)
    
    # 绘制训练损失曲线
    plt.figure(figsize=(8,5))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("PINN Training Loss")
    plt.grid(True)
    plt.show()
    
    # 对比学习到的 psi 与真实的 psi
    # 此处真实 psi 定义与数据生成时使用的固定势函数相同，即 psi_true(r)=1/((r^2+1)^0.01)
    def psi_true(r):
        return 1.0 / torch.pow(r**2 + 1, 0.5)
    
    r_values = torch.linspace(0.1, 10, 500, device=device).unsqueeze(1)  # (500, 1)
    psi_learned = psi_net(r_values).detach().cpu().numpy().squeeze()
    psi_true_vals = psi_true(r_values).detach().cpu().numpy().squeeze()
    
    plt.figure(figsize=(8,5))
    plt.plot(r_values.cpu().numpy(), psi_learned, label="Learned ψ", color='blue')
    plt.plot(r_values.cpu().numpy(), psi_true_vals, label="True ψ", color='red', linestyle='--')
    plt.xlabel("r")
    plt.ylabel("ψ(r)")
    plt.title("Learned ψ vs True ψ")
    plt.legend()
    plt.grid(True)
    plt.show()
