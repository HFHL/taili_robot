### 🤖 自研四足机器人：平地盲跑与站立 (Flat Ground Velocity Tracking) 核心工程规约

---

### 一、 观测与动作空间 (Observation & Action Space)

**1. 策略网络 (Actor) 观测空间 $O_t$：无特权本体感受器 (48 维)**
严格剥离线速度，依靠历史动作与本体感知进行隐式推断。所有分量按顺序扁平化拼接为 1D 张量，并在网络输入前执行限幅 (Clip to $[-5.0, 5.0]$)：

| 索引范围 | 变量名称 | 维度 | 物理含义 | 缩放处理 (obs_scales) |
| --- | --- | --- | --- | --- |
| `[0:3]` | `base_ang_vel` | 3 | 机体系角速度 (Roll, Pitch, Yaw) | $\times 0.25$ |
| `[3:6]` | `projected_gravity` | 3 | 重力向量在机体系的三维投影 | 无缩放 |
| `[6:18]` | `dof_pos_error` | 12 | 关节位置误差 ($q - q_{default}$) | $\times 1.0$ |
| `[18:30]` | `dof_vel` | 12 | 当前各关节角速度 | $\times 0.05$ |
| `[30:42]` | `history_actions` | 12 | 上一步 Actor 输出的原始动作 | 无缩放 |
| `[42:45]` | `commands` | 3 | 目标速度指令 ($v_x^{cmd}, v_y^{cmd}, \omega_z^{cmd}$) | $\times 2.0$ |
| `[45:48]` | `mob_commands` | 3 | MoB 指令 (步频 $f^{cmd}$, 高度 $h_z^{cmd}$, 相位 $\theta^{cmd}$) | 独立缩放 (视量纲而定) |

**2. 价值网络 (Critic) 观测空间 $O_t^{critic}$：特权信息 (51+ 维)**
包含 Actor 的全部 48 维输入，外加仿真器提供的特权信息：

* **特权状态:** `base_lin_vel` (机体真实线速度, 3维), 地面摩擦系数 (1维), 测量的质心偏移 (3维) 等。

**3. 动作映射管线 (5-Step Action Pipeline)**
Actor 输出 $a_{raw} \in \mathbb{R}^{12}$，经由以下管线转化为物理驱动器的目标角度 $q_{target}$：

1. **安全限幅:** $a = \text{clip}(a_{raw}, -\text{clip\_actions}, +\text{clip\_actions})$
2. **延迟模拟:** $a_{exec} = a_{t-1}$ (强制执行上一帧动作，应对 CAN 总线延迟)
3. **基准与缩放:** $q_{target} = q_{default} + (a_{exec} \times \text{action\_scale})$
4. **软限位保护:** $q_{target} = \text{clamp}(q_{target}, \text{soft\_limits})$ (硬限位 $\times\ 0.9$)
5. **底层 PD 执行:** $\tau = K_p(q_{target} - q) - K_d(\dot{q})$

---

### 二、 核心指令与采样规约 (Command Logic)

在环境 Reset 或达到最大超时步数时触发重采样，强制网络在“站立”与“不同步态移动”间切换。

```python
def resample_commands(env_ids):
    # 20% 环境分配为“绝对静止站立”任务
    stand_mask = torch.rand(len(env_ids)) < 0.2
    
    # 速度追踪指令域采样
    v_x_cmd = torch.empty(len(env_ids)).uniform_(-1.0, 2.0)
    v_y_cmd = torch.empty(len(env_ids)).uniform_(-0.5, 0.5)
    w_z_cmd = torch.empty(len(env_ids)).uniform_(-1.0, 1.0)
    
    # MoB 行为参数域采样
    freq_cmd = torch.empty(len(env_ids)).uniform_(1.5, 3.5) # 步频
    height_cmd = torch.empty(len(env_ids)).uniform_(0.25, 0.35) # 目标机身高度
    
    # 覆写站立任务的指令
    v_x_cmd[stand_mask] = 0.0
    v_y_cmd[stand_mask] = 0.0
    w_z_cmd[stand_mask] = 0.0
    freq_cmd[stand_mask] = 0.0 # 站立时步频锁定

```

---

### 三、 奖励函数方程组 (Reward Function Math)

采用乘法结构，确保机器狗在偏离安全姿态时任务奖励被指数级削弱。

$$Total\ Reward = r_{task} \times \exp(0.02 \times r_{aux})$$

**1. 追踪任务奖励 ($r_{task}$)**


$$r_{task} = 0.5 \exp\left(-\frac{(v_x - v_x^{cmd})^2}{0.1}\right) + 0.3 \exp\left(-\frac{(v_y - v_y^{cmd})^2}{0.05}\right) + 0.2 \exp\left(-\frac{(\omega_z - \omega_z^{cmd})^2}{0.05}\right)$$

**2. 辅助惩罚项 ($r_{aux} \le 0$)**


$$r_{aux} = w_1 L_{action\_rate} + w_2 L_{torque} + w_3 L_{phase} + w_4 L_{sym} + w_5 L_{dof\_vel}$$

* **动作平滑 ($w_1 = -0.01$):** $L_{action\_rate} = \sum_{i=1}^{12} (a_{t, i} - a_{t-1, i})^2$
* **扭矩功耗 ($w_2 = -0.0002$):** $L_{torque} = \sum_{i=1}^{12} \tau_i^2$
* **连续相位追踪 ($w_3 = -1.0$):** 设腿 $i$ 的目标相位为 $\Phi_i(t)$，期望状态 $C_i = \sin(\Phi_i(t))$。
* 若 $C_i > 0$ (应腾空) 且 触地力 $F_z > 0$: 误差 $= C_i \times F_z$
* 若 $C_i \le 0$ (应触地) 且 足端速度 $v_{xy} > 0$: 误差 $= |C_i| \times ||v_{xy}||$


* **对称性增强 ($w_4 = -0.5$):** $L_{sym} = ||a_t - \text{mirror}(a_t)||^2$
* **关节速度限制 ($w_5 = -0.001$):** $L_{dof\_vel} = \sum_{i=1}^{12} \dot{q}_i^2$

---

### 四、 参数配置 (Parameter Configuration)

```yaml
# 1. 物理环境与数据维度 (Genesis)
env:
  num_envs: 4096                  
  episode_length_s: 24.0          
  action_delay_steps: 1           # 对应管线 Step 2
  symmetry_augmentation: true     # 强制开启，防局部最优
  terrain:
    type: "plane"                 
    static_friction: 1.0          

# 2. 观测空间缩放 (Observation Scales)
obs_scales:
  lin_vel: 2.0                    # (仅供 Critic 使用)
  ang_vel: 0.25                   
  dof_pos: 1.0                    
  dof_vel: 0.05                   # 防止高频噪声导致梯度爆炸
  commands: 2.0                   

# 3. 自研硬件执行器 (Actuator Control Pipeline)
control:
  decimation: 4                   # 策略频率 (如 50Hz) 与底层物理频率 (如 200Hz) 的比值
  stiffness: 20.0                 # Kp
  damping: 0.5                    # Kd
  action_scale: 0.25              # 对应管线 Step 3
  clip_actions: 100.0             # 对应管线 Step 1
  soft_joint_pos_limit_factor: 0.9# 对应管线 Step 4
  torque_limits: 25.0             # 绝对峰值扭矩限制

# 4. 强化学习核心超参数 (RSL-RL PPO)
algorithm:
  class_name: PPO
  clip_param: 0.2                 
  entropy_coef: 0.01              # 初期 0.01，后期可退火至 0.005
  num_learning_epochs: 5          
  num_mini_batches: 4
  learning_rate: 1.0e-3           
  schedule: "adaptive"            
  desired_kl: 0.01                # 动态调低/调高 LR 的 KL 目标锚点

```