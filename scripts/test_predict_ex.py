import numpy as np
import aha
import json
from scipy.stats import multivariate_normal


def industrial_stable_validation():
    dim, rank = 4, 3
    
    # 构造分量 0: 正相关强耦合
    mu0 = np.array([1.0, 1.0, 2.0, 2.0])
    s0 = np.array([
        [1.2, 0.3, 0.8, 0.0],
        [0.3, 1.1, 0.0, 0.5],
        [0.8, 0.0, 1.5, 0.2],
        [0.0, 0.5, 0.2, 1.3]
    ])
    
    # 构造分量 1: 负相关强耦合
    mu1 = np.array([0.0, 0.0, 0.0, 0.0])
    s1 = np.array([
        [1.0, -0.2, -0.5, 0.0],
        [-0.2, 1.2, 0.0, -0.4],
        [-0.5, 0.0, 1.4, 0.1],
        [0.0, -0.4, 0.1, 1.1]
    ])

    # 构造分量 2: 独立高方差 (噪声项)
    mu2 = np.array([-1.0, -1.0, -2.0, -2.0])
    s2 = np.eye(4) * 2.0

    # 1. 构造模型 (w 无零项)
    config = {
        "r": rank, "d": dim,
        "w": [0.5, 0.3, 0.2],
        "c": [
            {"u": mu0.tolist(), "s": s0.flatten().tolist()},
            {"u": mu1.tolist(), "s": s1.flatten().tolist()},
            {"u": mu2.tolist(), "s": s2.flatten().tolist()}
        ]
    }
    
    model = aha.Model(rank, dim)
    model.Import(json.dumps(config))

    # 2. 准备输入
    x_obs = np.array([0.5, 0.5], dtype=np.float64)
    x_batch_d = x_obs.reshape(1, -1)
    x_batch_f = x_batch_d.astype(np.float32)

    # 3. 三路接口对齐验收
    p1, mu1_out, v1 = model.PredictEx(x_obs)
    p2, mu2_out, v2 = model.BatchPredictEx(x_batch_d)
    p3, mu3_out, v3 = model.FastPredictEx(x_batch_f)

    # 4. 输出结果
    print("=== Aha 工业级稳定性验收 (Rank=3, No-Zero Weights) ===")
    print(f"Prob (Double): {p1:.6f}")
    print(f"Mu   (Double): {mu1_out}")
    print(f"Var  (Double): {v1}")
    
    # 5. 严格的一致性断言
    assert np.allclose(mu1_out, mu2_out[0], atol=1e-12), "BatchPredictEx 均值对齐失败"
    assert np.allclose(v1, v2[0], atol=1e-12), "BatchPredictEx 方差对齐失败"
    assert np.allclose(mu1_out, mu3_out[0], atol=1e-5), "FastPredictEx 精度超出容差"

    print("\n[验收通过] 权重分布自然，三路接口逻辑一致。")


def independent_audit():
    # --- 1. 定义与 C++ 完全一致的模型参数 ---
    dim, rank = 4, 3
    w = np.array([0.5, 0.3, 0.2])
    
    # 分量参数定义 (与你刚才运行的 C++ 模型配置完全同步)
    mu = [
        np.array([1.0, 1.0, 2.0, 2.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([-1.0, -1.0, -2.0, -2.0])
    ]
    
    s = [
        np.array([[1.2, 0.3, 0.8, 0.0], [0.3, 1.1, 0.0, 0.5], [0.8, 0.0, 1.5, 0.2], [0.0, 0.5, 0.2, 1.3]]),
        np.array([[1.0, -0.2, -0.5, 0.0], [-0.2, 1.2, 0.0, -0.4], [-0.5, 0.0, 1.4, 0.1], [0.0, -0.4, 0.1, 1.1]]),
        np.eye(4) * 2.0
    ]

    x_obs = np.array([0.5, 0.5])

    # --- 2. 独立解析计算过程 ---
    cond_mus = []
    cond_covs = []
    likelihoods = []

    for k in range(rank):
        # 拆分矩阵 [11, 12 / 21, 22]
        s11 = s[k][:2, :2]
        s12 = s[k][:2, 2:]
        s21 = s[k][2:, :2]
        s22 = s[k][2:, 2:]
        m1 = mu[k][:2]
        m2 = mu[k][2:]

        # A. 计算该分量的观测似然度 (似然权重)
        l_k = multivariate_normal.pdf(x_obs, mean=m1, cov=s11)
        likelihoods.append(l_k)

        # B. 计算单个高斯分量的条件均值和方差 (Schur Complement)
        inv_s11 = np.linalg.inv(s11)
        gain = s21 @ inv_s11
        m_cond = m2 + gain @ (x_obs - m1)
        s_cond = s22 - gain @ s12
        
        cond_mus.append(m_cond)
        cond_covs.append(s_cond)

    # --- 3. 混合逻辑 (Mixing Logic) ---
    # 计算后验权重 w
    w_post = (w * likelihoods) / np.sum(w * likelihoods)
    
    # 计算最终混合均值
    final_mu = np.zeros(2)
    for k in range(rank):
        final_mu += w_post[k] * cond_mus[k]
        
    # 计算最终混合方差 (含补偿项)
    final_cov = np.zeros((2, 2))
    for k in range(rank):
        # Formula: Sum( w * (Sigma_cond + mu_cond * mu_cond.T) ) - mu_mix * mu_mix.T
        final_cov += w_post[k] * (cond_covs[k] + np.outer(cond_mus[k], cond_mus[k]))
    
    final_cov -= np.outer(final_mu, final_mu)
    final_var_diag = np.diag(final_cov)

    # --- 4. 打印审查报告 ---
    print(f"=== 独立第三方解析审计 (基于 NumPy/SciPy) ===")
    print(f"解析对数似然 Prob: {np.log(np.sum(w * likelihoods)):.6f}")
    print(f"解析预期均值   Mu: {final_mu}")
    print(f"解析预期方差  Var: {final_var_diag}")
    print(f"后验权重        w: {w_post}")



if __name__ == "__main__":
    industrial_stable_validation()
    independent_audit()

