import os
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
from scipy.spatial.distance import pdist, squareform
from itertools import combinations

# --- rl4co ---
from rl4co.envs.routing import TSPEnv, TSPGenerator
from rl4co.models import AttentionModelPolicy, POMO

# --- Optional solvers ---
CONCORDE_AVAILABLE = False
try:
    from concorde.tsp import TSPSolver
    CONCORDE_AVAILABLE = True
except ImportError:
    pass

# ==============================
# 经典 TSP Baseline 实现
# ==============================

def random_tour(n):
    tour = list(range(n))
    random.shuffle(tour)
    return tour

def nearest_neighbor(dist_matrix):
    n = len(dist_matrix)
    visited = [False] * n
    tour = [0]
    visited[0] = True
    for _ in range(n - 1):
        last = tour[-1]
        next_city = min(
            (j for j in range(n) if not visited[j]),
            key=lambda j: dist_matrix[last][j]
        )
        tour.append(next_city)
        visited[next_city] = True
    return tour

def christofides_tsp(coords):
    try:
        import networkx as nx
        
        n = len(coords)
        G = nx.complete_graph(n)
        for i, j in combinations(range(n), 2):
            dist = np.linalg.norm(coords[i] - coords[j])
            G[i][j]['weight'] = dist

        MST = nx.minimum_spanning_tree(G)
        odd_vertices = [v for v, d in MST.degree() if d % 2 == 1]
        subgraph = G.subgraph(odd_vertices)

        # 兼容 NetworkX >= 2.8（主流版本）
        matching = nx.algorithms.min_weight_matching(subgraph, maxcardinality=True)

        multigraph = nx.MultiGraph(MST.edges())
        multigraph.add_edges_from(matching)
        euler_tour = list(nx.eulerian_circuit(multigraph))
        hamiltonian = []
        visited = set()
        for u, v in euler_tour:
            if u not in visited:
                hamiltonian.append(u)
                visited.add(u)
        if len(hamiltonian) < n:
            missing = [x for x in range(n) if x not in visited]
            hamiltonian.extend(missing)
        return hamiltonian[:n]
    except Exception as e:
        print(f"Christofides failed: {e}. Falling back to NN.")
        dist_matrix = squareform(pdist(coords))
        return nearest_neighbor(dist_matrix)

def solve_with_concorde(coords):
    if not CONCORDE_AVAILABLE:
        raise RuntimeError("Concorde not available")
    solver = TSPSolver.from_data(coords[:, 0], coords[:, 1], norm="EUC_2D")
    solution = solver.solve()
    return solution.tour, solution.optimal_value

def compute_tour_length(tour, coords):
    n = len(tour)
    return sum(np.linalg.norm(coords[tour[(i+1) % n]] - coords[tour[i]]) for i in range(n))

# ==============================
# 2-opt 局部搜索
# ==============================
def two_opt(tour, dist_matrix, max_iter=10000):
    def calculate_total_distance(tour, dist_matrix):
        return sum(dist_matrix[tour[i], tour[i+1]] for i in range(len(tour)-1))

    if tour[0] != tour[-1]:
        raise ValueError("Input tour must be a cycle (first city == last city).")

    n = len(tour) - 1
    current_tour = tour[:]
    current_distance = calculate_total_distance(current_tour, dist_matrix)
    iteration = 0

    while iteration < max_iter:
        improved = False
        for i in range(1, n - 1):
            for j in range(i + 1, n):
                new_tour = current_tour[:i] + current_tour[i:j+1][::-1] + current_tour[j+1:]
                new_distance = calculate_total_distance(new_tour, dist_matrix)
                if new_distance < current_distance:
                    current_tour = new_tour
                    current_distance = new_distance
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
        iteration += 1

    return current_tour, current_distance

# ==============================
# 主程序
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", type=str, default=None,
                        help="Model version name (e.g., 'epochs50_numloc50_lr1e-4_bs64_seed12345'). If None, auto-find latest.")
    parser.add_argument("--num_instances", type=int, default=10, help="Number of test instances")
    parser.add_argument("--sampling", action="store_true", help="Use sampling (128x) for RL")
    parser.add_argument("--aug", action="store_true", help="Use 8-augmentation for RL")
    args = parser.parse_args()

    desktop = Path.home() / "Desktop"
    output_dir = desktop / "DASE7111_TR_ENHANCED"
    output_dir.mkdir(exist_ok=True)

    # === 加载模型 ===
    generator = TSPGenerator(num_loc=50)
    env = TSPEnv(generator)
    policy = AttentionModelPolicy(env_name=env.name, num_encoder_layers=6)
    model = POMO(env, policy, batch_size=64, optimizer_kwargs={"lr": 1e-4})

    # 确定 checkpoint 目录
    if args.version:
        ckpt_dir = Path("lightning_logs/tsp_pomo") / args.version / "checkpoints"
    else:
        log_root = Path("lightning_logs/tsp_pomo")
        if not log_root.exists():
            raise FileNotFoundError("No lightning_logs/tsp_pomo directory found!")
        versions = [d for d in log_root.iterdir() if d.is_dir()]
        if not versions:
            raise FileNotFoundError("No model versions found!")
        latest_version = max(versions, key=lambda d: d.stat().st_mtime)
        ckpt_dir = latest_version / "checkpoints"

    # 加载 checkpoint
    last_ckpt = ckpt_dir / "last.ckpt"
    if last_ckpt.exists():
        ckpt_path = last_ckpt
    else:
        ckpt_files = list(ckpt_dir.glob("*.ckpt"))
        if not ckpt_files:
            raise FileNotFoundError(f"No .ckpt files found in {ckpt_dir}")
        ckpt_path = max(ckpt_files, key=lambda f: f.stat().st_mtime)

    print(f"✅ Loading model from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()

    # === 构建方法列表（仅包含有效方法）===
    methods = ["RL (Greedy)", "RL (Sampling)", "RL (Aug)", 
               "Heuristic (NN+2opt)", "Christofides", "Random"]
    if CONCORDE_AVAILABLE:
        methods.append("Concorde (Optimal)")

    results = {method: [] for method in methods}
    all_coords = []

    torch.manual_seed(2025)
    np.random.seed(2025)
    random.seed(2025)

    for i in range(args.num_instances):
        print(f"🔄 Instance {i+1}/{args.num_instances}")
        td = env.reset(batch_size=[1])
        locs = td["locs"][0].cpu().numpy()
        all_coords.append(locs)

        # ---- RL Methods ----
        with torch.no_grad():
            out_greedy = model(td, decode_type="greedy")
            rl_greedy_len = -out_greedy["reward"][0].item()
            results["RL (Greedy)"].append(rl_greedy_len)

            if args.sampling:
                out_sample = model(td, decode_type="sampling", num_samples=128)
                best_idx = out_sample["reward"].argmax()
                rl_sample_len = -out_sample["reward"][best_idx].item()
                results["RL (Sampling)"].append(rl_sample_len)
            else:
                results["RL (Sampling)"].append(None)

            if args.aug:
                out_aug = model(td, decode_type="greedy", num_augment=8)
                best_idx = out_aug["reward"].argmax()
                rl_aug_len = -out_aug["reward"][best_idx].item()
                results["RL (Aug)"].append(rl_aug_len)
            else:
                results["RL (Aug)"].append(None)

        # ---- Baselines ----
        dist_mat = squareform(pdist(locs))
        nn_tour = nearest_neighbor(dist_mat)
        full_tour = nn_tour + [nn_tour[0]]
        try:
            _, heur_len = two_opt(full_tour, dist_mat)
            results["Heuristic (NN+2opt)"].append(heur_len)
        except Exception as e:
            print(f"⚠️ 2-opt failed on instance {i+1}: {e}")
            results["Heuristic (NN+2opt)"].append(compute_tour_length(nn_tour, locs))

        try:
            chris_tour = christofides_tsp(locs)
            chris_len = compute_tour_length(chris_tour, locs)
            results["Christofides"].append(chris_len)
        except Exception as e:
            print(f"⚠️ Christofides failed: {e}")
            results["Christofides"].append(None)

        rand_tour = random_tour(50)
        rand_len = compute_tour_length(rand_tour, locs)
        results["Random"].append(rand_len)

        if CONCORDE_AVAILABLE:
            try:
                _, concorde_len = solve_with_concorde(locs)
                results["Concorde (Optimal)"].append(concorde_len)
            except Exception as e:
                print(f"⚠️ Concorde failed: {e}")
                results["Concorde (Optimal)"].append(None)

    # === 可视化 1: 所有方法并排图（前3个实例）===
    valid_methods = [m for m in methods if any(v is not None for v in results[m])]
    for idx in range(min(3, args.num_instances)):
        fig, axs = plt.subplots(2, 4, figsize=(16, 8))
        axs = axs.flatten()
        locs = all_coords[idx]

        for ax_i, method in enumerate(valid_methods):
            if ax_i >= len(axs):
                break
            ax = axs[ax_i]
            ax.scatter(locs[:, 0], locs[:, 1], s=40, color="red", zorder=5)

            # 获取路径用于绘图
            if method == "RL (Greedy)":
                td_viz = env.reset(batch_size=[1])
                td_viz["locs"] = torch.tensor(locs).unsqueeze(0)
                with torch.no_grad():
                    out = model(td_viz, decode_type="greedy")
                tour = out["actions"][0].cpu().numpy().tolist()
            elif method == "Heuristic (NN+2opt)":
                nn_tour = nearest_neighbor(squareform(pdist(locs)))
                full_tour = nn_tour + [nn_tour[0]]
                opt_tour, _ = two_opt(full_tour, squareform(pdist(locs)))
                tour = opt_tour[:-1]
            elif method == "Christofides":
                tour = christofides_tsp(locs)
            elif method == "Random":
                tour = random_tour(50)
            else:
                tour = list(range(50))  # fallback

            full_tour_plot = tour + [tour[0]]
            ax.plot(locs[full_tour_plot, 0], locs[full_tour_plot, 1], linewidth=1.5)
            length = results[method][idx] or 0
            ax.set_title(f"{method}\n{length:.3f}", fontsize=9)
            ax.axis("equal")
            ax.set_xticks([])
            ax.set_yticks([])

        for ax_i in range(len(valid_methods), len(axs)):
            axs[ax_i].axis("off")

        plt.tight_layout()
        plt.savefig(output_dir / f"instance_{idx+1:02d}_all_methods.png", dpi=150)
        plt.close()

    
  # === 可视化 2: 箱线图（排除 Random）===
    plt.figure(figsize=(12, 6))
    plot_data = []
    labels = []
    for method in valid_methods:
        if method.lower() == "random":  # 跳过 Random
            continue
        data = [x for x in results[method] if x is not None]
        if data:
            plot_data.append(data)
            labels.append(method)

    if plot_data:  # 确保有数据可画
        plt.boxplot(plot_data, labels=labels)
        plt.ylabel("Tour Length")
        plt.title("Solution Quality Distribution Across Methods (Excluding Random)")
        plt.xticks(rotation=30, ha='right')
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.savefig(output_dir / "boxplot_comparison.png", dpi=200)
        plt.close()
    else:
        print("⚠️ No non-Random methods to plot in boxplot.")

        # === 可视化 3: 所有方法折线图 ===
    plt.figure(figsize=(12, 7))
    instances = np.arange(1, args.num_instances + 1)

    for method in valid_methods:
        data = []
        for i in range(args.num_instances):
            val = results[method][i]
            # Use NaN for missing values so matplotlib skips them
            data.append(val if val is not None else np.nan)
        plt.plot(instances, data, marker='o', linewidth=2, label=method, markersize=4)

    plt.xlabel('Instance ID')
    plt.ylabel('Tour Length')
    plt.title('Tour Length Comparison Across All Methods and Instances')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 减少重叠
    plt.tight_layout()
    plt.savefig(output_dir / "all_methods_lineplot.png", dpi=200, bbox_inches='tight')
    plt.close()
    # === 可视化 4: 所有方法（除了Random）折线图 ===
    plt.figure(figsize=(12, 7))
    instances = np.arange(1, args.num_instances + 1)

# 过滤掉 'random' 方法
    valid_methods = [method for method in valid_methods if method.lower() != 'random']

    for method in valid_methods:
        data = []
        for i in range(args.num_instances):
            val = results[method][i]
        # Use NaN for missing values so matplotlib skips them
            data.append(val if val is not None else np.nan)
        plt.plot(instances, data, marker='o', linewidth=2, label=method, markersize=4)

    plt.xlabel('Instance ID')
    plt.ylabel('Tour Length')
    plt.title('Tour Length Comparison Across All Methods (Excluding Random) and Instances')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # 减少重叠
    plt.tight_layout()
    plt.savefig(output_dir / "all_methods_except_random_lineplot.png", dpi=200, bbox_inches='tight')
    plt.close()
    # === 统计分析 ===
    print("\n📊 FINAL STATISTICS:")
    best_per_instance = []
    for i in range(args.num_instances):
        valid_lengths = [results[m][i] for m in valid_methods if results[m][i] is not None]
        best_per_instance.append(min(valid_lengths) if valid_lengths else np.nan)

    for method in valid_methods:
        data = [x for x in results[method] if x is not None]
        if not data:
            continue
        avg = np.mean(data)
        std = np.std(data)
        wins = sum(
            (results[method][i] is not None) and
            (results[method][i] <= best_per_instance[i] + 1e-5)
            for i in range(args.num_instances)
        )
        gap_vals = []
        for i in range(args.num_instances):
            if results[method][i] is not None and best_per_instance[i] is not None:
                gap = (results[method][i] - best_per_instance[i]) / best_per_instance[i] * 100
                gap_vals.append(gap)
        gap_to_best = np.mean(gap_vals) if gap_vals else np.nan
        print(f"{method:25s} | Avg: {avg:.4f} ± {std:.4f} | Wins: {wins}/{args.num_instances} | %Gap: {gap_to_best:.2f}%")

    print(f"\n🎉 All results saved to: {output_dir}")

if __name__ == "__main__":
    main()