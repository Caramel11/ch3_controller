"""
模糊逻辑 alpha 仲裁工具。

本文件主要服务于原有 `coop_fuzzy`、`force_margin` 等对照策略。当前推荐的
CAC2026 在线优先级策略 `online_priority` 不直接依赖本文件，但为了保持
STRATEGIES 中旧策略可运行，本文件仍保留在 0526_controller 目录下。

核心流程:
  1. init_fuzzy_sets 定义输入/输出模糊集；
  2. fuzzify 将物理输入映射为各模糊标签隶属度；
  3. infer 根据 27 条规则做 Mamdani min-max 推理；
  4. defuzzify 用重心法输出连续 lambda 或 delta_lambda。
"""
import numpy as np
import matplotlib.pyplot as plt
from kalman_filter import KalmanFilterFusion, DeltaLambdaUpdater


class FuzzyLogicTool:
    """三输入一输出的模糊推理器。

    type="lambda_based" 时直接输出 lambda/alpha 候选值；
    type="delta_lambda_based" 时输出 lambda 的变化量，用于与 KalmanFilterFusion
    融合。该类保留原项目的调用形状，避免旧调度器失效。
    """

    def __init__(self, type="lambda_based"):
        self.type = type  # "lambda_based" 或 "delta_lambda_based"
        self.init_fuzzy_sets(self.type)
        self.init_fuzzy_rules(self.type)
        self.kf_fusion = KalmanFilterFusion(dt=0.01, epsilon=0.01)
        self.delta_updater = DeltaLambdaUpdater(lambda0=0.5)
        self.lambda_kf_list = []
        self.lambda_list = []
        self.F_h1_list = []

    def init_fuzzy_sets(self, types="lambda_based"):
        """定义三角形隶属函数参数（严格匹配文章实验I）"""
        if types == "lambda_based":
            # 输入：F_h(N)[0, 2]、T_h(s)[0, 6]、D_r(cm)[0, 0.1]
            self.input_sets = [
                # F_h的3个模糊集：小(PS)、中(PM)、大(PL)
                {"PS": (0.0, 0.4, 0.8), "PM": (0.4, 1, 1.5), "PL": (1.2, 1.5, 2)},
                # T_h的3个模糊集
                {
                    "PS": (0, 1, 2),
                    "PM": (1, 3, 5),
                    "PL": (3, 5, 6),
                },
                # D_r的3个模糊集
                {
                    "PS": (0.0, 0.02, 0.04),
                    "PM": (0.03, 0.05, 0.07),
                    "PL": (0.06, 0.08, 0.1),
                },
            ]
            # 输出λ的5个模糊集：NL(0.0-0.2)、Z(0.15-0.45)、PS(0.4-0.6)、PM(0.55-0.85)、PL(0.8-1.0)
            self.output_sets = {
                "Z": (0.0, 0.05, 0.1),
                "PS": (0.05, 0.25, 0.5),
                "PM": (0.25, 0.5, 0.7),
                "P": (0.5, 0.7, 0.95),
                "PL": (0.9, 0.95, 1.0),
            }
        elif types == "delta_lambda_based":
            # 输入：dF_h(N/s)[-100,100]、dT_h(1)[-600,600]、dD_r(cm/s)[-5,5]
            self.input_sets = [
                {
                    "N": (-3.0, -2.0, -0.2),
                    "Z": (-1.5, 0.0, 1.5),
                    "P": (0.2, 2.0, 3.0),
                },
                {
                    "N": (0, 0.7, 1),
                    "Z": (0.7, 1, 1.3),
                    "P": (1, 1.3, 2),
                },
                {
                    "N": (-0.08, -0.04, -0.01),
                    "Z": (-0.03, 0.0, 0.03),
                    "P": (0.01, 0.04, 0.08),
                },
            ]
            # 输出Δλ的5个模糊集：NL(-0.5~-0.3)、N(-0.35~-0.05)、Z(-0.1~0.1)、P(0.05~0.35)、PL(0.3~0.5)
            self.output_sets = {
                "NL": (-0.5, -0.35, -0.2),
                "N": (-0.35, -0.2, -0.0),
                "Z": (-0.2, 0.0, 0.2),
                "P": (0.0, 0.2, 0.35),
                "PL": (0.2, 0.35, 0.5),
            }

    def triangular_mf(self, x, params):
        """三角形隶属函数：params=(a,b,c)，返回μ∈[0,1]"""
        a, b, c = params
        if x <= a or x >= c:
            return 0.0
        elif a < x <= b:
            return (x - a) / (b - a)
        else:
            return (c - x) / (c - b)

    def trapezoidal_mf_type1(self, x, params):
        """梯形隶属函数（类型1）：
        - x ≤ a 或 x > c → μ=0
        - a < x ≤ b → μ=1（平台段）
        - b < x ≤ c → μ从1线性下降到0（下降段）
        params=(a, b, c) 需满足 a < b < c
        """
        a, b, c = params
        if x < a or x > c:
            return 0.0
        elif a <= x <= b:
            return 1.0  # 平台段保持1
        else:  # b < x <= c
            return (c - x) / (c - b)  # 线性下降至0

    def trapezoidal_mf_type2(self, x, params):
        """梯形隶属函数（类型2）：
        - x ≤ a 或 x > c → μ=0
        - a < x ≤ b → μ从0线性上升到1（上升段）
        - b < x ≤ c → μ=1（平台段）
        params=(a, b, c) 需满足 a < b < c
        """
        a, b, c = params
        if x < a or x > c:
            return 0.0
        elif a <= x <= b:
            return (x - a) / (b - a)  # 线性上升至1
        else:  # b < x <= c
            return 1.0  # 平台段保持1

    def fuzzify(self, inputs, types="lambda_based"):
        """模糊化：输入原始信号，返回各模糊集的隶属度（已添加输入钳位）"""
        # 输入钳位（防止超出模糊集范围）
        if types == "lambda_based":
            inputs = [
                max(0.0, min(2, inputs[0])),  # F_h
                max(0.0, min(6, inputs[1])),  # T_h
                max(0.0, min(0.1, inputs[2])),  # D_r
            ]
        else:
            inputs = [
                max(-3.0, min(3.0, inputs[0])),  # dF_h
                max(0.0, min(2.0, inputs[1])),  # dT_h
                max(-0.08, min(0.08, inputs[2])),  # dD_r
            ]

        fuzzified = []
        # 遍历输入项，同时获取索引（区分F_h/T_h/D_r）
        for i, (input_val, sets) in enumerate(zip(inputs, self.input_sets)):
            membership = {}
            for label, params in sets.items():
                # 仅对lambda_based类型下的D_r（索引2）的PS标签使用梯形隶属函数type1
                if label == "PS" or label == "N":
                    membership[label] = self.trapezoidal_mf_type1(input_val, params)
                elif label == "PL" or label == "P":
                    membership[label] = self.trapezoidal_mf_type2(input_val, params)
                else:
                    membership[label] = self.triangular_mf(input_val, params)
            fuzzified.append(membership)

        # for input_val, sets in zip(inputs, self.input_sets):
        #     membership = {
        #         label: self.triangular_mf(input_val, params)
        #         for label, params in sets.items()
        #     }
        #     fuzzified.append(membership)
        return fuzzified

    def init_fuzzy_rules(self, types="lambda_based"):
        """定义27条模糊规则（与文章完全一致）"""
        if types == "lambda_based":
            self.rule_dict = {
                ("PS", "PS", "PS"): "Z",
                ("PS", "PS", "PM"): "Z",
                ("PS", "PS", "PL"): "PS",
                ("PS", "PM", "PS"): "Z",
                ("PS", "PM", "PM"): "PS",
                ("PS", "PM", "PL"): "PS",
                ("PS", "PL", "PS"): "Z",
                ("PS", "PL", "PM"): "PS",
                ("PS", "PL", "PL"): "PM",
                ("PM", "PS", "PS"): "Z",
                ("PM", "PS", "PM"): "PS",
                ("PM", "PS", "PL"): "PM",
                ("PM", "PM", "PS"): "PS",
                ("PM", "PM", "PM"): "PM",
                ("PM", "PM", "PL"): "PM",
                ("PM", "PL", "PS"): "PS",
                ("PM", "PL", "PM"): "P",
                ("PM", "PL", "PL"): "PL",
                ("PL", "PS", "PS"): "PS",
                ("PL", "PS", "PM"): "PM",
                ("PL", "PS", "PL"): "PM",
                ("PL", "PM", "PS"): "PS",
                ("PL", "PM", "PM"): "P",
                ("PL", "PM", "PL"): "PL",
                ("PL", "PL", "PS"): "PM",
                ("PL", "PL", "PM"): "P",
                ("PL", "PL", "PL"): "PL",
            }
        elif types == "delta_lambda_based":
            self.rule_dict = {
                # 1. 2N,人主导权很小 → Δλ=N/NL（正小/正大）
                ("N", "N", "N"): "NL",
                ("N", "N", "Z"): "NL",
                ("N", "N", "P"): "N",
                # 2. NZ,人主导权略小 → Δλ=NL/N/Z（负大/负小/零）
                ("N", "Z", "N"): "NL",
                ("N", "Z", "Z"): "N",
                ("N", "Z", "P"): "Z",
                # 3. NP,人主导权适中 → Δλ=N/Z/P（负小/零/正小）
                ("N", "P", "N"): "N",
                ("N", "P", "Z"): "Z",
                ("N", "P", "P"): "P",
                # 4. ZN，人主导权略小 → Δλ=NL/N/Z（负大/负小/零）
                ("Z", "N", "N"): "NL",
                ("Z", "N", "Z"): "N",
                ("Z", "N", "P"): "Z",
                # 5. 2Z,人主导权适中 → Δλ=N/Z/P（负小/零/正小）
                ("Z", "Z", "N"): "N",
                ("Z", "Z", "Z"): "Z",
                ("Z", "Z", "P"): "P",
                # 6. ZP，人主导权略大 → Δλ=Z/P/PL（零/正小/正大）
                ("Z", "P", "N"): "Z",
                ("Z", "P", "Z"): "P",
                ("Z", "P", "P"): "PL",
                # 7. PN，人主导权适中 → Δλ=N/Z/P（负小/零/正小）
                ("P", "N", "N"): "N",
                ("P", "N", "Z"): "Z",
                ("P", "N", "P"): "P",
                # 8. PZ，人主导权略大 → Δλ=Z/P/PL（零/正小/正大）
                ("P", "Z", "N"): "Z",
                ("P", "Z", "Z"): "P",
                ("P", "Z", "P"): "PL",
                # 9. 2P→ 人主导权很大 → Δλ=P/PL（正小/正大）
                ("P", "P", "N"): "P",
                ("P", "P", "Z"): "PL",
                ("P", "P", "P"): "PL",
            }

    def infer(self, fuzzified_inputs):
        """Mamdani推理：min-max法计算规则激活度"""
        output_membership = {}
        for (f_label, v_label, d_label), out_label in self.rule_dict.items():
            mu_f = fuzzified_inputs[0][f_label]
            mu_v = fuzzified_inputs[1][v_label]
            mu_d = fuzzified_inputs[2][d_label]
            activation = min(mu_f, mu_v, mu_d)
            if out_label in output_membership:
                if activation > output_membership[out_label]:
                    output_membership[out_label] = activation
            else:
                output_membership[out_label] = activation
        return output_membership

    def defuzzify(self, output_membership, types="lambda_based"):
        """重心法解模糊，输出连续值（已添加钳位）"""
        if types == "lambda_based":
            z_range = np.linspace(0.0, 1.0, 100)  # λ采样范围
        else:
            z_range = np.linspace(-0.5, 0.5, 100)  # Δλ采样范围

        numerator = 0.0
        denominator = 0.0
        for z in z_range:
            mu_total = 0.0
            for out_label, mu in output_membership.items():
                params = self.output_sets[out_label]
                if out_label == "PL":
                    mu_z = self.trapezoidal_mf_type2(z, params)
                elif (out_label == "Z" and types == "lambda_based") or (
                    out_label == "NL" and types == "delta_lambda_based"
                ):
                    mu_z = self.trapezoidal_mf_type1(z, params)
                else:
                    mu_z = self.triangular_mf(z, params)
                mu_total = max(mu_total, min(mu, mu_z))
            numerator += mu_total * z
            denominator += mu_total

        if denominator < 1e-6:
            result = 1e-6
        else:
            result = numerator / denominator

        # 输出钳位（确保范围正确）
        if self.type == "lambda_based":
            result = max(0.0, min(1.0, result))
        else:
            result = max(-0.5, min(0.5, result))
        return result

    def compute(self, inputs, types="lambda_based"):
        """完整计算流程：模糊化→推理→解模糊"""
        fuzzified = self.fuzzify(inputs, types)
        output_membership = self.infer(fuzzified)
        result = self.defuzzify(output_membership, types)
        return result

    def compute_kf_fusion(self, inputs, delta_inputs, dt=0.01):
        """计算模糊 lambda 并与 delta_lambda 通过 KF 融合。

        inputs 是原始物理量映射后的模糊输入；delta_inputs 是变化率输入。
        返回值 lambda_kf 是平滑后的仲裁量，主要供旧 PhaseAwareFuzzy 调度器使用。
        """
        # 先由 lambda_based 规则直接得到当前 lambda。
        lambda_w = self.compute(inputs, "lambda_based")
        dlambda_w = (
            (lambda_w - self.lambda_list[-1]) / dt if len(self.lambda_list) > 0 else 0.0
        )
        # 再由 delta_lambda_based 规则估计 lambda 变化趋势。
        delta_lambda_w = self.compute(delta_inputs, "delta_lambda_based")
        # KF 融合两个来源: 直接观测 lambda_w 和变化趋势 delta_lambda_w。
        tau_k = np.array([[dlambda_w], [delta_lambda_w]])
        lambda_kf = self.kf_fusion.update(lambda_w, tau_k)
        self.lambda_kf_list.append(lambda_kf)
        self.lambda_list.append(lambda_w)

        return lambda_kf


# Method-2（If-Else离散逻辑）：用于后续对比。
# 当前 no-RCM 推荐流程不会调用该类，但保留它可复现实验中的传统离散规则基线。
class Method2_IF_Else:
    """离散 if-else 规则基线。

    与 FuzzyLogicTool 不同，该方法不做连续隶属度推理，只判断输入落在哪些
    区间中，然后对触发规则的 lambda 取平均。适合作为简单对照组。
    """

    def __init__(self):
        self.input_intervals = {
            "F_h": {"小": (1.0, 2.0), "中": (1.5, 3.0), "大": (2.5, 3.5)},
            "V_h": {"小": (1.0, 4.0), "中": (3.0, 6.0), "大": (5.0, 7.0)},
            "D_r": {"小": (0.0, 1.0), "中": (0.8, 1.2), "大": (1.0, 1.5)},
        }
        self.rule_dict = {
            ("小", "小", "小"): 0.9,
            ("小", "小", "中"): 0.9,
            ("小", "小", "大"): 0.7,
            ("小", "中", "小"): 0.9,
            ("小", "中", "中"): 0.7,
            ("小", "中", "大"): 0.7,
            ("小", "大", "小"): 0.9,
            ("小", "大", "中"): 0.7,
            ("小", "大", "大"): 0.5,
            ("中", "小", "小"): 0.9,
            ("中", "小", "中"): 0.7,
            ("中", "小", "大"): 0.5,
            ("中", "中", "小"): 0.7,
            ("中", "中", "中"): 0.5,
            ("中", "中", "大"): 0.3,
            ("中", "大", "小"): 0.5,
            ("中", "大", "中"): 0.3,
            ("中", "大", "大"): 0.1,
            ("大", "小", "小"): 0.7,
            ("大", "小", "中"): 0.5,
            ("大", "小", "大"): 0.5,
            ("大", "中", "小"): 0.7,
            ("大", "中", "中"): 0.3,
            ("大", "中", "大"): 0.1,
            ("大", "大", "小"): 0.3,
            ("大", "大", "中"): 0.3,
            ("大", "大", "大"): 0.1,
        }

    def is_in_interval(self, value, interval):
        return interval[0] <= value <= interval[1]

    def get_matching_labels(self, input_name, value):
        labels = []
        for label, interval in self.input_intervals[input_name].items():
            if self.is_in_interval(value, interval):
                labels.append(label)
        return labels if labels else ["默认"]

    def compute(self, F_h, V_h, D_r):
        F_h = max(1.0, min(3.5, F_h))
        V_h = max(1.0, min(7.0, V_h))
        D_r = max(0.0, min(1.5, D_r))
        F_h_labels = self.get_matching_labels("F_h", F_h)
        V_h_labels = self.get_matching_labels("V_h", V_h)
        D_r_labels = self.get_matching_labels("D_r", D_r)
        triggered_lambdas = []
        for f in F_h_labels:
            for v in V_h_labels:
                for d in D_r_labels:
                    if (f, v, d) in self.rule_dict:
                        triggered_lambdas.append(self.rule_dict[(f, v, d)])
        return np.mean(triggered_lambdas) if triggered_lambdas else 0.5
