# # kl_fix.py
# import torch
# from torch.distributions import kl
# from torch.distributions.independent import Independent
#
# # PyTorch 内部帮助函数的等价行为：对最右边 n 个维度求和
# def _sum_rightmost(value: torch.Tensor, n: int) -> torch.Tensor:
#     if n == 0:
#         return value
#     # 兼容老版本 torch：把右侧 n 维拉平再在最后一维求和
#     return value.view(*value.shape[:-n], -1).sum(-1)
#
# # 注册 Independent vs Independent 的 KL
# @kl.register_kl(Independent, Independent)
# def _kl_independent_independent(p: Independent, q: Independent):
#     """
#     KL(Independent(p.base_dist), Independent(q.base_dist))
#     要求两者 reinterpreted_batch_ndims 一致。
#     """
#     if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
#         raise NotImplementedError(
#             "KL for Independent with different reinterpreted_batch_ndims "
#             f"({p.reinterpreted_batch_ndims} vs {q.reinterpreted_batch_ndims})"
#         )
#     # 对 base_dist 逐元素 KL，再把右侧 n 个维度求和
#     result = kl.kl_divergence(p.base_dist, q.base_dist)
#     return _sum_rightmost(result, p.reinterpreted_batch_ndims)
#
# print("Registered KL(Independent || Independent)")
#--------------------
# kl_fix.py —— 注册 KL(Independent || Independent)，支持解析式 KL
import torch
from torch.distributions import kl
from torch.distributions.independent import Independent

def _sum_rightmost(value: torch.Tensor, n: int) -> torch.Tensor:
    if n == 0:
        return value
    # 把右侧 n 维拉平，再在最后一维求和（兼容老版本）
    return value.view(*value.shape[:-n], -1).sum(-1)

@kl.register_kl(Independent, Independent)
def _kl_independent_independent(p: Independent, q: Independent):
    if p.reinterpreted_batch_ndims != q.reinterpreted_batch_ndims:
        raise NotImplementedError(
            f"KL for Independent with different reinterpreted_batch_ndims "
            f"({p.reinterpreted_batch_ndims} vs {q.reinterpreted_batch_ndims})"
        )
    result = kl.kl_divergence(p.base_dist, q.base_dist)
    return _sum_rightmost(result, p.reinterpreted_batch_ndims)

print("Registered KL(Independent || Independent)")
