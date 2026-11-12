import numpy as np

from .mpc import Shared, random_shared_perm, composed_inverse_perm, insecure_sort_perm, inverse
from .ring import U64, ALL1


class DagList:
    def __init__(self, ctx, n_vert, src, dst, w_fp, step="T3"):
        self.ctx = ctx
        self.nv = n_vert
        self.ne = len(src)
        self.N = n_vert + self.ne
        self.step = step
        meter = ctx.meter
        with meter.in_phase("init"):
            vid = np.arange(n_vert, dtype=np.int64)
            col_src = np.concatenate([vid, src]).astype(U64)
            col_dst = np.concatenate([vid, dst]).astype(U64)
            col_isv = np.concatenate([np.ones(n_vert), np.zeros(self.ne)]).astype(U64)
            col_w = np.concatenate([np.zeros(n_vert, dtype=U64), np.asarray(w_fp, dtype=U64)])
            vert_part = ctx.share_public(np.stack([col_src, col_dst, col_isv, col_w], axis=1)[:n_vert])
            edge_part = ctx.share_from_owner(np.stack([col_src, col_dst, col_isv, col_w], axis=1)[n_vert:], step + "-input")
            dag = Shared(np.concatenate([vert_part.s0, edge_part.s0]), np.concatenate([vert_part.s1, edge_part.s1]))
            self.pA = random_shared_perm(ctx, self.N, step + "-perm")
            self.pB = random_shared_perm(ctx, self.N, step + "-perm")
            self.pC = composed_inverse_perm(ctx, self.pA, self.pB, step + "-perm")
            shA = self.pA.apply(dag, step=step + "-init-shuffle")
            NN = U64(self.N)
            two = U64(2)
            tie_s = ctx.rng_01.permutation(self.N).astype(U64)
            tie_d = ctx.rng_01.permutation(self.N).astype(U64)
            key_s = (shA.col(0).mul_public(two) + shA.col(2).mul_public(ALL1).add_public(U64(1))).mul_public(NN).add_public(tie_s)
            self.piS, self.levels_s = insecure_sort_perm(ctx, key_s, step + "-isort")
            shB = self.pB.apply(shA, step=step + "-init-shuffle")
            key_d = (shB.col(1).mul_public(two) + shB.col(2)).mul_public(NN).add_public(tie_d)
            self.piD, self.levels_d = insecure_sort_perm(ctx, key_d, step + "-isort")
            self.piS_inv = inverse(self.piS)
            self.piD_inv = inverse(self.piD)
            self.w_src = shA.col(3).permute(self.piS)

    def iterate(self, x):
        ctx = self.ctx
        nv, N = self.nv, self.N
        s = Shared.zeros(N)
        s.s0[:nv] = x.s0
        s.s1[:nv] = x.s1
        d = s.copy()
        d.s0[1:nv] = x.s0[1:] - x.s0[:-1]
        d.s1[1:nv] = x.s1[1:] - x.s1[:-1]
        both = self.pA.apply(Shared.stack([d, s]), step=self.step + "-shuffle").permute(self.piS)
        r = both.col(0).cumsum() - both.col(1)
        t = ctx.mul(self.w_src, r, step=self.step + "-applyE")
        t = self.pB.apply(t.permute(self.piS_inv), step=self.step + "-shuffle").permute(self.piD)
        g = t.cumsum().permute(self.piD_inv)
        g = self.pC.apply(g, step=self.step + "-shuffle")
        gv = g[:nv]
        out = gv.copy()
        out.s0[1:] = gv.s0[1:] - gv.s0[:-1]
        out.s1[1:] = gv.s1[1:] - gv.s1[:-1]
        return out
