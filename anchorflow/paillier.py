import math
import secrets

try:
    import gmpy2

    def _powmod(b, e, m):
        return int(gmpy2.powmod(b, e, m))

    BACKEND = "gmpy2"
except ImportError:
    _powmod = pow
    BACKEND = "python"

_SMALL_PRIMES = [p for p in range(3, 2000) if all(p % d for d in range(2, int(p ** 0.5) + 1))]


def _is_probable_prime(n, rounds=40):
    if n < 2:
        return False
    for p in _SMALL_PRIMES:
        if n % p == 0:
            return n == p
    d, s = n - 1, 0
    while d % 2 == 0:
        d //= 2
        s += 1
    for _ in range(rounds):
        a = secrets.randbelow(n - 3) + 2
        x = pow(a, d, n)
        if x in (1, n - 1):
            continue
        for _ in range(s - 1):
            x = x * x % n
            if x == n - 1:
                break
        else:
            return False
    return True


def _prime(bits):
    while True:
        c = secrets.randbits(bits) | (1 << (bits - 1)) | 1
        if _is_probable_prime(c):
            return c


class PublicKey:
    def __init__(self, n):
        self.n = n
        self.n2 = n * n
        self.bits = n.bit_length()

    def randomizer(self):
        r = secrets.randbelow(self.n - 1) + 1
        return _powmod(r, self.n, self.n2)

    def encrypt(self, m, rn=None):
        if rn is None:
            rn = self.randomizer()
        return (1 + (m % self.n) * self.n) % self.n2 * rn % self.n2

    def add(self, c1, c2):
        return c1 * c2 % self.n2

    def smul(self, c, k):
        return _powmod(c, k, self.n2)

    def ct_bytes(self):
        return (2 * self.bits + 7) // 8


class ThresholdKey:
    def __init__(self, pub, shares):
        self.pub = pub
        self.shares = shares

    def partial(self, j, c):
        e = self.shares[j]
        if e < 0:
            return _powmod(pow(c, -1, self.pub.n2), -e, self.pub.n2)
        return _powmod(c, e, self.pub.n2)

    def combine(self, partials):
        n, n2 = self.pub.n, self.pub.n2
        x = 1
        for p in partials:
            x = x * p % n2
        return (x - 1) // n % n

    def decrypt(self, c):
        return self.combine([self.partial(j, c) for j in range(len(self.shares))])


def keygen(bits, parties, sigma=64):
    while True:
        p = _prime(bits // 2)
        q = _prime(bits // 2)
        if p != q and (p * q).bit_length() == bits:
            break
    n = p * q
    lam = (p - 1) * (q - 1) // math.gcd(p - 1, q - 1)
    d = lam * pow(lam, -1, n)
    bound = 1 << (d.bit_length() + sigma)
    shares = [secrets.randbelow(bound) for _ in range(parties - 1)]
    shares.append(d - sum(shares))
    return ThresholdKey(PublicKey(n), shares)


def slot_bits(k_bits, sigma, n_parties, wmax_bits, f_w):
    c_bits = wmax_bits + math.ceil(math.log2(max(n_parties, 2))) + k_bits + sigma
    rho_bits = c_bits + sigma
    return rho_bits + math.ceil(math.log2(max(n_parties, 2))) + 1, c_bits, rho_bits


def slots_per_ct(pub, beta):
    return (pub.bits - 2) // beta
