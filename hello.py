from dataclasses import dataclass
import random

random.seed(1337)


@dataclass
class Curve:
    """
    Elliptic Curve over the field of integers modulo a prime.
    Points on the curve satisfy y^2 = x^3 + a*x + b (mod p).
    """

    p: int  # the prime modulus
    a: int
    b: int


def create_bitcoin_curve():
    print("Creating bitcoin curve...")
    return Curve(
        p=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
        a=0x0000000000000000000000000000000000000000000000000000000000000000,  # a = 0
        b=0x0000000000000000000000000000000000000000000000000000000000000007,  # b = 7
    )


@dataclass
class Point:
    """An integer point (x, y) on a Curve"""

    curve: Curve
    x: int
    y: int

    def __add__(self, other):
        # Check if 'other' is an instance of Point
        if isinstance(other, Point):
            return elliptic_curve_addition(self, other)
        else:
            raise TypeError("C'mon, man...ya can't add a Point and {type(other).__name__}")


    def __rmul__(self, scalar):
        double_and_add(self, scalar)


def create_generator_point(bitcoin_curve):
    print("Creating generator point...")
    G = Point(
        bitcoin_curve,
        x=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
        y=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
    )

    # Verifying that the generator point is indeed on the curve: y^2 = x^3 + 7 (mod p)
    print(
        "    Generator is on the curve: ", (G.y**2 - G.x**3 - 7) % bitcoin_curve.p == 0
    )

    return G


@dataclass
class Generator:
    """
    A generator over a curve: an initial point and the (pre-computed) order
    """

    G: Point  # a generator point on the curve
    n: int  # the order of the generator point, so 0*G = n*G = INF


def create_generator(G):
    print("Creating generator...")
    bitcoin_gen = Generator(
        G=G,
        # the order of G is known and can be mathematically derived
        n=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
    )

    return bitcoin_gen


def create_secret_key(bitcoin_gen):
    # secret_key = random.randrange(1, bitcoin_gen.n)  # truly random
    secret_key = int.from_bytes(b"Andrej is cool :P", "big")  # reproducibly random
    assert 1 <= secret_key < bitcoin_gen.n
    print(f"    Secret key: {secret_key}")

    return secret_key


def preamble():
    print("-" * 88)
    print("Hello from karpathy-tour!")
    print("-" * 88)


def postamble():
    print("-" * 88)
    print("Toodles.")
    print("-" * 88)


INF = Point(None, None, None)  # a special point at infinity, kinda like a zero


def extended_euclidean_algorithm(a, b):
    """
    Returns (gcd, x, y) s.t. a * x + b * y == gcd
    This function implements the extended Euclidean
    algorithm and runs in O(log b) in the worst case,
    taken from Wikipedia.
    """
    old_r, r = a, b
    old_s, s = 1, 0
    old_t, t = 0, 1
    while r != 0:
        quotient = old_r // r
        old_r, r = r, old_r - quotient * r
        old_s, s = s, old_s - quotient * s
        old_t, t = t, old_t - quotient * t

    return old_r, old_s, old_t


def inv(n, p):
    """ returns modular multiplicate inverse m such that (n * m) % p == 1 """
    gcd, x, y = extended_euclidean_algorithm(n, p)

    return x % p


def elliptic_curve_addition(self, other: Point) -> Point:
    # handle special case of P + 0 = 0 + P = 0
    if self == INF:
        return other
    if other == INF:
        return self
    # handle special case of P + (-P) = 0
    if self.x == other.x and self.y != other.y:
        return INF
    # compute the "slope"
    if self.x == other.x: # (self.y = other.y is guaranteed too per above check)
        m = (3 * self.x**2 + self.curve.a) * inv(2 * self.y, self.curve.p)
    else:
        m = (self.y - other.y) * inv(self.x - other.x, self.curve.p)
    # compute the new point
    rx = (m**2 - self.x - other.x) % self.curve.p
    ry = (-(m*(rx - self.x) + self.y)) % self.curve.p

    return Point(self.curve, rx, ry)


def check_keypair(secret_key, private_key, bitcoin_curve):
    print(f'    secret key : {secret_key}')
    print(f'    private key: {private_key}')
    on_curve = (private_key.y**2 - private_key.x**3 - 7) % bitcoin_curve.p == 0
    print(f'    on curve?  : {on_curve}')


def recreate_keypair_examples(G, bitcoin_curve):
    print('Recreating keypair examples...')
    check_keypair(1, G, bitcoin_curve)
    check_keypair(2, G + G, bitcoin_curve)
    check_keypair(3, G + G + G, bitcoin_curve)


def double_and_add(self, k: int) -> Point:
    assert isinstance(k, int) and k >= 0
    result = INF
    append = self
    while k:
        if k & 1:
            result += append
        append += append
        k >>= 1

    return result


def recreate_more_efficient_examples(G):
    print('Recreating more efficient examples...')
    # Verify the redefinition of __rmul__ worked as intended
    # TODO:  This assertion fails...it should not.
    assert G == 1 * G


def main():
    preamble()
    bitcoin_curve = create_bitcoin_curve()
    G = create_generator_point(bitcoin_curve)
    bitcoin_gen = create_generator(G)
    create_secret_key(bitcoin_gen)
    recreate_keypair_examples(G, bitcoin_curve)
    recreate_more_efficient_examples(G)
    postamble()


if __name__ == "__main__":
    main()
