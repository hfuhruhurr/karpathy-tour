from dataclasses import dataclass

@dataclass
class Curve:
    """
    Elliptic Curve over
    """
    p: int # the prime modulus
    a: int
    b: int

    
@dataclass
class Point:
    """ An integer point (x, y) on a Curve """
    curve: Curve
    x: int
    y: int


bitcoin_curve = Curve(
    p = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFC2F,
    a = 0x0000000000000000000000000000000000000000000000000000000000000000, # a = 0
    b = 0x0000000000000000000000000000000000000000000000000000000000000007, # b = 7
)

G = Point(
    bitcoin_curve,
    x = 0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
    y = 0x483ada7726a3c4655da4fbfc0e1108a8fd17b448a68554199c47d08ffb10d4b8,
)

# Verifying that the generator point is indeed on the curve: y^2 = x^3 + 7 (mod p)
print("Generator is on the curve: ", (G.y**2 - G.x**3 - 7) % bitcoin_curve.p == 0)

@dataclass
class Generator:
    """
    A generator over a curve: an initial point and the (pre-computed) order
    """
    G: Point  # a generator point on the curve
    n: int    # the order of the generator point, so 0*G = n*G = INF

bitcoin_gen = Generator(
    G = G,
    # the order of G is known and can be mathematically derived
    n = 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
)

def main():
    print("Hello from karpathy-tour!")


if __name__ == "__main__":
    main()
