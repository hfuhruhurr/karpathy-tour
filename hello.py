from dataclasses import dataclass
import random
from typing import Type

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


def create_bitcoin_curve() -> Curve:
    # secp256k1 uses a = 0, b = 7, so we're dealing with the curve y^2 = x^3 + 7 (mod p)

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
        return double_and_add(self, scalar)


def create_generator_point(bitcoin_curve: Curve) -> Point:
    print("Creating the bitcoin generator point...")
    G = Point(
        bitcoin_curve,
        x=0x79BE667EF9DCBBAC55A06295CE870B07029BFCDB2DCE28D959F2815B16F81798,
        y=0x483ADA7726A3C4655DA4FBFC0E1108A8FD17B448A68554199C47D08FFB10D4B8,
    )

    # Verifying that the generator point is indeed on the curve: y^2 = x^3 + 7 (mod p)
    print(
        "    Generator is on the curve: ", verify_point_on_curve(G, bitcoin_curve)
    )

    return G


@dataclass
class Generator:
    """
    A generator over a curve: an initial point and the (pre-computed) order
    """

    G: Point  # a generator point on the curve
    n: int  # the order of the generator point, so 0*G = n*G = INF


def create_the_bitcoin_generator(the_bitcoin_generator_point: Point) -> Generator:
    print("Creating the bitcoin generator...")
    return Generator(
        G=the_bitcoin_generator_point,
        # the order of G is known and can be mathematically derived
        n=0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEBAAEDCE6AF48A03BBFD25E8CD0364141,
    )


def create_secret_key(bitcoin_gen: Generator) -> int:
    # secret_key = random.randrange(1, bitcoin_gen.n)  # truly random
    secret_key = int.from_bytes(b"Andrej is cool :P", "big")  # reproducibly random
    assert 1 <= secret_key < bitcoin_gen.n
    print(f"    Secret key: {secret_key}")

    return secret_key


def preamble() -> None:
    print("-" * 88)
    print("Hello from karpathy-tour!")
    print("-" * 88)


def postamble() -> None:
    print("-" * 88)
    print("Toodles.")
    print("-" * 88)


INF = Point(None, None, None)  # a special point at infinity, kinda like a zero


def extended_euclidean_algorithm(a: int, b: int) -> int:
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


def inv(n:int , p: int) -> int:
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


def check_keypair(secret_key: int, public_key: Point, bitcoin_curve: Curve) -> None:
    print(f'    secret key : {secret_key}')
    print(f'    public key: {public_key}')
    on_curve = verify_point_on_curve(public_key, bitcoin_curve)
    print(f'    on curve?  : {on_curve}')


def recreate_keypair_examples(G: Point, bitcoin_curve: Curve) -> None:
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


def recreate_more_efficient_examples(G: Point) -> None:
    print('Recreating more efficient examples...')
    # Verify the redefinition of __rmul__ worked as intended
    # TODID:  This assertion fails...it should not.
    # I forgot the "return" in __rmult__ ...d'oh!
    assert G == 1 * G
    assert G + G == 2 * G
    assert G + G + G == 3 * G


def verify_point_on_curve(point: Point, curve: Curve) -> bool:
    return (point.y**2 - point.x**3 - 7) % curve.p == 0


def generate_public_key(secret_key: int, bitcoin_generator_point: Point, bitcoin_curve: Curve) -> Point:
    # efficiently calculate our actual public key!
    print("Generating public key from secret_key...")
    public_key = secret_key * bitcoin_generator_point
    print(f"    x: {public_key.x}")
    print(f"    y: {public_key.y}")
    print("    Verify the public key is on the curve: ", verify_point_on_curve(public_key, bitcoin_curve))

    return public_key


def generate_sha256_hash(b: bytes) -> bytes:
    """
    SHA256 implementation to achieve self-imposed goal of having no dependencies. 
    (`hashlib` is more suited for this but it's a dependency.)

    Follows the FIPS PUB 180-4 description for calculating SHA-256 hash function
    https://nvlpubs.nist.gov/nistpubs/FIPS/NIST.FIPS.180-4.pdf

    Noone in their right mind should use this for any serious reason. This was written
    purely for educational purposes.

    The reason I wanted to implement this from scratch and paste it here is that I want 
    you to note that again there is nothing too scary going on inside. SHA256 takes 
    some bytes message that is to be hashed, it first pads the message, then breaks it 
    up into chunks, and passes these chunks into what can best be described as a fancy 
    “bit mixer”, defined in section 3, that contains a number of bit shifts and binary 
    operations orchestrated in a way that is frankly beyond me, but that results in the 
    beautiful properties that SHA256 offers. In particular, it creates a fixed-sized, 
    random-looking short digest of any variably-sized original message s.t. the 
    scrambling is not invertible and also it is basically computationally impossible to 
    construct a different message that hashes to any given digest.
    """

    import math
    from itertools import count, islice

    # -----------------------------------------------------------------------------
    # SHA-256 Functions, defined in Section 4

    def rotr(x: int, n: int, size:int =32) -> int:
        # Rotate right x by n bits (ie, circular right shift)
        # From Section 3.2.4
        return (x >> n) | (x << size - n) & (2**size - 1)

    def shr(x: int, n: int) -> int:
        # Shift right x by n bits
        # From Section 3.2.3
        return x >> n

    def sig0(x: int) -> int:
        # From Section 4.1.2, Formula 4.6
        return rotr(x, 7) ^ rotr(x, 18) ^ shr(x, 3)

    def sig1(x: int) -> int:
        # From Section 4.1.2, Formula 4.7
        return rotr(x, 17) ^ rotr(x, 19) ^ shr(x, 10)

    def capsig0(x: int) -> int:
        # From Section 4.1.2, Formula 4.4
        return rotr(x, 2) ^ rotr(x, 13) ^ rotr(x, 22)

    def capsig1(x: int) -> int:
        # From Section 4.1.2, Formula 4.5
        return rotr(x, 6) ^ rotr(x, 11) ^ rotr(x, 25)

    def ch(x: int, y: int, z: int) -> int:
        # From Section 4.1.2, Formula 4.2
        return (x & y)^ (~x & z)

    def maj(x: int, y: int, z: int) -> int:
        # From Section 4.1.2, Formula 4.3
        return (x & y) ^ (x & z) ^ (y & z)

    def b2i(b: bytes) -> int:
        return int.from_bytes(b, 'big')

    def i2b(i: int) -> bytes:
        return i.to_bytes(4, 'big')


    # -----------------------------------------------------------------------------
    # SHA-256 Constants

    def is_prime(n: int) -> bool:
        return not any(f for f in range(2, int(math.sqrt(n)) + 1) if n % f == 0)

    def first_n_primes(n: int) -> list[int]:
        return islice(filter(is_prime, count(start=2)), n)

    def frac_bin(f: int, n: int = 32) -> int:
        """ return the first n bits of fractional part of float f """
        f -= math.floor(f) # get only the fractional part
        f *= 2**n # shift left
        f = int(f) # truncate the rest of the fractional content
        return f

    def genK() -> list[int]:
        """
        Follows Section 4.2.2 to generate K

        The first 32 bits of the fractional parts of the cube roots of the first
        64 prime numbers:

        428a2f98 71374491 b5c0fbcf e9b5dba5 3956c25b 59f111f1 923f82a4 ab1c5ed5
        d807aa98 12835b01 243185be 550c7dc3 72be5d74 80deb1fe 9bdc06a7 c19bf174
        e49b69c1 efbe4786 0fc19dc6 240ca1cc 2de92c6f 4a7484aa 5cb0a9dc 76f988da
        983e5152 a831c66d b00327c8 bf597fc7 c6e00bf3 d5a79147 06ca6351 14292967
        27b70a85 2e1b2138 4d2c6dfc 53380d13 650a7354 766a0abb 81c2c92e 92722c85
        a2bfe8a1 a81a664b c24b8b70 c76c51a3 d192e819 d6990624 f40e3585 106aa070
        19a4c116 1e376c08 2748774c 34b0bcb5 391c0cb3 4ed8aa4a 5b9cca4f 682e6ff3
        748f82ee 78a5636f 84c87814 8cc70208 90befffa a4506ceb bef9a3f7 c67178f2
        """
        return [frac_bin(p ** (1/3.0)) for p in first_n_primes(64)]

    def genH() -> list[int]:
        """
        Follows Section 5.3.3 to generate the initial hash value H^0

        The first 32 bits of the fractional parts of the square roots of
        the first 8 prime numbers.

        6a09e667 bb67ae85 3c6ef372 a54ff53a 9b05688c 510e527f 1f83d9ab 5be0cd19
        """
        return [frac_bin(p ** (1/2.0)) for p in first_n_primes(8)]

    # -----------------------------------------------------------------------------
    
    def pad(b: bytes) -> bytes:
        """ 
        Follows Section 5.1: Padding the message 
        
        The purpose of this padding is to ensure that the padded message is a 
        multiple of 512 or 1024 bits, depending on the algorithm.
        """
        b = bytearray(b) # convert to a mutable equivalent
        l = len(b) * 8 # note: len returns number of bytes not bits

        # append but "1" to the end of the message
        b.append(0b10000000) # appending 10000000 in binary (=128 in decimal)

        # follow by k zero bits, where k is the smallest non-negative solution to
        # l + 1 + k = 448 mod 512
        # i.e. pad with zeros until we reach 448 (mod 512)
        while (len(b)*8) % 512 != 448:
            b.append(0x00)

        # the last 64-bit block is the length l of the original message
        # expressed in binary (big endian)
        b.extend(l.to_bytes(8, 'big'))

        return b
    
    # Section 4.2
    K = genK()
    
    # Section 5: Preprocessing
    # Section 5.1: Pad the message
    b = pad(b)
    # Section 5.2: Separate the message into blocks of 512 bits (64 bytes)
    blocks = [b[i:i+64] for i in range(0, len(b), 64)]
    # for each message block M^1 ... M^N
    H = genH() # Section 5.3
    
    # Section 6
    for M in blocks: # each block is a 64-entry array of 8-bit bytes
        # 1. Prepare the message schedule, a 64-entry array of 32-bit words
        W = []
        for t in range(64):
            if t <= 15:
                # the first 16 words are just a copy of the block
                W.append(bytes(M[t*4:t*4+4]))
            else:
                term1 = sig1(b2i(W[t-2]))
                term2 = b2i(W[t-7])
                term3 = sig0(b2i(W[t-15]))
                term4 = b2i(W[t-16])
                total = (term1 + term2 + term3 + term4) % 2**32
                W.append(i2b(total))
        # 2. Initialize the 8 working variables a,b,c,d,e,f,g,h with prev hash value
        a, b, c, d, e, f, g, h = H
        # 3.
        for t in range(64):
            T1 = (h + capsig1(e) + ch(e, f, g) + K[t] + b2i(W[t])) % 2**32
            T2 = (capsig0(a) + maj(a, b, c)) % 2**32
            h = g
            g = f
            f = e
            e = (d + T1) % 2**32
            d = c
            c = b
            b = a
            a = (T1 + T2) % 2**32
        # 4. Compute the i-th intermediate hash value H^i
        delta = [a, b, c, d, e, f, g, h]
        H = [(i1 + i2) % 2**32 for i1, i2 in zip(H, delta)]
    
    return b''.join(i2b(i) for i in H)


def generate_ripemd160_hash(b: bytes) -> bytes:
    """
    RIPEMD-160 implementation to achieve self-imposed goal of having no dependencies. 
    (`hashlib` is more suited for this but it's a dependency.)
    """

    import sys
    import struct

    # -----------------------------------------------------------------------------

    class RMDContext:
        def __init__(self):
            self.state = [0x67452301, 0xEFCDAB89, 0x98BADCFE, 0x10325476, 0xC3D2E1F0] # uint32
            self.count = 0 # uint64
            self.buffer = [0]*64 # uchar

    def RMD160Update(ctx: RMDContext, inp: list[int], inplen: int) -> None:
        have = int((ctx.count // 8) % 64)
        inplen = int(inplen)
        need = 64 - have
        ctx.count += 8 * inplen
        off = 0
        if inplen >= need:
            if have:
                for i in range(need):
                    ctx.buffer[have+i] = inp[i]
                RMD160Transform(ctx.state, ctx.buffer)
                off = need
                have = 0
            while off + 64 <= inplen:
                RMD160Transform(ctx.state, inp[off:])
                off += 64
        if off < inplen:
            for i in range(inplen - off):
                ctx.buffer[have+i] = inp[off+i]

    def RMD160Final(ctx: RMDContext) -> bytes:
        size = struct.pack("<Q", ctx.count)
        padlen = 64 - ((ctx.count // 8) % 64)
        if padlen < 1 + 8:
            padlen += 64
        RMD160Update(ctx, PADDING, padlen-8)
        RMD160Update(ctx, size, 8)
        return struct.pack("<5L", *ctx.state)

    # -----------------------------------------------------------------------------

    K0 = 0x00000000
    K1 = 0x5A827999
    K2 = 0x6ED9EBA1
    K3 = 0x8F1BBCDC
    K4 = 0xA953FD4E
    KK0 = 0x50A28BE6
    KK1 = 0x5C4DD124
    KK2 = 0x6D703EF3
    KK3 = 0x7A6D76E9
    KK4 = 0x00000000

    PADDING = [0x80] + [0]*63

    def ROL(n: int, x: int) -> int:
        return ((x << n) & 0xffffffff) | (x >> (32 - n))

    def F0(x: int, y: int, z: int) -> int:
        return x ^ y ^ z

    def F1(x: int, y: int, z: int) -> int:
        return (x & y) | (((~x) % 0x100000000) & z)

    def F2(x: int, y: int, z: int) -> int:
        return (x | ((~y) % 0x100000000)) ^ z

    def F3(x: int, y: int, z: int) -> int:
        return (x & z) | (((~z) % 0x100000000) & y)

    def F4(x: int, y: int, z: int) -> int:
        return x ^ (y | ((~z) % 0x100000000))

    def R(a: int, b: int, c: int, d: int, e: int, Fj: int, Kj: int, sj: int, rj: int, X: list[int]) -> tuple[int, int]:
        a = ROL(sj, (a + Fj(b, c, d) + X[rj] + Kj) % 0x100000000) + e
        c = ROL(10, c)
        return a % 0x100000000, c

    def RMD160Transform(state: list[int], block: list[int]) -> bytes: #uint32 state[5], uchar block[64]

        x = [0]*16
        assert sys.byteorder == 'little', "Only little endian is supported atm for RIPEMD160"
        x = struct.unpack('<16L', bytes(block[0:64]))

        a = state[0]
        b = state[1]
        c = state[2]
        d = state[3]
        e = state[4]

        #/* Round 1 */
        a, c = R(a, b, c, d, e, F0, K0, 11,  0, x)
        e, b = R(e, a, b, c, d, F0, K0, 14,  1, x)
        d, a = R(d, e, a, b, c, F0, K0, 15,  2, x)
        c, e = R(c, d, e, a, b, F0, K0, 12,  3, x)
        b, d = R(b, c, d, e, a, F0, K0,  5,  4, x)
        a, c = R(a, b, c, d, e, F0, K0,  8,  5, x)
        e, b = R(e, a, b, c, d, F0, K0,  7,  6, x)
        d, a = R(d, e, a, b, c, F0, K0,  9,  7, x)
        c, e = R(c, d, e, a, b, F0, K0, 11,  8, x)
        b, d = R(b, c, d, e, a, F0, K0, 13,  9, x)
        a, c = R(a, b, c, d, e, F0, K0, 14, 10, x)
        e, b = R(e, a, b, c, d, F0, K0, 15, 11, x)
        d, a = R(d, e, a, b, c, F0, K0,  6, 12, x)
        c, e = R(c, d, e, a, b, F0, K0,  7, 13, x)
        b, d = R(b, c, d, e, a, F0, K0,  9, 14, x)
        a, c = R(a, b, c, d, e, F0, K0,  8, 15, x) #/* #15 */
        #/* Round 2 */
        e, b = R(e, a, b, c, d, F1, K1,  7,  7, x)
        d, a = R(d, e, a, b, c, F1, K1,  6,  4, x)
        c, e = R(c, d, e, a, b, F1, K1,  8, 13, x)
        b, d = R(b, c, d, e, a, F1, K1, 13,  1, x)
        a, c = R(a, b, c, d, e, F1, K1, 11, 10, x)
        e, b = R(e, a, b, c, d, F1, K1,  9,  6, x)
        d, a = R(d, e, a, b, c, F1, K1,  7, 15, x)
        c, e = R(c, d, e, a, b, F1, K1, 15,  3, x)
        b, d = R(b, c, d, e, a, F1, K1,  7, 12, x)
        a, c = R(a, b, c, d, e, F1, K1, 12,  0, x)
        e, b = R(e, a, b, c, d, F1, K1, 15,  9, x)
        d, a = R(d, e, a, b, c, F1, K1,  9,  5, x)
        c, e = R(c, d, e, a, b, F1, K1, 11,  2, x)
        b, d = R(b, c, d, e, a, F1, K1,  7, 14, x)
        a, c = R(a, b, c, d, e, F1, K1, 13, 11, x)
        e, b = R(e, a, b, c, d, F1, K1, 12,  8, x) #/* #31 */
        #/* Round 3 */
        d, a = R(d, e, a, b, c, F2, K2, 11,  3, x)
        c, e = R(c, d, e, a, b, F2, K2, 13, 10, x)
        b, d = R(b, c, d, e, a, F2, K2,  6, 14, x)
        a, c = R(a, b, c, d, e, F2, K2,  7,  4, x)
        e, b = R(e, a, b, c, d, F2, K2, 14,  9, x)
        d, a = R(d, e, a, b, c, F2, K2,  9, 15, x)
        c, e = R(c, d, e, a, b, F2, K2, 13,  8, x)
        b, d = R(b, c, d, e, a, F2, K2, 15,  1, x)
        a, c = R(a, b, c, d, e, F2, K2, 14,  2, x)
        e, b = R(e, a, b, c, d, F2, K2,  8,  7, x)
        d, a = R(d, e, a, b, c, F2, K2, 13,  0, x)
        c, e = R(c, d, e, a, b, F2, K2,  6,  6, x)
        b, d = R(b, c, d, e, a, F2, K2,  5, 13, x)
        a, c = R(a, b, c, d, e, F2, K2, 12, 11, x)
        e, b = R(e, a, b, c, d, F2, K2,  7,  5, x)
        d, a = R(d, e, a, b, c, F2, K2,  5, 12, x) #/* #47 */
        #/* Round 4 */
        c, e = R(c, d, e, a, b, F3, K3, 11,  1, x)
        b, d = R(b, c, d, e, a, F3, K3, 12,  9, x)
        a, c = R(a, b, c, d, e, F3, K3, 14, 11, x)
        e, b = R(e, a, b, c, d, F3, K3, 15, 10, x)
        d, a = R(d, e, a, b, c, F3, K3, 14,  0, x)
        c, e = R(c, d, e, a, b, F3, K3, 15,  8, x)
        b, d = R(b, c, d, e, a, F3, K3,  9, 12, x)
        a, c = R(a, b, c, d, e, F3, K3,  8,  4, x)
        e, b = R(e, a, b, c, d, F3, K3,  9, 13, x)
        d, a = R(d, e, a, b, c, F3, K3, 14,  3, x)
        c, e = R(c, d, e, a, b, F3, K3,  5,  7, x)
        b, d = R(b, c, d, e, a, F3, K3,  6, 15, x)
        a, c = R(a, b, c, d, e, F3, K3,  8, 14, x)
        e, b = R(e, a, b, c, d, F3, K3,  6,  5, x)
        d, a = R(d, e, a, b, c, F3, K3,  5,  6, x)
        c, e = R(c, d, e, a, b, F3, K3, 12,  2, x) #/* #63 */
        #/* Round 5 */
        b, d = R(b, c, d, e, a, F4, K4,  9,  4, x)
        a, c = R(a, b, c, d, e, F4, K4, 15,  0, x)
        e, b = R(e, a, b, c, d, F4, K4,  5,  5, x)
        d, a = R(d, e, a, b, c, F4, K4, 11,  9, x)
        c, e = R(c, d, e, a, b, F4, K4,  6,  7, x)
        b, d = R(b, c, d, e, a, F4, K4,  8, 12, x)
        a, c = R(a, b, c, d, e, F4, K4, 13,  2, x)
        e, b = R(e, a, b, c, d, F4, K4, 12, 10, x)
        d, a = R(d, e, a, b, c, F4, K4,  5, 14, x)
        c, e = R(c, d, e, a, b, F4, K4, 12,  1, x)
        b, d = R(b, c, d, e, a, F4, K4, 13,  3, x)
        a, c = R(a, b, c, d, e, F4, K4, 14,  8, x)
        e, b = R(e, a, b, c, d, F4, K4, 11, 11, x)
        d, a = R(d, e, a, b, c, F4, K4,  8,  6, x)
        c, e = R(c, d, e, a, b, F4, K4,  5, 15, x)
        b, d = R(b, c, d, e, a, F4, K4,  6, 13, x) #/* #79 */

        aa = a
        bb = b
        cc = c
        dd = d
        ee = e

        a = state[0]
        b = state[1]
        c = state[2]
        d = state[3]
        e = state[4]

        #/* Parallel round 1 */
        a, c = R(a, b, c, d, e, F4, KK0,  8,  5, x)
        e, b = R(e, a, b, c, d, F4, KK0,  9, 14, x)
        d, a = R(d, e, a, b, c, F4, KK0,  9,  7, x)
        c, e = R(c, d, e, a, b, F4, KK0, 11,  0, x)
        b, d = R(b, c, d, e, a, F4, KK0, 13,  9, x)
        a, c = R(a, b, c, d, e, F4, KK0, 15,  2, x)
        e, b = R(e, a, b, c, d, F4, KK0, 15, 11, x)
        d, a = R(d, e, a, b, c, F4, KK0,  5,  4, x)
        c, e = R(c, d, e, a, b, F4, KK0,  7, 13, x)
        b, d = R(b, c, d, e, a, F4, KK0,  7,  6, x)
        a, c = R(a, b, c, d, e, F4, KK0,  8, 15, x)
        e, b = R(e, a, b, c, d, F4, KK0, 11,  8, x)
        d, a = R(d, e, a, b, c, F4, KK0, 14,  1, x)
        c, e = R(c, d, e, a, b, F4, KK0, 14, 10, x)
        b, d = R(b, c, d, e, a, F4, KK0, 12,  3, x)
        a, c = R(a, b, c, d, e, F4, KK0,  6, 12, x) #/* #15 */
        #/* Parallel round 2 */
        e, b = R(e, a, b, c, d, F3, KK1,  9,  6, x)
        d, a = R(d, e, a, b, c, F3, KK1, 13, 11, x)
        c, e = R(c, d, e, a, b, F3, KK1, 15,  3, x)
        b, d = R(b, c, d, e, a, F3, KK1,  7,  7, x)
        a, c = R(a, b, c, d, e, F3, KK1, 12,  0, x)
        e, b = R(e, a, b, c, d, F3, KK1,  8, 13, x)
        d, a = R(d, e, a, b, c, F3, KK1,  9,  5, x)
        c, e = R(c, d, e, a, b, F3, KK1, 11, 10, x)
        b, d = R(b, c, d, e, a, F3, KK1,  7, 14, x)
        a, c = R(a, b, c, d, e, F3, KK1,  7, 15, x)
        e, b = R(e, a, b, c, d, F3, KK1, 12,  8, x)
        d, a = R(d, e, a, b, c, F3, KK1,  7, 12, x)
        c, e = R(c, d, e, a, b, F3, KK1,  6,  4, x)
        b, d = R(b, c, d, e, a, F3, KK1, 15,  9, x)
        a, c = R(a, b, c, d, e, F3, KK1, 13,  1, x)
        e, b = R(e, a, b, c, d, F3, KK1, 11,  2, x) #/* #31 */
        #/* Parallel round 3 */
        d, a = R(d, e, a, b, c, F2, KK2,  9, 15, x)
        c, e = R(c, d, e, a, b, F2, KK2,  7,  5, x)
        b, d = R(b, c, d, e, a, F2, KK2, 15,  1, x)
        a, c = R(a, b, c, d, e, F2, KK2, 11,  3, x)
        e, b = R(e, a, b, c, d, F2, KK2,  8,  7, x)
        d, a = R(d, e, a, b, c, F2, KK2,  6, 14, x)
        c, e = R(c, d, e, a, b, F2, KK2,  6,  6, x)
        b, d = R(b, c, d, e, a, F2, KK2, 14,  9, x)
        a, c = R(a, b, c, d, e, F2, KK2, 12, 11, x)
        e, b = R(e, a, b, c, d, F2, KK2, 13,  8, x)
        d, a = R(d, e, a, b, c, F2, KK2,  5, 12, x)
        c, e = R(c, d, e, a, b, F2, KK2, 14,  2, x)
        b, d = R(b, c, d, e, a, F2, KK2, 13, 10, x)
        a, c = R(a, b, c, d, e, F2, KK2, 13,  0, x)
        e, b = R(e, a, b, c, d, F2, KK2,  7,  4, x)
        d, a = R(d, e, a, b, c, F2, KK2,  5, 13, x) #/* #47 */
        #/* Parallel round 4 */
        c, e = R(c, d, e, a, b, F1, KK3, 15,  8, x)
        b, d = R(b, c, d, e, a, F1, KK3,  5,  6, x)
        a, c = R(a, b, c, d, e, F1, KK3,  8,  4, x)
        e, b = R(e, a, b, c, d, F1, KK3, 11,  1, x)
        d, a = R(d, e, a, b, c, F1, KK3, 14,  3, x)
        c, e = R(c, d, e, a, b, F1, KK3, 14, 11, x)
        b, d = R(b, c, d, e, a, F1, KK3,  6, 15, x)
        a, c = R(a, b, c, d, e, F1, KK3, 14,  0, x)
        e, b = R(e, a, b, c, d, F1, KK3,  6,  5, x)
        d, a = R(d, e, a, b, c, F1, KK3,  9, 12, x)
        c, e = R(c, d, e, a, b, F1, KK3, 12,  2, x)
        b, d = R(b, c, d, e, a, F1, KK3,  9, 13, x)
        a, c = R(a, b, c, d, e, F1, KK3, 12,  9, x)
        e, b = R(e, a, b, c, d, F1, KK3,  5,  7, x)
        d, a = R(d, e, a, b, c, F1, KK3, 15, 10, x)
        c, e = R(c, d, e, a, b, F1, KK3,  8, 14, x) #/* #63 */
        #/* Parallel round 5 */
        b, d = R(b, c, d, e, a, F0, KK4,  8, 12, x)
        a, c = R(a, b, c, d, e, F0, KK4,  5, 15, x)
        e, b = R(e, a, b, c, d, F0, KK4, 12, 10, x)
        d, a = R(d, e, a, b, c, F0, KK4,  9,  4, x)
        c, e = R(c, d, e, a, b, F0, KK4, 12,  1, x)
        b, d = R(b, c, d, e, a, F0, KK4,  5,  5, x)
        a, c = R(a, b, c, d, e, F0, KK4, 14,  8, x)
        e, b = R(e, a, b, c, d, F0, KK4,  6,  7, x)
        d, a = R(d, e, a, b, c, F0, KK4,  8,  6, x)
        c, e = R(c, d, e, a, b, F0, KK4, 13,  2, x)
        b, d = R(b, c, d, e, a, F0, KK4,  6, 13, x)
        a, c = R(a, b, c, d, e, F0, KK4,  5, 14, x)
        e, b = R(e, a, b, c, d, F0, KK4, 15,  0, x)
        d, a = R(d, e, a, b, c, F0, KK4, 13,  3, x)
        c, e = R(c, d, e, a, b, F0, KK4, 11,  9, x)
        b, d = R(b, c, d, e, a, F0, KK4, 11, 11, x) #/* #79 */

        t = (state[1] + cc + d) % 0x100000000
        state[1] = (state[2] + dd + e) % 0x100000000
        state[2] = (state[3] + ee + a) % 0x100000000
        state[3] = (state[4] + aa + b) % 0x100000000
        state[4] = (state[0] + bb + c) % 0x100000000
        state[0] = t % 0x100000000

    """ simple wrapper for a simpler API to this hash function, just bytes to bytes """
    ctx = RMDContext()
    RMD160Update(ctx, b, len(b))
    digest = RMD160Final(ctx)
    
    return digest
    

def b58encode(b: bytes) -> str:
    # base58 encoding / decoding utilities
    # reference: https://en.bitcoin.it/wiki/Base58Check_encoding

    alphabet = '123456789ABCDEFGHJKLMNPQRSTUVWXYZabcdefghijkmnopqrstuvwxyz'
    
    assert len(b) == 25 # version is 1 byte, pkb_hash 20 bytes, checksum 4 bytes
    n = int.from_bytes(b, 'big')
    chars = []
    while n:
        n, i = divmod(n, 58)
        chars.append(alphabet[i])
    # special case handle the leading 0 bytes... ¯\_(ツ)_/¯
    num_leading_zeros = len(b) - len(b.lstrip(b'\x00'))
    res = num_leading_zeros * alphabet[0] + ''.join(reversed(chars))
    return res


class PublicKey(Point):
    """
    The public key is just a Point on a Curve, but has some additional specific
    encoding / decoding functionality that this class implements.
    """

    @classmethod
    def from_point(cls: Type['PublicKey'], pt: Point) -> 'PublicKey':
        """ promote a Point to be a PublicKey """
        return cls(pt.curve, pt.x, pt.y)

    def encode(self, compressed, hash160=False):
        """ return the SEC bytes encoding of the public key Point """
        # calculate the bytes
        if compressed:
            # (x,y) is very redundant. Because y^2 = x^3 + 7,
            # we can just encode x, and then y = +/- sqrt(x^3 + 7),
            # so we need one more bit to encode whether it was the + or the -
            # but because this is modular arithmetic there is no +/-, instead
            # it can be shown that one y will always be even and the other odd.
            prefix = b'\x02' if self.y % 2 == 0 else b'\x03'
            pkb = prefix + self.x.to_bytes(32, 'big')
        else:
            pkb = b'\x04' + self.x.to_bytes(32, 'big') + self.y.to_bytes(32, 'big')
        # hash if desired
        return generate_ripemd160_hash(generate_sha256_hash(pkb)) if hash160 else pkb

    def address(self, net: str, compressed: bool) -> str:
        """ return the associated bitcoin address for this public key as string """
        # encode the public key into bytes and hash to get the payload
        pkb_hash = self.encode(compressed=compressed, hash160=True)
        # add version byte (0x00 for Main Network, or 0x6f for Test Network)
        version = {'main': b'\x00', 'test': b'\x6f'}
        ver_pkb_hash = version[net] + pkb_hash
        # calculate the checksum
        checksum = generate_sha256_hash(generate_sha256_hash(ver_pkb_hash))[:4]
        # append to form the full 25-byte binary Bitcoin Address
        byte_address = ver_pkb_hash + checksum
        # finally b58 encode the result
        b58check_address = b58encode(byte_address)
        return b58check_address


def verify_shaw() -> None:
    print("Verifying sha256 hash function...")
    print("    verify empty hash:", generate_sha256_hash(b'').hex()) # should be e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
    print("    ", generate_sha256_hash(b'here is a random bytes message, cool right?').hex())
    print("    number of bytes in a sha256 digest: ", len(generate_sha256_hash(b'')))


def verify_ripemd160() -> None:
    print("Verifying ripemd160 hash function...")
    print("    ", generate_ripemd160_hash(b'hello this is a test').hex())
    print("    number of bytes in a RIPEMD-160 digest: ", len(generate_ripemd160_hash(b'')))


def verify_bitcoin_address(address: str) -> None:
    print("Verifying bitcoin address...")
    print("    ", address)
    

def part_one_summary(secret_key: int, public_key: Point, address: str) -> None:
    print("Our first Bitcoin identity:")
    print("    Picking a random number *is* our secret key:")
    print("        secret key      : ", secret_key)
    print("    Adding the generator point to itself secret_key times gives us the public key:")
    print("    (which is just a point on the curve --> it has x and y coordinates)")
    print("        public key.x    : ", public_key.x)
    print("        public key.y    : ", public_key.y)
    print("    After some sha256 and ripemd160 hashing, we Base58 encode to get the bitcoin address:")
    print("        Bitcoin address : ", address)


def main():
    preamble()
    
    # Part 1
    bitcoin_curve = create_bitcoin_curve()
    G = create_generator_point(bitcoin_curve)
    bitcoin_gen = create_the_bitcoin_generator(G)
    secret_key = create_secret_key(bitcoin_gen)
    recreate_keypair_examples(G, bitcoin_curve)
    recreate_more_efficient_examples(G)
    public_key = generate_public_key(secret_key, G, bitcoin_curve)
    verify_shaw()
    verify_ripemd160()    
    address = PublicKey.from_point(public_key).address(net='test', compressed=True)
    verify_bitcoin_address(address)
    part_one_summary(secret_key, public_key, address)
    
    postamble()


if __name__ == "__main__":
    main()
