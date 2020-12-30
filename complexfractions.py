"""
Infinite precision complex numbers with rational real and imaginary components.
"""

# Based on the CPython standard library fractions module.

import fractions
import math
import numbers
import operator

__all__ = ['ComplexFraction', 'J']


def _to_immutable_rational(arg):
    """
    Convert arg into an immutable type in an exact, rational form.  This can
    either be a fraction or an integer.
    """
    if arg is None:
        return 0
    if isinstance(arg, tuple) and len(arg) == 2:
        return fractions.Fraction(arg[0], arg[1])
    if isinstance(arg, (numbers.Integral, fractions.Fraction)):
        # Exact numbers.
        return arg
    if isinstance(arg, numbers.Real):
        return fractions.Fraction(arg)
    if isinstance(arg, numbers.Complex):
        raise TypeError("arg must not be complex")
    raise TypeError("unknown type passed: {}".format(arg))


def _generate_operators(monomorphic, polymorphic, name):
    """
    Generate a pair of forwards and backwards operators for a mathematical
    operation in the ComplexFraction class. `monomorphic` should implement this
    for two ComplexFraction instances, which `polymorphic` should be a general
    Python form that accepts any two `numbers.Complex` instances, and likely
    returns a standard Python type.  `name` is the string name of the function,
    used to generate the '__add__' and similar names.
    """
    def forwards(a, b):
        # a is guaranteed to be a ComplexFraction.
        if isinstance(b, (numbers.Integral, fractions.Fraction)):
            b = ComplexFraction(b, 0)
        if isinstance(b, ComplexFraction):
            return monomorphic(a, b)
        if isinstance(b, numbers.Real) and a._imag == 0:
            return polymorphic(a._real, b)
        if isinstance(b, numbers.Complex):
            return polymorphic(complex(a), b)
        return NotImplemented
    forwards.__name__ = '__' + name + '__'
    forwards.__doc__ = monomorphic.__doc__

    def backwards(b, a):
        # b is guaranteed to be a ComplexFraction.
        if isinstance(a, (numbers.Integral, fractions.Fraction)):
            a = ComplexFraction(a, 0)
        if isinstance(a, ComplexFraction):
            return monomorphic(a, b)
        if isinstance(a, numbers.Real) and b._imag == 0:
            return polymorphic(a, b._real)
        if isinstance(a, numbers.Complex):
            return polymorphic(a, complex(b))
        return NotImplemented
    backwards.__name__ = '__r' + name + '__'
    backwards.__doc__ = monomorphic.__doc__
    return forwards, backwards


def _add(a, b):
    """a + b"""
    return ComplexFraction(a._real + b._real, a._imag + b._imag)


def _sub(a, b):
    """a - b"""
    return ComplexFraction(a._real - b._real, a._imag - b._imag)


def _mul(a, b):
    """a * b"""
    return ComplexFraction(a._real*b._real - a._imag*b._imag,
                           a._real*b._imag + a._imag*b._real)


def _div(a, b):
    """a / b"""
    denom = b._real*b._real + b._imag*b._imag
    if isinstance(denom, fractions.Fraction):
        scale = 1 / denom
    else:
        scale = fractions.Fraction(1, denom)
    real = scale * (a._real*b._real + a._imag*b._imag)
    imag = scale * (a._imag*b._real - a._real*b._imag)
    return ComplexFraction(real, imag)


def _str_real(x):
    """Get a simplified string representation of a real number x."""
    out = str(x)
    return "({})".format(out) if '/' in out else out


class ComplexFraction(numbers.Complex):
    """
    Implements complex numbers in the form `a + b*j` where `a` and `b` are both
    rational numbers, represented by the standard library `fractions.Fraction`
    class.
    """

    __slots__ = ('_real', '_imag')

    def __new__(cls, real=0, imag=None):
        self = super(ComplexFraction, cls).__new__(cls)
        if imag is None and isinstance(real, numbers.Complex):
            if isinstance(real, ComplexFraction):
                # Immutability means we can safely just return the input.
                return real
            real, imag = real.real, real.imag
        self._real = _to_immutable_rational(real)
        self._imag = _to_immutable_rational(imag)
        return self

    @property
    def real(self):
        return self._real

    @property
    def imag(self):
        return self._imag

    def conjugate(self):
        return ComplexFraction(self._real, -self._imag)

    def __bool__(self):
        return bool(self._real) or bool(self._imag)

    def __complex__(self):
        return complex(self._real, self._imag)

    def __abs__(self):
        return math.sqrt(self._real*self._real + self._imag*self._imag)

    def __repr__(self):
        return "{}({}, {})".format(self.__class__.__name__,
                                   repr(self._real),
                                   repr(self._imag))

    def __str__(self):
        if self._imag == 0:
            # no _str_real because it's being returned alone, not with a '+'.
            return str(self._real)
        if self._real == 0:
            return "{}j".format(_str_real(self._imag))
        return "({} + {}j)".format(_str_real(self._real), _str_real(self._imag))

    __add__, __radd__ = _generate_operators(_add, operator.add, 'add')
    __sub__, __rsub__ = _generate_operators(_sub, operator.sub, 'sub')
    __mul__, __rmul__ = _generate_operators(_mul, operator.mul, 'mul')
    __truediv__, __rtruediv__ = _generate_operators(_div, operator.truediv, 'truediv')

    def __neg__(self):
        return ComplexFraction(-self._real, -self._imag)

    def __pos__(self):
        return self

    def __pow__(self, other):
        if isinstance(other, ComplexFraction) and other._imag == 0:
            other = other._real
        if isinstance(other, fractions.Fraction) and other.denominator == 1:
            other = other.numerator
        if not isinstance(other, numbers.Integral):
            # Decays to inexact powers if we've not ended up with an integer.
            return complex(self) ** complex(other)
        if other == 0:
            # Python defines 0**0 == 1, so don't error on that case.
            return 1
        if other < 0:
            if self == 0:
                raise ZeroDivisionError("0 cannot be raised to a negative power")
            base, exp = 1/self, -other
        else:
            base, exp = self, other
        # Use a divide-and-conquer power expansion rather than the binomial
        # expansion; requires O(log(n)) operations instead of O(n).  We avoid
        # using `Fraction` for intermediate calculations to avoid many
        # expensive calls to math.gcd.  This function is still dominated by the
        # call to math.gcd on construction of the two fractions at the end.
        re, im = fractions.Fraction(base._real), fractions.Fraction(base._imag)
        lcm = (
            re.denominator*im.denominator
            // math.gcd(re.denominator, im.denominator)
        )
        lcm_pow = lcm**exp
        re, im = (re*lcm).numerator, (im*lcm).numerator
        pow_re, pow_im = re, im
        out_re, out_im = (pow_re, pow_im) if exp % 2 == 1 else (1, 0)
        exp //= 2
        while exp:
            # a is guaranteed to be ComplexFraction, so bypass type checks.
            pow_re, pow_im = pow_re*pow_re - pow_im*pow_im, 2*pow_re*pow_im
            if exp % 2 == 1:
                out_re, out_im = (
                    out_re*pow_re - out_im*pow_im,
                    out_re*pow_im + out_im*pow_re,
                )
            exp //= 2
        return ComplexFraction(
            fractions.Fraction(out_re, lcm_pow),
            fractions.Fraction(out_im, lcm_pow),
        )

    def __rpow__(self, other):
        if self._imag == 0:
            return other ** self._real
        return other ** complex(self)

    def __eq__(self, other):
        """a == b"""
        if isinstance(other, ComplexFraction):
            return self._real == other._real and self._imag == other._imag
        if isinstance(other, numbers.Real):
            return self._imag == 0 and self._real == other
        if isinstance(other, numbers.Complex):
            return self._real == other.real and self._imag == other.imag
        return NotImplemented

    # Support for pickling and copying operations.

    def __reduce__(self):
        return self.__class__, (self._real, self._imag)

    def __copy__(self):
        if type(self) is ComplexFraction:
            # As ComplexFraction is immutable, self is its own duplicate.
            return self
        return self.__class__(self._real, self._imag)

    def __deepcopy__(self):
        if type(self) is ComplexFraction:
            # ComplexFraction only uses immutable components.
            return self
        return self.__class__(self._real, self._imag)


# Exact version of the 1j constant. Choose J over I to match Python convention.
J = ComplexFraction(0, 1)
