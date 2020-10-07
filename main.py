import math
from matplotlib import pylab
import numpy
from prettytable import PrettyTable
import time
from datetime import timedelta

function_test0 = lambda x: -5 * x ** 5 + 4 * x ** 4 - 12 * x ** 3 + 11 * x ** 2 - 2 * x + 1
function_test1 = lambda x: math.log10(x - 2) ** 2 + math.log10(10 - x) ** 2 - x ** 0.2
function_test2 = lambda x: -3.0 * x * math.sin(0.75 * x) + math.exp(-2.0 * x)
function_test3 = lambda x: math.exp(3.0 * x) + 5.0 * math.exp(-2.0 * x)
function_test4 = lambda x: 0.2 * x * math.log10(x) + (x - 2.3) ** 2
# func_first = lambda x: 6 * x ** 2 - 6 * x - 12

# def func_first(x):
#   return 6*x**2-6*x-12

# a1, b1 = -0.5, 0.5
a1, b1 = 6.0, 9.9
e = 0.001

list_function = [(function_test0, -0.5, 0.5), (function_test1, 6.0, 9.9), (function_test2, 0.0, 2.0 * math.pi),
                 (function_test3, 0.0, 1.0), (function_test4, 0.5, 2.5)]

times = 1000000


def check_time(f, times):
    start_time = time.monotonic()
    for i in range(0, times):
        f;
    return timedelta(seconds=time.monotonic() - start_time)


def Dichotomy_method(a, b, f):
    i = 1
    while math.fabs(a - b) >= e:
        x = (a + b) / 2
        if f(x - e) < f(x + e):
            a, b = a, x
        else:
            a, b = x, b
        table.add_row([i, a, b, x, a - b])
        i += 1
    return x


def Golden_Section_method(a, b, f):
    i = 1
    phi = 2 - (1 + math.sqrt(5)) * 0.5

    x1 = a + phi * (b - a)
    x2 = b - phi * (b - a)
    f1 = f(x1)
    f2 = f(x2)

    while math.fabs(a - b) >= e:

        if f1 < f2:
            b = x2
            x2 = x1
            f2 = f1
            x1 = a + phi * (b - a)
            f1 = f(x1)
            # a, b, x2, f2, x1, f1 =a, x2, x1, f1, a+phi*(b-a), f(x1)
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - phi * (b - a)
            f2 = f(x2)
            # a, b, x1,f1, x2, f2 = x1, b, x2, f2, b-phi*(b-a), f(x2)
        table.add_row([i, a, b, (x1 + x2) * 0.5, a - b])
        i += 1
    return (x1 + x2) * 0.5


def fibonacci(n):
    fib = 1
    if n > 2:
        fib = fibonacci(n - 1) + fibonacci(n - 2)
    return fib


def Fibonacci_method(a, b, f):
    i = 1
    n = 0
    while fibonacci(n) < ((b - a) / e):
        n += 1
    # n=10
    x1 = a + (fibonacci(n - 2) / fibonacci(n)) * (b - a)
    y1 = f(x1)
    x2 = a + (fibonacci(n - 1) / fibonacci(n)) * (b - a)
    y2 = f(x2)

    for k in range(0, n - 2):
        if y1 <= y2:
            b = x2
            x2 = x1
            y2 = y1
            x1 = a + (fibonacci(n - k - 3) / fibonacci(n - k - 1)) * (b - a)
            y1 = f(x1)
        else:
            a = x1
            x1 = x2
            y1 = y2
            x2 = a + (fibonacci(n - k - 2) / fibonacci(n - k - 1)) * (b - a)
            y2 = f(x2)
        #       x2=x1+e
        #      y2=f(x2)
        table.add_row([k + 1, a, b, (a + b) * 0.5, a - b])
    #    x2 = x1 + e
    #    y2 = f(x2)
    #    if y1<=y2:
    #        b=y1
    #    else:
    #        a=x1
    return (a + b) * 0.5


def parabolic_method(a, b, f):
    i = 1

    x1 = a
    x2 = (a + b) * 0.5
    x3 = b
    f1 = f(x1)
    f2 = f(x2)
    f3 = f(x3)

    xm = 0.5 * (x1 + x2 - (f2 - f1) * (x3 - x2) / (x2 - x1) / ((f3 - f1) / (x3 - x1) - (f2 - f1) / (x2 - x1)))
    # fm=f(xm)
    last = b
    while math.fabs(last - xm) >= e:
        fm = f(xm)
        if xm < x2:
            x1 = xm
            f1 = fm
        else:
            x1 = x2
            x2 = xm
            f1 = f2
            f2 = fm
        last = xm
        xm = 0.5 * (x1 + x2 - (f2 - f1) * (x3 - x2) / (x2 - x1) / ((f3 - f1) / (x3 - x1) - (f2 - f1) / (x2 - x1)))
        table.add_row([i, x1, x3, xm, x3 - x1])
        i += 1
    return xm


def parabolic_method1(a, b, f):
    i = 1

    x1 = a
    x2 = (a + b) * 0.5
    x3 = b
    f1 = f(x1);
    f3 = f(x3);
    f2 = f(x2);

    while math.fabs(x3 - x1) > e:
        xm = x2 + 0.5 * ((x3 - x2) * (x3 - x2) * (f1 - f2) - (x2 - x1) * (x2 - x1) * (f3 - f2)) / (
                    (x3 - x2) * (f1 - f2) + (x2 - x1) * (f3 - f2))
        if xm == x2:
            t = (x1 + x2) / 2.0
        else:
            t = xm
        yt = f(t)
        if t < x2:
            if yt < f2:
                x3 = x2
                f3 = f2
                x2 = t
                f2 = yt
            elif yt > f2:
                x1 = t
                f1 = yt
            else:
                x1 = t
                f1 = yt
                x3 = x2
                f3 = f2
                x2 = (x1 + x3) / 2
                f2 = f(x2)
        elif t > x2:
            if yt < f2:
                x1 = x2
                f1 = f2
                x2 = t
                f2 = yt
            elif yt > f2:
                x3 = t
                f3 = yt
            else:
                x1 = x2
                f1 = f2
                x3 = t
                f3 = yt
                x2 = (x1 + x3) / 2
                f2 = f(x2)
        i+=1
        table.add_row([i, x1, x3, xm, x3 - x1])
    x = (x1 + x3) / 2;
    y = f(x);

    return x


foo = list_function[4]
a1 = foo[1]
b1 = foo[2]
function_test = foo[0]

table = PrettyTable()
table.field_names = ["Iteration", "a", "b", "x", "a-b"]

X = numpy.arange(a1, b1, 0.1)
pylab.plot([x for x in X], [function_test(x) for x in X])
pylab.grid(True)
pylab.show()

# start_time=time.time()
print('\nlocal minimum of the function by Dichotomy method %s' % Dichotomy_method(a1, b1, function_test))
print(table)
print('\n Method executed %d times, elapsed time %s' % (
times, check_time(Dichotomy_method(a1, b1, function_test), times)))
table.clear_rows()

print('\nlocal minimum of the function by Golden Section method %s' % Golden_Section_method(a1, b1, function_test))
print(table)
print('\n Method executed %d times, elapsed time %s' % (
times, check_time(Golden_Section_method(a1, b1, function_test), times)))
table.clear_rows()

print('\nlocal minimum of the function by Fibonacci method %s' % Fibonacci_method(a1, b1, function_test))
print(table)
print('\n Method executed %d times, elapsed time %s' % (
times, check_time(Fibonacci_method(a1, b1, function_test), times)))
table.clear_rows()

print('\nlocal minimum of the function by Parabolic method %s' % parabolic_method1(a1, b1, function_test))
print(table)
print('\n Method executed %d times, elapsed time %s' % (
times, check_time(parabolic_method1(a1, b1, function_test), times)))
table.clear_rows()
