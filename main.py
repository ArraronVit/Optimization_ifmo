# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double⇧ to search everywhere for classes, files, tool windows, actions, and settings.

import math
from matplotlib import pylab
import numpy
from prettytable import PrettyTable
import time
from datetime import timedelta
import matplotlib.pyplot as plt


def check_time(f, times):
    start_time = time.monotonic()
    for i in range(0, times):
        f
    return timedelta(seconds=time.monotonic() - start_time)


def dichotomy_method(a, b, f):
    i = 1
    counter = 0
    while math.fabs(a - b) >= e:
        x = (a + b) / 2
        if f(x - e/2) < f(x + e/2):
            a, b = a, x
        else:
            a, b = x, b
        table.add_row([i, a, b, x, a - b])
        counter += 2
        i += 1
    return x, counter


def golden_section_method(a, b, f):
    i = 1
    counter = 2
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
        else:
            a = x1
            x1 = x2
            f1 = f2
            x2 = b - phi * (b - a)
            f2 = f(x2)
        table.add_row([i, a, b, (x1 + x2) * 0.5, a - b])
        counter += 1
        i += 1
    return (x1 + x2) * 0.5, counter


def fibonacci1(n):
    fib = 1
    if n > 2:
        fib = fibonacci(n - 1) + fibonacci(n - 2)
    return fib


def fibonacci(n):
    return int(((1 + math.sqrt(5)) * 0.5) ** n / math.sqrt(5) + 0.5)


def fibonacci_method(a, b, f):
    counter = 2
    n = 0
    while fibonacci(n) < ((b - a) / e):
        n += 1
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
        table.add_row([k + 1, a, b, (a + b) * 0.5, a - b])
        counter += 1
    return (a + b) * 0.5, counter


def parabolic_method(a, b, f):
    i = 1
    counter = 3
    x1 = a
    x2 = (a + b) * 0.5
    x3 = b
    f1 = f(x1)
    f3 = f(x3)
    f2 = f(x2)

    while math.fabs(x3 - x1) > e:
        try:
            xm = x2 + 0.5 * ((x3 - x2) * (x3 - x2) * (f1 - f2) - (x2 - x1) * (x2 - x1) * (f3 - f2)) / (
                    (x3 - x2) * (f1 - f2) + (x2 - x1) * (f3 - f2))
        except ZeroDivisionError:
            break

        if xm == x2:
            temp = (x1 + x2) / 2.0
        else:
            temp = xm
        f_temp = f(temp)
        counter += 1
        if temp < x2:
            if f_temp < f2:
                x3 = x2
                f3 = f2
                x2 = temp
                f2 = f_temp
            elif f_temp > f2:
                x1 = temp
                f1 = f_temp
            else:
                x1 = temp
                f1 = f_temp
                x3 = x2
                f3 = f2
                x2 = (x1 + x3) / 2
                f2 = f(x2)
                counter += 1
        elif temp > x2:
            if f_temp < f2:
                x1 = x2
                f1 = f2
                x2 = temp
                f2 = f_temp
            elif f_temp > f2:
                x3 = temp
                f3 = f_temp
            else:
                x1 = x2
                f1 = f2
                x3 = temp
                f3 = f_temp
                x2 = (x1 + x3) / 2
                f2 = f(x2)
                counter += 1
        table.add_row([i, x1, x3, xm, x3 - x1])
        i += 1
    middle = (x1 + x3) * 0.5
    return middle, counter


def brent_method(a, b, f):
    i = 1
    counter = 1

    gs = (3.0 - math.sqrt(5.0)) * 0.5
    x = w = v = a + gs * (b - a)
    # x = w = v =b
    tol = e * math.fabs(x) + e * 0.1
    fw = fv = fx = f(x)
    delta2 = delta = 0

    mid = (a + b) * 0.5
    fraction_1 = tol * math.fabs(x) + tol * 0.25
    fraction_2 = 2.0 * fraction_1

    while math.fabs(x - mid) > (fraction_2 - (b - a) * 0.5):
        if math.fabs(delta2) > fraction_1:

            # try and construct a parabolic method:
            r = (x - w) * (fx - fv)
            q = (x - v) * (fx - fw)
            p = (x - v) * q - (x - w) * r
            q = 2.0 * (q - r)

            if q > 0:
                p = -p

            q = math.fabs(q)
            td = delta2
            delta2 = delta
            # determine whether a parabolic step is acceptable or not:
            if (math.fabs(p) >= math.fabs(q * td * 0.5)) or (p <= q * (a - x)) or (p >= q * (b - x)):

                # not, try golden section instead
                if x >= mid:
                    delta2 = a - x
                else:
                    delta2 = b - x
                delta = gs * delta2

            else:

                # this is parabolic method:
                delta = p / q
                u = x + delta
                if ((u - a) < fraction_2) or ((b - u) < fraction_2):
                    if (mid - x) < 0:
                        delta = -math.fabs(fraction_1)
                    else:
                        delta = math.fabs(fraction_1)

        else:

            # golden section:
            if x >= mid:
                delta2 = a - x
            else:
                delta2 = b - x
            delta = gs * delta2

        # update current position:
        if math.fabs(delta) >= fraction_1:
            u = x + delta
        else:
            if delta > 0:
                u = x + math.fabs(fraction_1)
            else:
                u = x - math.fabs(fraction_1)

        fu = f(u)
        counter += 1
        if fu <= fx:

            # good new point is an improvement!
            # update brackets:
            if u >= x:
                a = x
            else:
                b = x
            # update control points:
            v = w
            w = x
            x = u
            fv = fw
            fw = fx
            fx = fu

        else:

            # point u is worse than what we have already,
            # even so it must be better than one of our endpoints:
            if u < x:
                a = u
            else:
                b = u
            if (fu <= fw) or (w == x):

                # however it is at least second best:
                v = w
                w = u
                fv = fw
                fw = fu

            elif (fu <= fv) or (v == x) or (v == w):

                # third best:
                v = u
                fv = fu

        table.add_row([i, a, b, x, b - a])
        i += 1
        mid = (a + b) * 0.5
        fraction_1 = tol * math.fabs(x) + tol * 0.25
        fraction_2 = 2.0 * fraction_1
    return x, counter
# ----------------------------------------------------------------------------------------------


function_test0 = lambda x: -5 * x ** 5 + 4 * x ** 4 - 12 * x ** 3 + 11 * x ** 2 - 2 * x + 1
function_test1 = lambda x: math.log10(x - 2) ** 2 + math.log10(10 - x) ** 2 - x ** 0.2
function_test2 = lambda x: -3.0 * x * math.sin(0.75 * x) + math.exp(-2.0 * x)
function_test3 = lambda x: math.exp(3.0 * x) + 5.0 * math.exp(-2.0 * x)
function_test4 = lambda x: 0.2 * x * math.log10(x) + (x - 2.3) ** 2
function_test5 = lambda x: math.sin(x)
e = 0.00001

list_function = [(function_test0, -0.5, 0.5),
                 (function_test1, 6.0, 9.9),
                 (function_test2, 0.0, 2.0 * math.pi),
                 (function_test3, 0.0, 1.0),
                 (function_test4, 0.5, 2.5),
                 (function_test5, -2 * math.pi, 2 * math.pi)]

list_method = [dichotomy_method, golden_section_method, fibonacci_method, parabolic_method, brent_method]

times = 1000000
# ===========change here=========
cortege = list_function[5]
# ===========change here=========
a1 = cortege[1]
b1 = cortege[2]
function_test = cortege[0]

table = PrettyTable()
table.field_names = ["Iteration", "a", "b", "x", "a-b"]

X = numpy.arange(a1, b1, 0.1)
pylab.plot([x for x in X], [function_test(x) for x in X])
pylab.grid(True)
pylab.show()

for method in list_method:
    x, counter = method(a1, b1, function_test)
    print('\nLocal minimum of the function by %s is %s' % (method, x))
    # print(table)
    print(table.get_string(title="Results"))
    print(' If method executed %d times, elapsed time %s. Number function calls is %s' % (
        times, check_time(method(a1, b1, function_test), times), counter))
    table.clear_rows()

eps = []
counters = [[] for i in range(len(list_method))]

for row in range(-10, -1, 1):
    e = 10.0 ** row
    col = 0
    for method in list_method:
        x, counter = method(a1, b1, function_test)
        counters[col].append(counter)
        col += 1
    eps.append(row)
    table.clear_rows()

lines = plt.plot(eps, counters[0], eps, counters[1], eps, counters[2], eps, counters[3], eps, counters[4])
plt.legend(lines[:5], ['Dichotomy', 'Golden_Section', 'Fibonacci', 'Parabolic', 'Brent'])

pylab.grid(True)
pylab.ylabel('Call function number')
pylab.xlabel('Log(eps)')

pylab.show()
