# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double⇧ to search everywhere for classes, files, tool windows, actions, and settings.

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
#a1, b1 = 6.0, 9.9
e = 0.00001

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
    counter=0
    while math.fabs(a - b) >= e:
        x = (a + b) / 2
        if f(x - e) < f(x + e):
            a, b = a, x
        else:
            a, b = x, b
        table.add_row([i, a, b, x, a - b])
        counter+=2
        i += 1
    return x, counter


def Golden_Section_method(a, b, f):
    i = 1
    counter=2
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
        counter+=1
        i += 1
    return (x1 + x2) * 0.5, counter


def fibonacci(n):
    fib = 1
    if n > 2:
        fib = fibonacci(n - 1) + fibonacci(n - 2)
    return fib


def Fibonacci_method(a, b, f):
    i = 1
    counter=2
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
        counter+=1
    #    x2 = x1 + e
    #    y2 = f(x2)
    #    if y1<=y2:
    #        b=y1
    #    else:
    #        a=x1
    return (a + b) * 0.5, counter


def parabolic_method(a, b, f):
    i = 1
    counter=3
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
            temp = (x1 + x2) / 2.0
        else:
            temp = xm
        ftemp = f(temp)
        counter+=1
        if temp < x2:
            if ftemp < f2:
                x3 = x2
                f3 = f2
                x2 = temp
                f2 = ftemp
            elif ftemp > f2:
                x1 = temp
                f1 = ftemp
            else:
                x1 = temp
                f1 = ftemp
                x3 = x2
                f3 = f2
                x2 = (x1 + x3) / 2
                f2 = f(x2)
                counter+=1
        elif temp > x2:
            if ftemp < f2:
                x1 = x2
                f1 = f2
                x2 = temp
                f2 = ftemp
            elif ftemp > f2:
                x3 = temp
                f3 = ftemp
            else:
                x1 = x2
                f1 = f2
                x3 = temp
                f3 = ftemp
                x2 = (x1 + x3) / 2
                f2 = f(x2)
                counter+=1
        table.add_row([i, x1, x3, xm, x3 - x1])
        i+=1
    x = (x1 + x3) *0.5
    return x, counter


def brent_method(a, b, f):
    i = 1
    counter=1

    gs=(3.0-math.sqrt(5.0))*0.5
    x = w = v = a+gs*(b-a)
    #x = w = v =b
    tol = e * math.fabs(x) + e * 0.1
    fw = fv = fx = f(x)
    delta2 = delta = 0

    mid = (a + b)*0.5
    fract1 = tol * math.fabs(x) + tol*0.25;
    fract2 = 2.0 * fract1

    while math.fabs(x - mid) > (fract2 - (b - a) *0.5):
 #       mid = (a + b) * 0.5
 #       fract1 = tol * math.fabs(x) + tol * 0.25;
 #       fract2 = 2.0 * fract1

        if math.fabs(delta2) > fract1:

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
            if(math.fabs(p) >= math.fabs(q * td *0.5 )) or (p <= q * (a - x)) or (p >= q * (b - x)):

            # not, try golden section instead
                if x >= mid:
                    delta2 =  a - x
                else:
                    delta2=b - x
                delta = gs * delta2

            else:

            # this's parabolic method:
                delta = p / q
                u = x + delta
                if(((u - a) < fract2) or ((b- u) < fract2)):
                    if (mid - x) < 0:
                        delta =-math.fabs(fract1)
                    else:
                        delta = math.fabs(fract1)


        else:

         # golden section:
            if x >= mid:
                delta2=a-x
            else:
                delta2=b-x
            delta = gs * delta2

      # update current position:
        if math.fabs(delta) >= fract1:
            u=x + delta
        else:
            if delta > 0:
                u= x + math.fabs(fract1)
            else:
                u= x - math.fabs(fract1)

        fu = f(u)
        counter+=1
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
        i+=1
        mid = (a + b) * 0.5
        fract1 = tol * math.fabs(x) + tol * 0.25;
        fract2 = 2.0 * fract1
    return x, counter






foo = list_function[2]
a1 = foo[1]
b1 = foo[2]
function_test = foo[0]

table = PrettyTable()
table.field_names = ["Iteration", "a", "b", "x", "a-b"]

X = numpy.arange(a1, b1, 0.1)
pylab.plot([x for x in X], [function_test(x) for x in X])
pylab.grid(True)
pylab.show()



x, counter=Dichotomy_method(a1, b1, function_test)
print('\nLocal minimum of the function by Dichotomy method %s' % x)
print(table)
print(' If method executed %d times, elapsed time %s. Number function calls is %s' % (
times, check_time(Dichotomy_method(a1, b1, function_test), times), counter))
table.clear_rows()

x, counter= Golden_Section_method(a1, b1, function_test)
print('\nLocal minimum of the function by Golden Section method %s' % x)
print(table)
print(' If method executed %d times, elapsed time %s. Number function calls is %s' % (
times, check_time(Golden_Section_method(a1, b1, function_test), times), counter))
table.clear_rows()

x, counter=Fibonacci_method(a1, b1, function_test)
print('\nLocal minimum of the function by Fibonacci method %s' % x)
print(table)
print(' If method executed %d times, elapsed time %s. Number function calls is %s' % (
times, check_time(Fibonacci_method(a1, b1, function_test), times), counter))
table.clear_rows()

x, counter=parabolic_method(a1, b1, function_test)
print('\nLocal minimum of the function by Parabolic method %s' % x)
print(table)
print(' If method executed %d times, elapsed time %s. Number function calls is %s' % (
times, check_time(parabolic_method(a1, b1, function_test), times), counter))
table.clear_rows()

x, counter=brent_method(a1, b1, function_test)
print('\nLocal minimum of the function by Brent method %s' % x)
print(table)
print(' If method executed %d times, elapsed time %s. Number function calls is %s' % (
times, check_time(brent_method(a1, b1, function_test), times), counter))
table.clear_rows()
