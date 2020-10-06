# This is a sample Python script.

# Press ⇧F10 to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import math
from matplotlib import pylab
import numpy
from prettytable import PrettyTable

function_test =  lambda x: -5*x**5+4*x**4-12*x**3+11*x**2-2*x+1
#func_first = lambda x: 6 * x ** 2 - 6 * x - 12

#def func_first(x):
 #   return 6*x**2-6*x-12

a1, b1 = -0.5, 0.5

e = 0.001


def Dichotomy_method(a, b, f):
    i=1
    while math.fabs(a-b) >= e:
        x = (a + b) / 2
        if f(x-e)<f(x+e):
            a,b=a,x
        else:
            a,b=x,b
        table.add_row([i, a, b, x,a-b])
        i+=1
    return x

def Golden_Section_method(a, b, f):
    i=1
    phi=2-(1+math.sqrt(5))*0.5

    x1 = a + phi * (b - a)
    x2 = b - phi * (b - a)
    f1=f(x1)
    f2=f(x2)

    while math.fabs(a-b) >= e:

        if f1<f2:
            b=x2
            x2=x1
            f2=f1
            x1=a+phi*(b-a)
            f1=f(x1)
            #a, b, x2, f2, x1, f1 =a, x2, x1, f1, a+phi*(b-a), f(x1)
        else:
            a=x1
            x1=x2
            f1=f2
            x2=b-phi*(b-a)
            f2=f(x2)
            #a, b, x1,f1, x2, f2 = x1, b, x2, f2, b-phi*(b-a), f(x2)
        table.add_row([i, a, b, (x1+x2)*0.5,a-b])
        i+=1
    return (x1+x2)*0.5


def fibonacci (n):
    fib=1
    if n>2:
        fib=fibonacci(n-1)+ fibonacci(n-2)
    return fib


def Fibonacci_method(a, b, f):
    i=1
    n = 0
    while fibonacci(n) < ((b - a)/e):
        n+=1
    #n=10
    x1=a+(fibonacci(n-2)/fibonacci(n))*(b-a)
    y1=f(x1)
    x2=a+(fibonacci(n-1)/fibonacci(n))*(b-a)
    y2=f(x2)

    for k in range(0,n-2):
        if y1<=y2:
            b=x2
            x2=x1
            y2=y1
            x1=a+(fibonacci(n-k-3)/fibonacci(n-k-1))*(b-a)
            y1=f(x1)
        else:
            a=x1
            x1=x2
            y1=y2
            x2 = a + (fibonacci(n - k - 2) / fibonacci(n - k - 1))*(b - a)
            y2 = f(x2)
 #       x2=x1+e
  #      y2=f(x2)
        table.add_row([k+1, a, b, (a+b)*0.5, a - b])
#    x2 = x1 + e
#    y2 = f(x2)
#    if y1<=y2:
#        b=y1
#    else:
#        a=x1
    return  (a+b)*0.5



table = PrettyTable()
table.field_names = ["Iteration", "a", "b", "x", "a-b"]

X = numpy.arange(-0.6, 0.6, 0.1)
pylab.plot([x for x in X], [function_test(x) for x in X])
pylab.grid(True)
pylab.show()

print ('\nlocal minimum of the function by Dichotomy method %s' % Dichotomy_method(a1, b1, function_test))
print(table)
table.clear_rows()

print ('\nlocal minimum of the function by Golden Section method %s' % Golden_Section_method(a1, b1, function_test))
print(table)
table.clear_rows()

print ('\nlocal minimum of the function by Fibonacci method %s' % Fibonacci_method(a1, b1, function_test))
print(table)
table.clear_rows()



