import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt_contour
import matplotlib.pyplot as plt2D
from matplotlib import pylab
import math

# import addendum as add

x_init = [100.0, 100.0]
# x_init = np.zeros(2)
# x_init = [2.0, 2.0, 2.0, 2.0]
max_iterations = 6000
epsilon = 1e-6
f_name = 'My function'


def any_function(V):
    # Rosenbrock_100 (0,0)
    # x, y = V
    # return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

    # # Rosenbrock (0,0)
    # x,y=V
    # return  (1.0 - x)**2 +  (y - x**2)**2

    # # Beale [3, 0.5]
    # x, y = V
    # return (1.5 - x * (1 - y)) ** 2 + (2.25 - x * (1 - y ** 2)) ** 2 + (2.625 - x * (1 - y ** 3)) ** 2

    # Test from lab [0,0,0,0]
    # x1, x2, x3, x4 = V
    # return (x1 + x2) ** 2 + 5.0 * (x3 - x4) ** 2 + (x2 - 2.0 * x3) ** 4 + 10.0 * (x1 - x4) ** 4

    # # Ravine function1 (8,1)
    # x, y=V
    # return  70.0*(x-8.0)**2+(y-1)**2+1

    # Ravine function2 (8,1)
    x, y = V
    return (x-8.0)**2+(y-1.0)**2+70.0*(y+(x-8.0)**2-1)**2+1

    # # Cylinder (0,0)
    # x, y=V
    # return x**2+y**2+25

    # # Himmelblau 𝑓(3.0,2.0)=0 𝑓(−2.805118,3.131312)=0 𝑓 (−3.779310,−3.283186)=0 𝑓(3.584428,−1.848126)=0
    # x, y = V
    # return  (x**2+y-11)**2+(x+y**2-7)**2

    # # Easom  has a global minimum in  (𝑥,𝑦)=(𝜋,𝜋) , where  𝑓(𝜋,𝜋)=−1 .
    # x, y = V
    # return -np.cos(x) * np.cos(y) * np.exp(-(x - np.pi)**2 -(y - np.pi)**2)


def partial_difference_quotient(f, v, i,
                                h):  # i is the variable that this function is being differentiated with respect to
    w = []
    for j, k in enumerate(v):
        if i == j:  # add h to just the ith element of v
            w.append(k + h)
        else:
            w.append(k)
    return (f(w) - f(v)) / h


def any_grad(f, v):
    h = 0.0001
    list_quotient = []
    for i, _ in enumerate(v):
        list_quotient.append(partial_difference_quotient(f, v, i, h))
    return np.array(list_quotient)


def golden_section_method(f, a, b):
    e = 1e-7
    #    i = 1
    #    counter=2
    phi = 2 - (1 + math.sqrt(5.0)) * 0.5

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
    #        table.add_row([i, a, b, (x1 + x2) * 0.5, a - b])
    #        counter+=1
    #       i += 1
    return (x1 + x2) * 0.5


def svenn(f, x0, eps):
    x1 = 0
    h = 0
    f0 = f(x0)
    if f0 > f(x0 + eps):
        x1 = x0 + eps
        h = eps
    elif f0 > f(x0 - eps):
        x1 = x0 - eps
        h = -eps
    h *= 2.0
    f1 = f(x1)
    x2 = x1 + h
    f2 = f(x2)
    i = 1
    while f1 > f2:
        x0 = x1
        f0 = f1
        x1 = x2
        f1 = f2
        h *= 2.0
        x2 = x1 + h
        f2 = f(x2)
        i += 1
    if h < 0:
        x0, x2 = x2, x0
    return x0, x2, i


def find_parabola_brent(x1, x2, y1, y2):
    a2 = (y1 - y2) / (x1 - x2)
    b = (x1 * y2 - x2 * y1) / (x1 - x2)
    return -b / a2


def eval_deriv(func, x):
    x_inc = 0.0000000001
    y_inc = func(x + x_inc) - func(x)
    return y_inc / x_inc


def two_dimensional_brent_deriv(func, a, c):
    eps = 1e-7
    its = -1

    x = w = v = (a + c) * 0.5
    fx = fw = fv = func(x)
    fxd = fwd = fvd = eval_deriv(func, x)

    a_bounds, b_bounds, x1_points, f1_values = [a], [c], [], []
    prev, prev_len = c - a, [c - a]
    d = e = c - a
    u1 = u2 = dist_u1 = dist_u2 = 0

    for i in range(50):
        g = e
        e = d
        took_u1 = took_u2 = False
        if x != w and fxd != fwd:
            u1 = find_parabola_brent(x, w, fxd, fwd)
            took_u1 = a + eps <= u1 <= c - eps and math.fabs(u1 - x) < g / 2
            dist_u1 = math.fabs(u1 - x)
        if x != v and fxd != fvd:
            u2 = find_parabola_brent(x, v, fxd, fvd)
            took_u2 = a + eps <= u2 <= c - eps and math.fabs(u2 - x) < g / 2
            dist_u2 = math.fabs(u2 - x)
        if took_u1 or took_u2:
            u, dist = (u1, dist_u1) if dist_u1 < dist_u2 else (u2, dist_u2)
        else:
            u = (a + x) / 2 if fxd > 0 else (x + c) / 2
        if math.fabs(u - x) < eps:
            u = -eps if u - x < 0 else eps
            u += x

        d = math.fabs(x - u)
        fu, fud = func(u), eval_deriv(func, u)
        x1_points.append(u)
        f1_values.append(fu)

        if fu <= fx:
            if u >= x:
                a = x
            else:
                c = x
            v, w, x, fv, fw = w, x, u, fw, fx
            fx, fvd, fwd, fxd = fu, fwd, fxd, fud
        else:
            if u >= x:
                c = u
            else:
                a = u
            if fu <= fw or w == x:
                v, w, fv, fw, fvd, fwd = w, u, fw, fu, fwd, fud
            elif fu <= fv or v == x or v == w:
                v, fv, fvd = u, fu, fud

        a_bounds.append(a)
        b_bounds.append(c)
        prev_len.append(prev)
        prev = c - a

        if math.fabs(x - w) < eps or math.fabs(func(x) - func(w)) < eps:
            its = i + 1
            break

    return x


def draw(f, f_name, x_init, history):
    if len(x_init) == 2:
        x = np.linspace(-9, 7, 100)
        y = np.linspace(-25, 50, 100)
        X, Y = np.meshgrid(x, y)
        Z = f((X, Y))
        coord_x, coord_y, coord_z = zip(*history)
        # Z = beale((X, Y))

        fig = plt.figure(figsize=(20, 10))
        ax = plt.axes(projection='3d')
        ax.set_title(f_name)
        ax.view_init(elev=50., azim=30)
        # s = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='jet', edgecolor='none')
        s = ax.plot_wireframe(X, Y, Z, rstride=1, cstride=1)
        ax.plot(coord_x, coord_y, coord_z, marker='*', color='r', alpha=.4, label='Gradient descent')
        # fig.colorbar(s, shrink=0.5, aspect=5)
        plt.show()

        # Рисуем контур
        fig_contour = plt_contour.figure(facecolor='white')
        ax_contour = fig_contour.gca()
        cs = ax_contour.contour(X, Y, Z, 40)
        # ax_contour.clabel(cs, v)

        # plot antigradient lane
        anglesx = np.array(coord_x)[1:] - np.array(coord_x)[:-1]
        anglesy = np.array(coord_y)[1:] - np.array(coord_y)[:-1]
        ax_contour.quiver(coord_x[:-1], coord_y[:-1], anglesx, anglesy, scale_units='xy', angles='xy', scale=1,
                          color='r', alpha=.9)

        plt_contour.show()


def steepest_gradient_descent(J, J_grad, x_init, epsilon, max_iterations):
    # метод наискорейшего спуска с автоматическим выбора шага (по золотому сечению)
    x = x_init
    J_history = []
    # foo = x
    # foo.append(J(x))
    # J_history.append(foo)
    for i in range(max_iterations):
        q = lambda alpha: J(x - alpha * J_grad(J, x))
        # alpha = gss(q, 0, 1)
        alpha = golden_section_method(q, 0, 1)
        # alpha = brent_method_deriv(q, 0, 1)
        # alpha=minimize_scalar(q)
        x = x - alpha * J_grad(J, x)
        # (x.tolist()).append(J(x))
        foo = x.tolist()
        foo.append(J(x))
        J_history.append(foo)
        if np.linalg.norm(J_grad(J, x)) < epsilon:
            return x, i + 1, J_history
    return x, max_iterations, J_history


def coord_descent(J, J_grad, x_init, epsilon, max_iterations):
    dim = len(x_init)
    args = x_init
    delta = 0.0000000000001
    J_history = []
    # foo = x_init
    # foo.append(J(x_init))
    # J_history.append(foo)
    #   J_history.append(args)
    #   val = J(args)
    iterations = 0
    for i in range(max_iterations):
        val = J(args)
        deriv_avg = 0.0
        for coord in np.identity(dim):
            new_arg = np.array(args) + coord * delta
            new_val = J(new_arg)
            deriv = (new_val - val) / delta
            deriv_avg += math.fabs(deriv)

            g = lambda a: J(args - a * coord * deriv)

            # a,b=svenn(g,args,0.1)
            # step = brent_method(g, 0.0, 1.0)
            # step = brent_method_deriv(g, 0.0, 1.0)
            step = golden_section_method(g, 0.0, 1.0)
            args -= step * deriv * coord
            val = J(args)

        foo = args.tolist()
        foo.append(J(args))
        J_history.append(foo)
        # J_history.append(args.tolist())
        deriv_avg /= dim
        iterations += 1
        if math.fabs(deriv_avg) < epsilon:
            break
    return args, iterations, J_history


def multidimensional_brent_derivative(J, J_grad, x_init, epsilon, max_iterations):
    dim = len(x_init)
    args = x_init
    delta = 0.0000000000001
    J_history = []
    # foo = x_init
    # foo.append(J(x_init))
    # J_history.append(foo)
    #   J_history.append(args)
    #   val = J(args)
    iterations = 0
    for i in range(max_iterations):
        val = J(args)
        deriv_avg = 0.0
        for coord in np.identity(dim):
            new_arg = np.array(args) + coord * delta
            new_val = J(new_arg)
            deriv = (new_val - val) / delta
            deriv_avg += math.fabs(deriv)

            g = lambda a: J(args - a * coord * deriv)

            # a,b=svenn(g,args,0.1)
            step = two_dimensional_brent_deriv(g, 0.0, 1.0)
            args -= step * deriv * coord
            val = J(args)

        foo = args.tolist()
        foo.append(J(args))
        J_history.append(foo)
        # J_history.append(args.tolist())
        deriv_avg /= dim
        iterations += 1
        if math.fabs(deriv_avg) < epsilon:
            break
    return args, iterations, J_history


def fact_grad_method_ravine_args_eval_1(func, args1, args2, rav_step):
    f1, f2 = func(args1), func(args2)
    args = args1 - rav_step * (args2 - args1) * (f2 - f1) / (np.linalg.norm(args2 - args1) ** 2)
    return args


def fact_grad_method_ravine_args_eval_2(func, x1, x0, rav_step):
    f1, f0 = func(x1), func(x0)
    args = x1 - rav_step * (x1 - x0) * math.copysign(1.0, f1 - f0) / np.linalg.norm(x1 - x0)
    return args


def fact_grad_method_get(func, args, dim, vects, delta, step_search_eps):
    vals = np.repeat(func(args), dim)
    new_args = np.tile(args, (dim, 1)) + vects
    new_val = np.apply_along_axis(lambda v: func(v), 1, new_args)
    grad = (new_val - vals) / delta

    ray = lambda a: func(args - a * grad)
    # step = brent_method(ray, 0.0, 0.1, step_search_eps)
    step = golden_section_method(ray, 0.0, 0.1)
    args -= grad * step
    return args


def fact_grad_method_ravine_get(func, args, dim, vects, delta, step_search_eps, ngbr_ratio, rav_step, method=2):
    if method == 1:
        eval_args = fact_grad_method_ravine_args_eval_1
    else:
        eval_args = fact_grad_method_ravine_args_eval_2

    ngbr_delta = np.random.rand(dim) * ngbr_ratio
    ngbr = args + ngbr_delta
    arg_list = np.array([args, ngbr])
    new_arg_list = np.apply_along_axis(
        lambda ar: fact_grad_method_get(func, ar, dim, vects, delta, step_search_eps), 1, arg_list)
    args1, args2 = new_arg_list
    args = eval_args(func, args1, args2, rav_step)
    return args


def grad_method_ravine(J, J_grad, x_init, epsilon, max_iterations):
    rav_step = 0.0001
    step_search_eps = 0.0000000001
    delta = 0.00000001
    J_history = []
    # foo = start
    # foo.append(func(start))
    # J_history.append(foo)

    dim, args = len(x_init), np.array(x_init)
    val = J(args)
    #   J_history.append(args)

    ident_array = np.identity(dim)
    vects = ident_array * delta
    ngbr_ratio = 0.00001

    for i in range(max_iterations):

        args = fact_grad_method_ravine_get(J, args, dim, vects, delta, step_search_eps, ngbr_ratio, rav_step)
        new_val = J(args)

        foo = args.tolist()
        foo.append(J(args))
        J_history.append(foo)
        # J_history.append(args.tolist())
        if math.fabs(val - new_val) < epsilon:
            return args, i + 1, J_history
        val = new_val
    return args, max_iterations, J_history


def getProjection(yx):
    ykx = yx
    ykx_ = yx[0] + (0.9 - yx[0]) / math.fabs(0.9 - yx[0]) * 0.5
    return ykx


def projection_gradient_descent(J, J_grad, x_init, epsilon, max_iterations):  # projection on sphere
    #
    x = x_init
    J_history = []
    # foo = x
    # foo.append(J(x))
    # J_history.append(foo)
    #    beta0=0.05
    ro = 0.8789
    for i in range(max_iterations):

        q = lambda alpha: J(x - alpha * J_grad(J, x))
        alpha = golden_section_method(q, 0, 1)  # basic step
        # alpha = brent_method_deriv(q, 0, 1)

        yx = x - alpha * J_grad(J, x)

        # ykx=yx+(0.9-yx)/math.fabs(0.9-yx)*0.5
        ykx = getProjection(yx)  # set projection of calculated point

        # ro=np.linalg.norm(J(x-ro*(x-ykx)))
        # ro=0.0003
        x_new = x - ro * (x - ykx)  # next point

        foo = x_new.tolist()
        foo.append(J(x_new))
        J_history.append(foo)
        if math.fabs(J(x_new) - J(x)) < epsilon:
            return x, i + 1, J_history
        x = x_new
    return x, max_iterations, J_history


list_method = [(multidimensional_brent_derivative, 'Brent derivative'), (coord_descent, 'Coordinate descent'),
               (steepest_gradient_descent, 'Steepest gradient descent'), (grad_method_ravine, 'Ravine gradient'),
               (projection_gradient_descent, 'Projection gradient')]


def picture_dependence_iterations_epsilon(any_function, any_grad, x_init, max_iterations, list_method):
    eps = []
    iterations = [[] for i in range(len(list_method))]
    result = []
    for i in range(-6, -1, 1):
        e = 10.0 ** i
        j = 0
        for foo in list_method:
            method = foo[0]
            result, iteration, history = method(any_function, any_grad, x_init, e, max_iterations)
            #   (any_function, any_grad, x_init, epsilon, max_iterations)
            iterations[j].append(iteration)
            j += 1
        eps.append(i)
    #    table.clear_rows()

    # pylab.plot(eps, count[0],   eps, count[1], eps, count[2], eps, count[3], eps, count[4])
    # plt.plot(eps, count[0])
    # plt.legend(['1'])
    lines = plt2D.plot(eps, iterations[0], eps, iterations[1], eps, iterations[2], eps, iterations[3])
    name_list = []
    for foo in list_method:
        name_list.append(foo[1])
    plt2D.legend(lines[:len(list_method)], name_list)
    # pylab.plot(eps, count[2])
    # pylab.plot(eps, count[3])
    # pylab.plot(eps, count[4])

    # for i in range(len(count[0])):
    #    pylab.plot(eps, [pt[i] for pt in count])
    # pylab.plot(eps, count)
    # pylab.plot(eps, count)
    pylab.grid(True)
    pylab.ylabel('Iterations')
    pylab.xlabel('Log(eps)')
    # pylab.legend(loc='best')
    pylab.show()


# picture_dependence_iterations_epsilon(any_function, any_grad, x_init, max_iterations, list_method)

# uncomment draw to see method that you choose on chosen function(now Ravine is active)
f_name = 'PROJECTION GRADIENT'
x_min, it, history = projection_gradient_descent(any_function, any_grad, x_init, epsilon, max_iterations)
# draw(any_function, f_name,x_init, history)
print('\npoint[min] =', x_min)
print(f_name + '(point[min]) =', any_function(x_min))
# print('Grad '+f_name+'(point[min]) =', any_grad(any_function,x_min))
print('Iterations =', it)
print('\n')

f_name = 'BRENT DERIVATIVE'
x_min, it, history = multidimensional_brent_derivative(any_function, any_grad, x_init, epsilon, max_iterations)
# draw(any_function, f_name,x_init, history)
print('\npoint[min] =', x_min)
print(f_name + '(point[min]) =', any_function(x_min))
# print('Grad '+f_name+'(point[min]) =', any_grad(any_function,x_min))
print('Iterations =', it)
print('\n')

f_name = 'COORDINATE DESCENT'
x_min, it, history = coord_descent(any_function, any_grad, x_init, epsilon, max_iterations)
# draw(any_function, f_name,x_init, history)
print('\npoint[min] =', x_min)
print(f_name + '(point[min]) =', any_function(x_min))
# print('Grad '+f_name+'(point[min]) =', any_grad(any_function,x_min))
print('Iterations =', it)
print('\n')

f_name = 'STEEPEST  DESCENT'
x_min, it, history = steepest_gradient_descent(any_function, any_grad, x_init, epsilon, max_iterations)
# draw(any_function, f_name,x_init, history)
print('\npoint[min] =', x_min)
print(f_name + '(point[min]) =', any_function(x_min))
# print('Grad '+f_name+'(point[min]) =', any_grad(any_function,x_min))
print('Iterations =', it)
print('\n')

f_name = 'RAVINE GRADIENT'
x_min, it, history = grad_method_ravine(any_function, any_grad, x_init, epsilon, max_iterations)
draw(any_function, f_name, x_init, history)
print('\npoint[min] =', x_min)
print(f_name + '(point[min]) =', any_function(x_min))
# print('Grad '+f_name+'(point[min]) =', any_grad(any_function,x_min))
print('Iterations =', it)
print('\n')

# f_name=list_method[i][1]
# x_min, it = method(any_function,  x_init, epsilon, max_iterations)
# print('\npoint[min] =', x_min)
# print(f_name+'(point[min]) =', any_function(x_min))
# #print('Grad '+f_name+'(point[min]) =', any_grad(any_function,x_min))
# print('Iterations =', it)
# print('\n')
