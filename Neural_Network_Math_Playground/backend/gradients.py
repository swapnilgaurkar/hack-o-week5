import sympy as sp

x, y = sp.symbols("x y")

def derivative_example(value):
    expr = x**3 + 2*x**2 + x
    derivative = sp.diff(expr, x)
    slope = derivative.subs(x, value)
    return {
        "function": "f(x) = x³ + 2x² + x",
        "derivative": "f'(x) = 3x² + 4x + 1",
        "x": value,
        "slope": float(slope)
    }

def gradient_example(x_value, y_value):
    expr = x**2 + 2*y**2
    gx = sp.diff(expr, x)
    gy = sp.diff(expr, y)
    return {
        "function": "f(x,y) = x² + 2y²",
        "gradient": ["∂f/∂x = 2x", "∂f/∂y = 4y"],
        "at_point": [x_value, y_value],
        "gradient_value": [float(gx.subs({x:x_value,y:y_value})),
                           float(gy.subs({x:x_value,y:y_value}))]
    }
