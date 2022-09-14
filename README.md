# CSMethod
Python library to efficiently calculate dynamical coefficients in dark matter perturbation theory using the Chebyshev Spectral Method (CSM).

## About

## Table of Contents

- [Pre-requisites](#requirements)

- [Using the library](#usage)

  [LCDMSolver](#LCDMSolver)

  [LCDMSolverFAST](#LCDMSolverFAST)

  [LKN](#LKN)
  
  [N](#N)
  
  [phi](#phi)
  
  [T](#T)
  
  [total](#total)
  
  [LamKapFinderEDS](#LamKapFinderEDS)


- [How to cite](#citations)

## Pre-requisites <a name="requirements"></a>
The CSM library was written in *Python 3.10.6* making use of several well-known libraries. These include *math*, *numpy* and *scipy*.
In order to make full use of the library these must be pre-installed, along with CSMethod itself.

The functions included herein can then be used once CSMethod has been cloned and imported.

## Using the library <a name="usage"></a>
Here I present some documentation on the correct usage and output of every function provided. All done using:

```python
from CSMethod import *
```

### LCDMSolver <a name="LCDMSolver"></a>
This function utilises the Chebyshev spectral method to calculate dynamical coefficients in the $\Lambda\mathrm{CDM}$ universe. It is called as follows:

```python
[Lambda, Kappa, LambdaOld, KappaOld] = LCDMSolver(n: int, N: int, Wm: float)
```

where $n$ is the desired order (e.g. $n = 3$ returns $\lambda_3$ and $\kappa_3$ etc.); $N$ is the maximum order of Chebyshev polynomials used (e.g. $N = 4$ returns the first 5 components); and $\Omega_m$ is the present-day matter density parameter.

This function will output `Lambda`, `Kappa` as arrays of length $N + 1$, representing the desired components of $\lambda_n$ and $\kappa_n$ respectively. `LambdaOld` and `KappaOld` are multi-dimensional arrays corresponding to all lower-order components. These can be parsed as follows:

```python
Lambda_n_l = Lambda[l-1]
Lambda_m_l = LambdaOld[m-1][l-1] #for 1 < m < n
Lambda_1 = LambdaOld[0] #for m = 1
```

The same method can be used to retrieve the associated values of $\kappa$.

In this way, the user is able to find all unknown components of the dynamical coefficients required to plot the two-loop matter-matter  $\Lambda\mathrm{CDM}$ power spectrum with $N = 4$ just by calling `LCDMSolver(5,4,Wm)` once.

Examples of how to proceed with these arrays are given in **total** below.

### LCDMSolverFAST <a name="LCDMSolverFAST"></a>

`LCDMSolverFAST` is a secondary function which works identically to `LCDMSolver` except taking components of $f_-/f_+^2$ and $1f_+$ $(c$ and $d$ respectively) as inputs. This bypasses the time-consuming step of the process.

As a result, this function is typically called as a time-saving step within the iterative structure of `LCDMSolver`. The user may call it as follows:

```python
[Lambda, Kappa, LambdaOld, KappaOld] = LCDMSolverFAST(n: int, N: int, c: array, d: array)
````

with the same output structure as `LCDMSolver` described above.

### LKN <a name="LKN"></a>

Though this library is built primarily around the Chebyshev spectral method, this provides an alternative brute force numerical integration method. The user calls `LKN` as follows:

```python
[Lambda, Kappa] = LKN(n: int, Wm: float, N: int)
````

Here, $n$ is the desired order, (e.g. $n = 3$ returns $\lambda_3$ and $\kappa_3$ etc.); and $\Omega_m$ is the present day matter density parameter. $N$ here defines the desired number of subdivisions, with $N/2$ used in the integration routine.

`Lambda` and `Kappa` are returned as multi-dimensional arrays containing all of the relevant $l$ modes. These are parsed as follows:

```python
Lambda_n_l = Lambda[l-1]
```

### N <a name="N"></a>

This is a numbering function used by the algorithm. Call `N(n)` to return the solution of:

$$ N(n) = \delta^K_{\frac{n}{2},\lfloor\frac{n}{2}\rfloor}\frac{1}{2}N\bigg{(}\frac{n}{2}\bigg{)}\bigg{[}1 + 3N\bigg{(}\frac{n}{2}\bigg{)}\bigg{]} + 3\sum_{m = 1}^{\lfloor (n - 1)/2\rfloor}N(m)N(n - m). $$

### phi <a name="phi"></a>

These are a series of bijective maps used by the algorithm. Call:

```python
x = phi1(n: int, i: int, j: int)
x = phi2(n: int, i: int, j: int)
x = phi3(n: int, m: int, i: int, j: int)
x = phi4(n: int, m: int, i: int, j: int)
x = phi5(n: int, m: int, i: int, j: int)
```

This returns the solution of:

$$ 	\phi_1(n,i,j) = N\bigg{(}\frac{n}{2}\bigg{)}(i - j) + i, $$

$$ \phi_2(n,i,j) = \bigg{(}N\bigg{(}\frac{n}{2}\bigg{)}\bigg{)}^2 - \frac{1}{2}i(i-1) + \phi_1(n,i,j), $$

$$ \phi_3(n,m,i,j) = \delta^K_{\frac{n}{2},\lfloor\frac{n}{2}\rfloor}\frac{1}{2}N\bigg{(}\frac{n}{2}\bigg{)}\bigg{[}1+3N\bigg{(}\frac{n}{2}\bigg{)} \bigg{]} + \sum_{k = 1}^{m-1}N(k)N(n - k) + (i-1)N(n-m) + j,  $$

$$ 	\phi_4(n,m,i,j) = \sum_{k = 1}^{\lfloor (n-1)/2\rfloor}N(k)N(n-k) + \phi_3(n,m,i,j), $$

$$ \phi_5(n,m,i,j) = 2\sum_{k = 1}^{\lfloor (n-1)/2\rfloor}N(k)N(n-k) + \phi_3(n,m,i,j). $$

### T <a name="T"></a>

These allow the user to use the Chebyshev polynomial of the first kind.

Calling `T(n: int, x: float)` is the Chebyshev polynomial of the first kind, returning the solution of:

$$ T_{n}(x) = 2xT_{n-1}(x) - T_{n-2}(x), \qquad T_1(x) = 1, \qquad T_0(x) = 1. $$

Calling `Ts(n: int, x: float)' is the shifted Chebyshev polynomial of the first kind, given by:

$$ \tilde{T}_n(x) = T_n(2x - 1). $$


### total <a name="total"></a>

This function allows the user to recombine a vector of components with the shifted Chebyshev polynomial basis, producing the full function. the user does this by calling:

```python
import numpy as np 
xs = np.linspace(0, 1, N: int) 
y = total(xs, d: array)
```

Here, $N$ is desired number of subdivisions and $d$ is the vector of components. This routine will most often be used alongside `LCDMSolver`, e.g.: 

```python
ts = np.linspace(0, 1, 100)
[Lambda, Kappa, LambdaOld, KappaOld] = LCDMSolver(n,N,Wm)
Lambda_n_l = total(ts, Lambda[l-1])
```

This would return $\lambda_n^{(l)}(t)$ as a dynamical function with 100 subdivisions.

### LamKapFinderEdS <a name="LamKapFinderEDS"></a>

This routine returns the EdS (Einstein-de-Sitter), i.e. initial, values of $\lambda_n^{(l)} and $\kappa_n^{(l)}. Call:

```python
[LambdaEDS, KappaEDS] = LamKapFinderEDS(n: int, l: int)
```

This returns $[\lambda_n^{(l), \mathrm{EdS}}, \kappa_n^{(l), \mathrm{EdS}}]$ as an array.

## How to cite <a name="citations"></a>
