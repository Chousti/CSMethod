import math
import numpy as np
from scipy.special import *
import scipy as sc
import scipy.integrate as q

#Important constants
a = 0
b = 1
H0 = 2.19507453e-18 #Present-day Hubble parameter

def deltaK(A: float, B: float):
    #Kronecker delta function
    if A == B:
        return 1
    else:
        return 0

def N(n: int):
    #Numbering function
    if n == 0:
        return 0
    if n == 1:
        return 1
    else:
        sum = 0
        for i in range(1, int(np.floor(0.5*(n + 1)))):
            sum += N(i)*N(n - i)
        if deltaK(n/2,np.floor(n/2)) == 1:
            return 0.5*N(n/2)*(3*N(n/2) + 1) + 3*sum
        else:
            return (3*sum)

def phi1(n: int, i: int, j: int):
    #returns first bijective map
    return N(n/2)*(i - 1) + j

def phi2(n: int, i: int, j: int):
    #Returns second bijective map
    return N(n/2)**2 - 0.5*i*(i-1) + phi1(n,i,j)

def phi3(n: int, m: int, i: int, j: int):
    #Returns third bijective map
    sum = 0
    for k in range(1,m):
        sum += N(k)*N(n - k)
    return deltaK(n/2,np.floor(n/2))*0.5*N(n/2)*(3*N(n/2) + 1) + (i - 1)*N(n-m) + j + sum 

def phi4(n: int, m: int, i: int, j: int):
    #Returns fourth bijective map
    sum = 0
    for k in range(1,int(np.floor(0.5*(n+1)))):
        sum += N(k)*N(n - k)
    return phi3(n,m,i,j) + sum

def phi5(n: int, m: int, i: int, j: int):
    #returns fifth bijective map
    sum = 0
    for k in range(1,int(np.floor(0.5*(n+1)))):
        sum += N(k)*N(n - k)
    return phi3(n,m,i,j) + 2*sum

def T(n: int,x: float):
    #Returns Chebyshev polynomial of the first kind
	if n == 0:
		return 1
	elif n == 1:
		return x
	else:
		return 2*x*T(n-1,x) - T(n-2,x)

def Ts(n: int, x: float):
    #Returns shifted Chebyshev polynomial of the first kind
    #Ts_n(x) = T_n(2*x - 1)
	y = (2*x - a - b)/(b - a)
	return T(n,y)

def total(x: float ,c: list):
    #Returns recombined function from components in shifted Chebyshev basis
    t = 0
    for i in range(0,len(c)):
        t+= c[i]*Ts(i,x)
    return t

def AlphaEDS(n: int, I: int, J: int, m1: int, m2: int):
    #Returns EdS solution for W_alpha, U_alpha for l mode of nth order
    [l1,k1] = LamKapfinderEDS(m1,I)
    [l2,k2] = LamKapfinderEDS(m2,J)
    WEDS = k1*l2*(2*n + 1)/(2*n**2 + n - 3)
    UEDS = WEDS*3/(2*n + 1)
    return [WEDS,UEDS]

def BetaEDS(n: int, I: int, J: int, m1: int, m2: int):
    #Returns EdS solution for W_beta, U_beta for l mode of nth order
    [l1,k1] = LamKapfinderEDS(m1,I)
    [l2,k2] = LamKapfinderEDS(m2,J)
    WEDS = k1*k2/(n**2 + n/2 - 3/2)
    UEDS = WEDS*n
    return [WEDS,UEDS]

def LamKapfinderEDS(n: int, l: int):
    #Returns EdS solution for \lambda_n^l, \kappa_n^l
    if n == 0:
        return [0,0]
    elif n == 1:
        return [1,1]
    else:
        s1 = 0
        s2 = 0
        if deltaK(n/2, np.floor(n/2)) == 1:
            t1 = 0
            t2 = 0
            for i in range(1,int(N(n/2)+1)):
                for j in range(1,int(N(n/2)+1)):
                    if deltaK(l,phi1(n,i,j)) == 1:
                        [Walpha, Ualpha] = AlphaEDS(n,i,j,n/2,n/2)
                        t1 += Walpha
                        t2 += Ualpha
                #Maybe speed this up by using if i \leq i then (could get rid of the extra loop)
                for j in range(i,int(N(n/2)+1)):
                    if deltaK(l,phi2(n,i,j)) == 1:
                        [Wbeta, Ubeta] = BetaEDS(n,i,j,n/2,n/2)
                        t1 += Wbeta
                        t2 += Ubeta
            s1 += t1
            s2 += t2
        
        for m in range(1, int(np.floor((n+1)/2))):
            for i in range(1,int(N(m)+1)):
                for j in range(1,int(N(n - m)+1)):
                    if deltaK(l,phi3(n,m,i,j)) == 1:
                        [Walpha, Ualpha] = AlphaEDS(n,i,j,m,n-m)
                        s1 += Walpha
                        s2 += Ualpha
                    if deltaK(l,phi4(n,m,i,j)) == 1:
                        [Walpha, Ualpha] = AlphaEDS(n,j,i,n-m,m)
                        s1 += Walpha
                        s2 += Ualpha
                    if deltaK(l,phi5(n,m,i,j)) == 1:
                        [Wbeta, Ubeta] = BetaEDS(n,i,j,m,n-m)
                        s1 += Wbeta
                        s2 += Ubeta
        return [s1, s2]

def LCDMSolver(n: int, M: int, Wm: float):
    #Returns M + 1 unknown components of lambda_n, kappa_n for LambdaCDM universe with Wm
    def Dminus(x: float, H0: int, Wm: int):
        y = ((1 - Wm)/Wm)*x**(3)
        #Returns D_- (negative growing mode, equal to conformal Hubble parameter)
        return H0*Wm**(0.5)*x**(-1.5)*np.sqrt(1 + y)

    def Dplus(x: float, Wm: float):
        #Returns D_+ (positive growing mode)
        y = ((1 - Wm)/Wm)*x**(3)
        w = -1
        #return x*(1 + y)**(0.5)*(hyp2f1(1.5, 5/6, 11/6 ,-y))
        #return x*hyp2f1(1/3, 1, 11/6, -y)
        return x*(1 + y)**(0.5)*(hyp2f1(1.5,-5*(6*w)**(-1),1-5*(6*w)**(-1),-y ) + y*((5*(1+w))/(5 - 6*w))*hyp2f1(1.5, 1-5*(6*w)**(-1), 2-5*(6*w)**(-1), -y))
        
        

    def fminus(x: float, Wm: float):
        #Returns f_- (logarithmic growth rate of negative mode)
        y = ((1 - Wm)/Wm)*x**(3)
        return -1.5 + 1.5*y*(1+y)**(-1)

    def fplus(x: float, Wm: float):
        #Returns f_+ (logarithmic growth rate of growing mode)
        y = ((1 - Wm)/Wm)*x**(3)
        return 1 + 1.5*y*(1+y)**(-1) - (45*y*(1+y)**(0.5)/(22))*(x/Dplus(x,Wm))*hyp2f1(2.5,(11)/(6), (17)/(6),-y)

    if n == 1:
        #n = 1 perturbative solution is known
        L = np.zeros(M+1)
        L[0] += 1
        K = np.zeros(M+1)
        K[0] += 1
        return [L,K,0,0]
    else:
        """
        Step 1: Calculate required weights of f_-/f_+^2 and 1/f_+ as well as previous \lambda,\kappa
        Step 2: Use CSM to calculate U,W's
        Step 3: Use LamKapfinder to sum these into desired \lambda,\kappa
        Step 4: Return info as array of arrays

        Current data format: Arrays of form: L = [[\lambda_1^1],[[\lambda_2^1],[\lambda_2^2]],...]
        \lambda_n^l = L[n-1][l-1], for n > 1
        \lambda_n^l = L[0], for n = 1

        Step 1
        Finding components for combinations of f_+, f_-
        """
        
        #c: components of 1/f_+
        def cweight(M: int, Wm: float):
            #Returns M + 1 components of 1/f_+ 
            cs = []
            def integrand(x: float):
                return 2*(fplus(x,Wm))**(-1)*Ts(i,x)/np.sqrt((b-a)**2 - (2*x - a - b)**2)
            for i in range(0,M):
                if i == 0:
                    s = q.quad(integrand,a,b)[0]/np.pi
                else:
                    s = 2*(q.quad(integrand,a,b)[0]/np.pi)
                cs.append(s)
            return cs  	
            
        #d: components of f_-/f_+^2
        def dweight(M: int, Wm: float):
            #Returns M + 1 components of f_-/f_+^2
            ds = []
            def integrand(x: float):
                return 2*fminus(x,Wm)*(fplus(x,Wm))**(-2)*Ts(i,x)/np.sqrt((b-a)**2 - (2*x - a - b)**2)
            for i in range(0,M):
                if i == 0:
                    s = q.quad(integrand,a,b)[0]/np.pi
                else:
                    s = 2*(q.quad(integrand,a,b)[0]/np.pi)
                ds.append(s)
            return ds  
            
        #Finding function weights
        c = cweight(M+1,Wm)
        d = dweight(M+1,Wm)
        
        #Finding previous functions        
        L = []
        K = []
        [l2,k2,l1,k1] = LCDMSolverFAST(n-1,M,c,d) #LCDMSolverFAST is used to save calculating for c,d again
        for i in range(1,n-1):
            L.append(l1[i-1])
            K.append(k1[i-1])
        L.append(l2)
        K.append(k2)  

        #Step 2
        #"Product" matrix, for left multiplying with a function with components x[n]
        def Product(x: list, M: float):
            def term(x: list, i: float, j: float):
                if j == 0:
                    if i == 0:
                        return 2*x[0]
                    else:
                        return x[i]
                elif i == 0:
                    if j == 0:
                        return 2*x[0]
                    else:
                        return 2*x[j]
                elif i == j:
                    if i + j > M:
                        return 2*x[0]
                    else:
                        return x[i+j] + 2*x[0]
                else:
                    if i + j > M: 
                        return x[np.abs(i - j)]
                    else:
                        return x[i+j] + x[np.abs(i - j)]
            X = np.zeros((M+1,M+1))
            for i in range(0,M+1):
                for j in range(0,M+1):
                    X[j,i] += 0.5*term(x,i,j)
            return X
				
		#"Derivative" matrix, for left multiplying with d/dx
        def Derivative(M: int):
            def term(i: int, j: int):
                if j >= i:
                    return 0
                elif j%2 == i%2:
                    return 0
                elif j%2 == 0:
                    if j == 0:
                        return 0.5*i
                    else:
                        return i
                else:
                    return i
            X = np.zeros((M+1,M+1))
            for i in range(0,M+1):
                for j in range(0,M+1):
                    X[j,i] += 4*term(i,j)
            return X

        #Components for f(x) = x
        def lin(M: int):
            l = np.zeros((M+1))
            l[0] += 0.5
            l[1] += 0.5
            return l

        #CSM method to find Ualpha and Walpha components
        def Alpha(n: int, I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
            P1 = Product(c,M)
            P2 = Product(d,M)
            P3 = Product(lin(M),M)
            D = Derivative(M)

            #Defining source terms for alpha system of ODEs
            def Wsource(I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
                m1 = int(m1)
                m2 = int(m2)
                g = 0
                h = 0
                for ii in range(1,int(m1)):
                    g += N(ii)
                for ii in range(1,int(m2)):
                    h += N(ii)
                g += I - 1
                h += J - 1
                if g == 0:
                    prod = Product(K[g],M)
                    if h == 0:
                        return prod@L[h]
                    else:
                        return prod@(L[m2 - 1][J - 1])
                else:
                    prod = Product(K[m1-1][I - 1],M)
                    if h == 0:
                        return prod@L[h]
                    else:
                        return prod@(L[m2 - 1][J - 1])

            def Usource(I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
                return np.zeros(M+1)

            #Finding equation of motion matrices
            Ws1 = (P3@P1@D + n*np.eye(M +1))
            Ws2 = -np.eye(M+1)
            Us1 = P2
            Us2 = (P3@P1@D + (n-1)*np.eye(M+1) - P2)

            A = np.hstack((Ws1,Ws2))
            B = np.hstack((Us1,Us2))
            source1 = Wsource(I,J,m1,m2,M,L,K)
            source2 = Usource(I,J,m1,m2,M,L,K)

            #Replacing last lines in A, B, source with constraints:	
            line = -1 #Final line
            for i in range(0,2*M+2):
                if i <= M:
                    if i == 0:
                        A[line,i] = (-1)**i
                    else:
                        A[line,i] = (-1)**i
                    B[line,i] = 0
                else:
                    A[line,i] = 0
                    if i == M+1:
                        B[line,i] = (-1)**(i - M - 1)
                    else:
                        B[line,i] = (-1)**(i - M - 1)

            [Weds, Ueds] = AlphaEDS(n,I,J,m1,m2) 
            source1[-1] = Weds
            source2[-1] = Ueds
            Mat = np.vstack((A,B))
            source = np.array(np.concatenate((source1,source2),axis=None))

			#Doing the linear algebra:
            Mat = np.array(Mat, dtype=np.double)
            source = np.array(source,dtype=np.double)
            sol = np.linalg.solve(Mat,source)
            W = np.asarray(sol[0:M+1])
            U = np.asarray(sol[M+1:])
            return [W,U]

        #CSM method to find Ubeta and Wbeta
        def Beta(n: int, I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
            P1 = Product(c,M)
            P2 = Product(d,M)
            P3 = Product(lin(M),M)
            D = Derivative(M)

            #Defining source terms for beta system of ODEs
            def Wsource(I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
                return np.zeros(M+1)

            def Usource(I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
                m1 = int(m1)
                m2 = int(m2)
                g = 0
                h = 0
                for ii in range(1,int(m1)):
                    g += N(ii)
                for ii in range(1,int(m2)):
                    h += N(ii)
                g += I - 1
                h += J - 1
                if g == 0:
                    prod = Product(K[g],M)
                    if h == 0:
                        return prod@K[h]
                    else:
                        return prod@(K[m2 - 1][J - 1])
                else:
                    prod = Product(K[m1-1][I - 1],M)
                    if h == 0:
                        return prod@K[h]
                    else:
                        return prod@(K[m2 - 1][J - 1])

            #Finding equation of motion matrices
            Ws1 = (P3@P1@D + n*np.eye(M +1))
            Ws2 = -np.eye(M+1)
            Us1 = P2
            Us2 = (P3@P1@D + (n-1)*np.eye(M+1) - P2)

            A = np.hstack((Ws1,Ws2))
            B = np.hstack((Us1,Us2))
            source1 = Wsource(I,J,m1,m2,M,L,K)
            source2 = Usource(I,J,m1,m2,M,L,K)

            #Replacing last lines in A, B, source with constraint:		
            line = -1 #Final line
            for i in range(0,2*M+2):
                if i <= M:
                    if i == 0:
                        A[line,i] = (-1)**i
                    else:
                        A[line,i] = (-1)**i
                    B[line,i] = 0
                else:
                    A[line,i] = 0
                    if i == M+1:
                        B[line,i] = (-1)**(i - M - 1)
                    else:
                        B[line,i] = (-1)**(i - M - 1)

            [Weds,Ueds] = BetaEDS(n,I,J,m1,m2)
            source1[-1] = Weds
            source2[-1] = Ueds
            Mat = np.vstack((A,B))
            source = np.array(np.concatenate((source1,source2),axis=None))

            #Doing the linear algebra:
            Mat = np.array(Mat, dtype=np.double)
            source = np.array(source,dtype=np.double)
            sol = np.linalg.solve(Mat,source)
            W = np.asarray(sol[0:M+1])
            U = np.asarray(sol[M+1:])
            return [W,U]

        #Step 3
        #Lambda and Kappa functions are found in terms of coefficient weights - then recombined
        def LamKapfinder(n: int, l: int, M: int, L: list, K: list): #Finds coefficients of \lambda and \kappa
            s1 = np.zeros(M+1)
            s2 = np.zeros(M+1)
            if deltaK(n/2, np.floor(n/2)) == 1:
                t1 = np.zeros(M+1)
                t2 = np.zeros(M+1)
                for i in range(1,int(N(n/2)+1)):
                    for j in range(1,int(N(n/2)+1)):
                        if deltaK(l,phi1(n,i,j)) == 1:
                            [Walpha, Ualpha] = Alpha(n,i,j,n/2,n/2,M,L,K)
                            t1 = np.add(t1,Walpha)
                            t2 = np.add(t2,Ualpha)
                    for j in range(i,int(N(n/2)+1)):
                        if deltaK(l,phi2(n,i,j)) == 1:
                            [Wbeta, Ubeta] = Beta(n,i,j,n/2,n/2,M,L,K)
                            t1 = np.add(t1,Wbeta)
                            t2 = np.add(t2,Ubeta)
                s1 = np.add(s1,t1)
                s2 = np.add(s2,t2)

            for m in range(1, int(np.floor((n+1)/2))):
                for i in range(1,int(N(m)+1)):
                    for j in range(1,int(N(n - m)+1)):
                        if deltaK(l,phi3(n,m,i,j)) == 1:
                            [Walpha, Ualpha] = Alpha(n,i,j,m,n-m,M,L,K)
                            s1 = np.add(s1,Walpha)
                            s2 = np.add(s2,Ualpha)
                        if deltaK(l,phi4(n,m,i,j)) == 1:
                            [Walpha, Ualpha] = Alpha(n,j,i,n-m,m,M,L,K)
                            s1 = np.add(s1,Walpha)
                            s2 = np.add(s2,Ualpha)
                        if deltaK(l,phi5(n,m,i,j)) == 1:
                            [Wbeta, Ubeta] = Beta(n,i,j,m,n-m,M,L,K)
                            s1 = np.add(s1,Wbeta)
                            s2 = np.add(s2,Ubeta)
            return [s1, s2]

        Lnew = []
        Knew = []
        for l in range(1,int(N(n)+1)):
            [l1,k1] = LamKapfinder(n,l,M,L,K)
            Lnew.append(l1)
            Knew.append(k1)
        return [Lnew,Knew,L,K] #Use func total() to recombine these into the full dynamical function

#Faster solver, which avoids calculating c, d based on inputs
def LCDMSolverFAST(n: int, M: int, c: list, d: list):
    #Returns M + 1 unknown components of lambda_n, kappa_n for LambdaCDM universe with Wm
    if n == 1:
        #n = 1 perturbative solution is known
        L = np.zeros(M+1)
        L[0] += 1
        K = np.zeros(M+1)
        K[0] += 1
        return [L,K,0,0]
    else:
        """
        Step 1: Calculate previous \lambda,\kappa
        Step 2: Use CSM to calculate U,W's
        Step 3: Use LamKapfinder to sum these into desired \lambda,\kappa
        Step 4: Return info as array of arrays

        Current data format: Arrays of form: L = [[\lambda_1^1],[[\lambda_2^1],[\lambda_2^2]],...]
        \lambda_n^l = L[n-1][l-1], for n > 1
        \lambda_n^l = L[0], for n = 1

        Step 1
        Finding components for combinations of f_+, f_-
        """
        L = []
        K = []
        [l2,k2,l1,k1] = LCDMSolverFAST(n-1,M,c,d)
        for i in range(1,n-1):
            L.append(l1[i-1])
            K.append(k1[i-1])
        L.append(l2)
        K.append(k2)
    
        def Product(x: list, M: float):
            def term(x: list, i: float, j: float):
                if j == 0:
                    if i == 0:
                        return 2*x[0]
                    else:
                        return x[i]
                elif i == 0:
                    if j == 0:
                        return 2*x[0]
                    else:
                        return 2*x[j]
                elif i == j:
                    if i + j > M:
                        return 2*x[0]
                    else:
                        return x[i+j] + 2*x[0]
                else:
                    if i + j > M: 
                        return x[np.abs(i - j)]
                    else:
                        return x[i+j] + x[np.abs(i - j)]
            X = np.zeros((M+1,M+1))
            for i in range(0,M+1):
                for j in range(0,M+1):
                    X[j,i] += 0.5*term(x,i,j)
            return X
				
		#"Derivative" matrix, for left multiplying with d/dx
        def Derivative(M: int):
            def term(i: int, j: int):
                if j >= i:
                    return 0
                elif j%2 == i%2:
                    return 0
                elif j%2 == 0:
                    if j == 0:
                        return 0.5*i
                    else:
                        return i
                else:
                    return i
            X = np.zeros((M+1,M+1))
            for i in range(0,M+1):
                for j in range(0,M+1):
                    X[j,i] += 4*term(i,j)
            return X

        #Components for f(x) = x
        def lin(M: int):
            l = np.zeros((M+1))
            l[0] += 0.5
            l[1] += 0.5
            return l

        def Alpha(n: int, I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
            P1 = Product(c,M)
            P2 = Product(d,M)
            P3 = Product(lin(M),M)
            D = Derivative(M)

            def Wsource(I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
                m1 = int(m1)
                m2 = int(m2)
                g = 0
                h = 0
                for ii in range(1,int(m1)):
                    g += N(ii)
                for ii in range(1,int(m2)):
                    h += N(ii)
                g += I - 1
                h += J - 1
                if g == 0:
                    prod = Product(K[g],M)
                    if h == 0:
                        return prod@L[h]
                    else:
                        return prod@(L[m2 - 1][J - 1])
                else:
                    prod = Product(K[m1-1][I - 1],M)
                    if h == 0:
                        return prod@L[h]
                    else:
                        return prod@(L[m2 - 1][J - 1])

            def Usource(I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
                return np.zeros(M+1)

            Ws1 = (P3@P1@D + n*np.eye(M +1))
            Ws2 = -np.eye(M+1)
            Us1 = P2
            Us2 = (P3@P1@D + (n-1)*np.eye(M+1) - P2)

            A = np.hstack((Ws1,Ws2))
            B = np.hstack((Us1,Us2))
            source1 = Wsource(I,J,m1,m2,M,L,K)
            source2 = Usource(I,J,m1,m2,M,L,K)
		
            line = -1 #Final line
            for i in range(0,2*M+2):
                if i <= M:
                    if i == 0:
                        A[line,i] = (-1)**i
                    else:
                        A[line,i] = (-1)**i
                    B[line,i] = 0
                else:
                    A[line,i] = 0
                    if i == M+1:
                        B[line,i] = (-1)**(i - M - 1)
                    else:
                        B[line,i] = (-1)**(i - M - 1)

            [Weds, Ueds] = AlphaEDS(n,I,J,m1,m2)
            source1[-1] = Weds
            source2[-1] = Ueds
            Mat = np.vstack((A,B))
            source = np.array(np.concatenate((source1,source2),axis=None))
			#Doing the linear algebra:
            Mat = np.array(Mat, dtype=np.double)
            source = np.array(source,dtype=np.double)
            sol = np.linalg.solve(Mat,source)
            W = np.asarray(sol[0:M+1])
            U = np.asarray(sol[M+1:])
            return [W,U]

        def Beta(n: int, I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
            P1 = Product(c,M)
            P2 = Product(d,M)
            P3 = Product(lin(M),M)
            D = Derivative(M)

            def Wsource(I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
                return np.zeros(M+1)

            def Usource(I: int, J: int, m1: int, m2: int, M: int, L: list, K: list):
                m1 = int(m1)
                m2 = int(m2)
                g = 0
                h = 0
                for ii in range(1,int(m1)):
                    g += N(ii)
                for ii in range(1,int(m2)):
                    h += N(ii)
                g += I - 1
                h += J - 1
                if g == 0:
                    prod = Product(K[g],M)
                    if h == 0:
                        return prod@K[h]
                    else:
                        return prod@(K[m2 - 1][J - 1])
                else:
                    prod = Product(K[m1-1][I - 1],M)
                    if h == 0:
                        return prod@K[h]
                    else:
                        return prod@(K[m2 - 1][J - 1])

            Ws1 = (P3@P1@D + n*np.eye(M +1))
            Ws2 = -np.eye(M+1)
            Us1 = P2
            Us2 = (P3@P1@D + (n-1)*np.eye(M+1) - P2)

            A = np.hstack((Ws1,Ws2))
            B = np.hstack((Us1,Us2))
            source1 = Wsource(I,J,m1,m2,M,L,K)
            source2 = Usource(I,J,m1,m2,M,L,K)
            		
            line = -1 #Final line
            for i in range(0,2*M+2):
                if i <= M:
                    if i == 0:
                        A[line,i] = (-1)**i
                    else:
                        A[line,i] = (-1)**i
                    B[line,i] = 0
                else:
                    A[line,i] = 0
                    if i == M+1:
                        B[line,i] = (-1)**(i - M - 1)
                    else:
                        B[line,i] = (-1)**(i - M - 1)

            [Weds,Ueds] = BetaEDS(n,I,J,m1,m2)
            source1[-1] = Weds
            source2[-1] = Ueds
            Mat = np.vstack((A,B))
            source = np.array(np.concatenate((source1,source2),axis=None))
            Mat = np.array(Mat, dtype=np.double)
            source = np.array(source,dtype=np.double)
            sol = np.linalg.solve(Mat,source)
            W = np.asarray(sol[0:M+1])
            U = np.asarray(sol[M+1:])
            return [W,U]

        def LamKapfinder(n: int, l: int, M: int, L: list, K: list): #Finds coefficients of \lambda
            s1 = np.zeros(M+1)
            s2 = np.zeros(M+1)
            if deltaK(n/2, np.floor(n/2)) == 1:
                t1 = np.zeros(M+1)
                t2 = np.zeros(M+1)
                for i in range(1,int(N(n/2)+1)):
                    for j in range(1,int(N(n/2)+1)):
                        if deltaK(l,phi1(n,i,j)) == 1:
                            [Walpha, Ualpha] = Alpha(n,i,j,n/2,n/2,M,L,K)
                            t1 = np.add(t1,Walpha)
                            t2 = np.add(t2,Ualpha)
                    for j in range(i,int(N(n/2)+1)):
                        if deltaK(l,phi2(n,i,j)) == 1:
                            [Wbeta, Ubeta] = Beta(n,i,j,n/2,n/2,M,L,K)
                            t1 = np.add(t1,Wbeta)
                            t2 = np.add(t2,Ubeta)
                s1 = np.add(s1,t1)
                s2 = np.add(s2,t2)

            for m in range(1, int(np.floor((n+1)/2))):
                for i in range(1,int(N(m)+1)):
                    for j in range(1,int(N(n - m)+1)):
                        if deltaK(l,phi3(n,m,i,j)) == 1:
                            [Walpha, Ualpha] = Alpha(n,i,j,m,n-m,M,L,K)
                            s1 = np.add(s1,Walpha)
                            s2 = np.add(s2,Ualpha)
                        if deltaK(l,phi4(n,m,i,j)) == 1:
                            [Walpha, Ualpha] = Alpha(n,j,i,n-m,m,M,L,K)
                            s1 = np.add(s1,Walpha)
                            s2 = np.add(s2,Ualpha)
                        if deltaK(l,phi5(n,m,i,j)) == 1:
                            [Wbeta, Ubeta] = Beta(n,i,j,m,n-m,M,L,K)
                            s1 = np.add(s1,Wbeta)
                            s2 = np.add(s2,Ubeta)           
            return [s1, s2]

        Lnew = []
        Knew = []
        for l in range(1,int(N(n)+1)):
            [l1,k1] = LamKapfinder(n,l,M,L,K)
            Lnew.append(l1)
            Knew.append(k1)
        return [Lnew,Knew,L,K] 

#Numerical solver for LCDM
def LCDMNumerical(n: int, Wm: float, M: int):
    def LamKapNumerical(n: int, l: int, Wm: float, x: float, M: int): #Finds coefficients of \lambda
        #LCDM Formulae:
        def Dplus(x: float, Wm: float):
            #Returns D_+ (positive growing mode)
            y = ((1 - Wm)/Wm)*x**(3)
            return x*(1 + y)**(0.5)*(hyp2f1(1.5,5*(6)**(-1),1+5*(6)**(-1),-y))

        def fminus(x: float, Wm: float):
            #Returns f_- (logarithmic growth rate of negative mode)
            y = ((1 - Wm)/Wm)*x**(3)
            return -1.5 + 1.5*y*(1+y)**(-1)

        def fplus(x: float, Wm: float):
            #Returns f_+ (logarithmic growth rate of growing mode)
            y = ((1 - Wm)/Wm)*x**(3)
            return 1 + 1.5*y*(1+y)**(-1) - (45*y*(1+y)**(0.5)/(22))*(x/Dplus(x,Wm))*hyp2f1(2.5,(11)/(6), (17)/(6),-y) 
        #Numerical integration code (brute force, slow)
        def NAlpha(n: int, i: int, j: int, m1: int, m2: int, Wm: float, x: float, M: int):
            WU0 = AlphaEDS(n,i,j,m1,m2)
            def integrand(y: list, u: float):
                fp = fplus(u,Wm)
                fm = fminus(u,Wm)
                [L1,K1] = LamKapNumerical(m1,i,Wm,u, M)
                [L2,K2] = LamKapNumerical(m2,j,Wm,u, M)
                return [fp/u*(y[1] - n*y[0] + K1*L2) , fp/u*( (fm*fp**(-2))*(y[1] - y[0]) - (n-1)*y[1])] #integrand vector
            u = np.linspace(0.000001,x,int(np.floor(M/2))) #M: Accuracy on sub-step
            sol = q.odeint(integrand,WU0,u)
            return [sol[-1,0], sol[-1,1]]

        def NBeta(n: int, i: int, j: int, m1: int, m2: int, Wm: float, x: float, M: int):
            WU0 = BetaEDS(n,i,j,m1,m2)
            def integrand(y: list, u: float):
                fp = fplus(u,Wm)
                fm = fminus(u,Wm)
                [L1,K1] = LamKapNumerical(m1,i,Wm,u, M)
                [L2,K2] = LamKapNumerical(m2,j,Wm,u, M)
                return [fp/u*(y[1] - n*y[0]) , fp/u*( (fm*fp**(-2))*(y[1] - y[0]) - (n-1)*y[1] + K1*K2)] #integrand vector
            a = np.linspace(0.000001,x,int(np.floor(M/2))) #M: Accuracy on sub-step
            sol = q.odeint(integrand,WU0,a)
            return [sol[-1,0],sol[-1,1]]
        if n == 1:
            return [1,1]
        else:
            s1 = 0
            s2 = 0
            if deltaK(n/2, np.floor(n/2)) == 1:
                t1 = 0
                t2 = 0
                for i in range(1,int(N(n/2)+1)):
                    for j in range(1,int(N(n/2)+1)):
                        if deltaK(l,phi1(n,i,j)) == 1:
                            [Walpha, Ualpha] = NAlpha(n,i,j,n/2,n/2,Wm,x, M)
                            t1 += Walpha
                            t2 += Ualpha
                    #Maybe speed this up by using if i \leq i then (could get rid of the extra loop)
                    for j in range(i,int(N(n/2)+1)):
                        if deltaK(l,phi2(n,i,j)) == 1:
                            [Wbeta, Ubeta] = NBeta(n,i,j,n/2,n/2,Wm,x, M)
                            t1 += Wbeta
                            t2 += Ubeta
                s1 += t1
                s2 += t2

            for m in range(1, int(np.floor((n+1)/2))):
                for i in range(1,int(N(m)+1)):
                    for j in range(1,int(N(n - m)+1)):
                        if deltaK(l, phi3(n,m,i,j)) == 1:
                            [Walpha, Ualpha] = NAlpha(n,i,j,m,n-m,Wm,x, M)
                            s1 += Walpha
                            s2 += Ualpha
                        if deltaK(l, phi4(n,m,i,j)) == 1:
                            [Walpha, Ualpha] = NAlpha(n,j,i,n-m,m,Wm,x, M)
                            s1 += Walpha
                            s2 += Ualpha
                        if deltaK(l, phi5(n,m,i,j)) == 1:
                            [Wbeta, Ubeta] = NBeta(n,i,j,m,n-m,Wm,x,M)
                            s1 += Wbeta
                            s2 += Ubeta          
            return [s1, s2]
    xs = np.linspace(0.0001,1,M)
    Lam = []
    Kap = []
    for i in range(1,int(N(n)+1)):
        l = []
        k = []
        for x in range(0,len(xs)):
            [L,K] = LamKapNumerical(n,i,Wm,xs[x], M)
            l.append(L)
            k.append(K)
        Lam.append(l)
        Kap.append(k)
    return [Lam,Kap]

