from cvxopt import solvers, matrix
import numpy as np

def standarize(X):
    # make each column of matrix X zero-mean and unit-variance 
    n,dim = X.shape
    col_mean = X.mean(axis = 0)
    col_std = X.std(axis=0)
    col_std = col_std.reshape((1,dim))
    col_std[col_std==0] = 1 # avoid the case of zero standard variance 
    return (X - col_mean.reshape((1,dim))) / col_std

def kernel_matrix(A,B,arg):
# calculate the kernel matrix H: H[i,j] = kernel(A[:,i],B[:,j])
    H = np.array([])
    k,m = A.shape
    k2,n = B.shape
    if k!= k2:
        print('Error: The number of rows of input matices do not coincide')
        return H
    arg['name'] = str.lower(arg['name'])
    r1 = arg['par']
    if arg['name'] == 'rbf':
        coef = 2*r1*r1
        H = np.zeros((m,n))
        for i in np.arange(n):
            U = A - B[:,i].reshape((k,1))
            H[:,i] = (np.exp(-np.sum(U*U,axis=0)/coef)).T
    elif arg['name'] == 'linear':
        H = A.T @ B + r1
    else: 
        print('do not support the specified kernel function')
    return H

def train_ramp(X,y,arg_model,arg_alg):
# train the classifier with ramp loss by CCCP algorithm
    debug_on = 1
    # 0. initilization and preparation
    n_sample,dim = X.shape
    n_labeled  = len(y)
    y = np.array(y)
    y = y.reshape((n_labeled,1))# ensure y is a n_labeled-by-1 column vector 
    
    # 0.0 parameter setting
    if 'gamma_A' not in arg_model:
        arg_model['gamma_A'] = 1
    gamma_A = arg_model['gamma_A']
    if 'arg_kernel' not in arg_model:
        arg_model['arg_kernel'] = {'name':'rbf','par':1}
    if 't' not in arg_model:
        arg_model['t'] = 1

    arg_kernel = arg_model['arg_kernel']        
    t = arg_model['t']
    if 'maxIte' not in arg_alg:
        arg_alg['maxIte'] = 50         
    if 'tolX'  not in arg_alg: 
        arg_alg['tolX'] = 1E-3
    if 'tolFun'  not in arg_alg: 
        arg_alg['tolFun'] = 1E-4
    # 0.1 compute the kernel matrix 
    K = kernel_matrix(X.T,X.T,arg_kernel)
    
    # 0.2 construct data adjacency graph matrix 
    arg_heatkernel = {'name':'rbf','par': np.sqrt(2*t)}
    W = kernel_matrix(X.T,X.T,arg_heatkernel)
    
    # 0.3 construct matrices for the CCCP subproblem
    n_w = n_sample+n_labeled
    # 0.3.1 set quadratic matrix Q 
    Q = np.zeros((n_w,n_w))
    Q[:n_sample,:n_sample] = 2*(gamma_A*K)
    Q_m = matrix(Q)
    # 0.3.2 set A, b for inequality constraints  A * w >= b   <=>  -A * w <= -b
    # set Diag(y)
    Diag_y = np.eye(n_labeled)
    row, col = np.diag_indices_from(Diag_y)
    Diag_y[row,col] =  y.reshape(n_labeled)
    # construct A 
    A = np.hstack((0.5*Diag_y @K[:n_labeled,:],np.identity(n_labeled)))
    A = np.vstack((A, np.hstack((np.zeros((n_labeled,n_sample)),np.identity(n_labeled))) ) )
    A_m = matrix(-A)
    # construct b in R^(2*n_labeled)
    b = np.vstack( ( 0.5*np.ones((n_labeled,1)),np.zeros((n_labeled,1))) )
    b_m = matrix(-b)
    
    # 0.4 define classifier 
    def classifier(X2,alpha):
        n_test,dim = X2.shape
        K2 = kernel_matrix(X2.T,X.T,arg_kernel) 
        y  = np.ones((n_test,1))
        y[K2@alpha <0] = -1
        return y
    
    # 0.5 initialization 
    w1 =  np.random.rand(n_w,1) *0.8
    alpha1 = w1[:n_sample]
    iteFlag = -1
    fun_obj_1 = 0.0
    
    # 1. CCCP iterations
    for ite in range(arg_alg['maxIte']):
        # solve CCCP subproblem 
        fun_obj_0  = fun_obj_1
        w0 = w1.copy()
        # 1.1 calculate linear term 
        # calculate eta 
        alpha0 = alpha1.copy() 
        eta = np.zeros((n_labeled,1))
        eta[y * (K[:n_labeled,:] @ alpha0)<-1] = -0.5 
        # construct c
        c = 1/n_labeled * np.vstack( (K[:,:n_labeled] @ (-eta*y), np.ones((n_labeled,1)) ) )
        c_m = matrix(c)
        # 1.2 solve the quadratic programming 
        model = solvers.qp(Q_m,c_m,A_m,b_m,initvals=matrix(w0))
        if debug_on:
            print('ite:',ite,model['status'],'gap:',model['gap'],'obj',model['primal objective'])
        w1= model['x']  
        w1 = np.array(w1) # transfrom w1 to numpy array from cvxopt.base.matrix 
        fun_obj_1 = model['primal objective'] 
        alpha1 = w1[:n_sample].reshape((n_sample,1))
        # 1.3 stopping criteria 
        if np.linalg.norm(alpha1-alpha0)< arg_alg['tolX']:
            iteFlag  =0 
            break
        if np.fabs(fun_obj_1-fun_obj_0)< arg_alg['tolFun']:
            iteFlag  =0
            break 
    print("the number of eta==-0.5:",np.sum(eta==-0.5))
    
    # 2. outputs      
    model ={'alpha':alpha1,'f':classifier,'arg_model':arg_model,'arg_alg':arg_alg}
    iteInf = {'flag':iteFlag,'ite':ite}
    return model,iteInf 
