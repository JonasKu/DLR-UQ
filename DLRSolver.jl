__precompile__
include("quadrature.jl")
include("Basis.jl")
include("TimeSolver.jl")

using ProgressMeter
using LinearAlgebra

struct DLRSolver
    # spatial grid of cell interfaces
    x;

    # quadrature
    q::Quadrature;

    # DLRSolver settings
    settings::Settings;

    # spatial basis functions
    basis::Basis;

    # low-rank solution matrices
    X::Array{Float64,2}
    W::Array{Float64,2}
    S::Array{Float64,2}

    # preallocated matrices for Rhs
    A::Array{Float64,3}
    B::Array{Float64,3}
    Y::Array{Float64,3}
    Y1::Array{Float64,2}
    ACons::Array{Float64,3}

    fluxS::Array{Float64,2}
    fluxL::Array{Float64,2}

    yL::Array{Float64,2}
    yS::Array{Float64,2}
    yK::Array{Float64,2}

    # Dirichlet BCs
    uL::Array{Float64,1}
    uR::Array{Float64,1}

    # time solver
    rkUpdate::TimeSolver

    # constructor
    function DLRSolver(settings)
        x = settings.x;
        r = settings.r;
        q = Quadrature(settings.Nq,"Gauss");
        basis = Basis(q,settings);

        # note that these are actually the hat variables
        X = zeros(settings.Nx,r)
        S = zeros(r,r)
        W = zeros(settings.N,r) 

        A = zeros(r,r,r);
        B = zeros(r,r,settings.Nq^2);
        Y = zeros(r,r,r);
        Y1 = zeros(r,r);

        fluxS = zeros(r,r);
        fluxL = zeros(r,settings.N^2);
        yL = zeros(r,settings.N^2);
        yS = zeros(r,r);
        yK = zeros(settings.Nx,r);

        uL = zeros(settings.Nq^2);
        uR = zeros(settings.Nq^2);

        ACons = zeros(settings.NCons,settings.N^2,settings.N^2);

        rkUpdate = TimeSolver(settings);

        new(x,q,settings,basis,X,S,W,A,B,Y,Y1,ACons,fluxS,fluxL,yL,yS,yK,uL,uR,rkUpdate);
    end
end


function RhsK(obj::DLRSolver,K::Array{Float64,2},W::Array{Float64,2})
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq;
    N = obj.settings.N;
    fXi = 0.25;
    dt = obj.settings.dt;
    flux = zeros(r);
    dx = obj.settings.dx;

    WQuad = EvalAtQuad(obj.basis,W)';

    # Compute A_{i,j,m} = E[W_i W_j W_m]
    WQuad = EvalAtQuad(obj.basis,W)';
    for i = 1:r
        for j = 1:r
            for m = 1:r
                obj.A[i,j,m] = Integral(obj.q,WQuad[i,:].*WQuad[j,:].*WQuad[m,:].*fXi);
            end
        end
    end

    for j = 2:(Nx-1)
        for p = 1:r
            flux[p] = 0.0;
            for l = 1:r
                for m = 1:r
                    flux[p] += 1/4/dx * (K[j+1,l]*K[j+1,m]-K[j-1,l]*K[j-1,m])*obj.A[l,m,p];
                end
            end
        end
        for p = 1:r
            obj.yK[j,p] = 1/2/dt .* (K[j+1,p]-2*K[j,p]+K[j-1,p]) - flux[p];
        end
    end
    return obj.yK;
end

function RhsS(obj::DLRSolver,X::Array{Float64,2},W::Array{Float64,2},L::Array{Float64,2})
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2;
    fXi = 0.25;
    dt = obj.settings.dt;
    # Compute A_{i,j,m} = E[L_i L_j phi_m]
    LQuad = EvalAtQuad(obj.basis,L)';
    WQuad = EvalAtQuad(obj.basis,W)';
    for i = 1:r
        for j = 1:r
            for m = 1:r
                obj.A[i,j,m] = Integral(obj.q,LQuad[i,:].*LQuad[j,:].*WQuad[m,:].*fXi);
            end
        end
    end

    for p = 1:r
        for l = 1:r
            obj.Y1[p,l] = 0.0;
            for j = 2:(Nx-1)
                if obj.settings.stabilization != 1
                    if obj.settings.stabilization == 2
                        obj.Y1[p,l] += -1/2/dt * X[j,p]*(X[j+1,l]-2*X[j,l]+X[j-1,l]);# DLR first, discretize second; stable version for projector splitting
                    else
                        obj.Y1[p,l] += 1/2/dt * X[j,p]*(X[j+1,l]-2*X[j,l]+X[j-1,l]); # discretize first, DLR second
                    end
                end
            end
        end
    end

    for p = 1:r
        for m = 1:r
            for l = 1:r
                obj.Y[m,l,p] = 0;
                for j = 2:(Nx-1) 
                    obj.Y[m,l,p] += 1/4/obj.settings.dx * X[j,p]*(X[j+1,l]*X[j+1,m]-X[j-1,l]*X[j-1,m]);
                end
            end
        end
    end

    for q = 1:r
        for l = 1:r
            obj.fluxS[q,l] = 0.0;
            for p = 1:r
                for m = 1:r
                    obj.fluxS[q,l] += obj.Y[m,p,q]*obj.A[m,p,l];
                end
            end
        end
        for l = 1:r
            obj.yS[q,l] = 0.0
            for m = 1:r
                for i = 1:N
                    obj.yS[q,l] += obj.Y1[q,m]*L[i,m]*W[i,l];
                end
            end
            obj.yS[q,l] -= obj.fluxS[q,l];
        end
    end
    return obj.yS;

end

function RhsL(obj::DLRSolver,X::Array{Float64,2},L::Array{Float64,2},recompute::Bool=true)
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2;
    fXi = 0.25;
    # Compute A_{i,j,m} = E[L_i L_j phi_m]
    LQuad = EvalAtQuad(obj.basis,L)';
    for i = 1:r
        for j = 1:r
            for m = 1:N
                obj.B[i,j,m] = Integral(obj.q,LQuad[i,:].*LQuad[j,:].*obj.basis.PhiQuad[:,m].*fXi);
            end
        end
    end

    if recompute
        for p = 1:r
            for m = 1:r
                for l = 1:r
                    obj.Y[m,l,p] = 0;
                    for j = 2:(Nx-1) 
                        obj.Y[m,l,p] += 1/4/obj.settings.dx * X[j,p]*(X[j+1,l]*X[j+1,m]-X[j-1,l]*X[j-1,m]);
                    end
                end
            end
        end

        for p = 1:r
            for l = 1:r
                obj.Y1[p,l] = 0.0;
                for j = 2:(Nx-1)
                    if obj.settings.stabilization != 1
                        obj.Y1[p,l] += 1/2/obj.settings.dt * X[j,p]*(X[j+1,l]-2*X[j,l]+X[j-1,l]);
                    end
                end
            end
        end
    end
    
    for p = 1:r
        for i = 1:N
            obj.fluxL[p,i] = 0.0;
            for l = 1:r
                for m = 1:r
                    obj.fluxL[p,i] += obj.Y[m,l,p]*obj.B[m,l,i];
                end
            end
        end
    end

    for p = 1:r
        for i = 1:N
            obj.yL[p,i] = 0.0;
            for l = 1:r
                obj.yL[p,i] += obj.Y1[p,l]*L[i,l];
            end
            obj.yL[p,i] -= obj.fluxL[p,i];
        end
    end

    return obj.yL;
end

function projector(u::Array{Float64,1},a::Array{Float64,1})
    factor = u'a/(u'u);
    return  factor .* u;
end

function qrGramSchmidt(A::Array{Float64,2})
    XNew,S = qr(A);
    X = XNew[:,1:size(S,2)]
    return X,S;
    N = size(A,1);
    r = size(A,2);

    Q = zeros(N,r);
    R = zeros(r,r);

    for k = 1:r
        Q[:,k] .= A[:,k];
        for j = 1:k-1
            Q[:,k] .-= projector(Q[:,j],A[:,k]);
        end
    end

    for k = 1:r
        Q[:,k] ./= sqrt(Q[:,k]'Q[:,k]); # normalize
    end

    for k = 1:r
        for j = r:-1:k
            R[k,j] = Q[:,k]'A[:,j];
        end
    end

    return Q,R;

end

function SetupIC(obj::DLRSolver)
    u = zeros(obj.settings.N*obj.settings.N,obj.settings.Nx);
    uCons = zeros(obj.settings.Nx,obj.settings.NCons);
    uVals = zeros(obj.settings.Nq^2)
    for j = 1:obj.settings.Nx
        for q = 1:obj.settings.Nq
            for k = 1:obj.settings.Nq
                uVals[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[j],obj.q.xi[k],obj.q.xi[q])[1];
            end
        end
        u[:,j] = ComputeMomentsDLR(obj.basis,uVals*0.25);
        uCons[j,:] = ComputeMomentsCons(obj.basis,uVals*0.25);
    end
    return u,uCons;
end

function Slope(obj::DLRSolver,u::Array{Float64,1},v::Array{Float64,1},w::Array{Float64,1})
    if obj.settings.limiterType == "Minmod"
        return minmod.(w.-v,v.-u)/obj.settings.dx;
    else 
        return 0.0;
    end
end

function Solve(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    N = obj.settings.N; # number of moments
    r = obj.settings.r; # DLR rank

    # Set up initial condition
    u = SetupIC(obj);
    uNew = deepcopy(u);

    # Low-rank approx of init data:
    X,S,W = svd(u'); 
    
    # rank-r truncation:
    X = X[:,1:obj.settings.r]; 
    W = W[:,1:obj.settings.r];
    S = Diagonal(S);
    #println(S);
    S = S[1:obj.settings.r, 1:obj.settings.r]; 

    A = zeros(r,r,r);
    B = zeros(r,r,N^2);
    Y = zeros(r,r,r);
    SFlux = zeros(r,r);
    numFlux = zeros(r);
    K = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N^2,r);
    LFlux = zeros(N^2,r);
    
    Nt = round(tEnd/dt);

    # compute Dirichlet values if they are independent of time
    uL = zeros(Nq^2);
    uR = zeros(Nq^2);
    for k = 1:Nq
        for q = 1:Nq
            uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    fXi = 0.25;
    
    # time loop
    @showprogress 0.1 "Progress " for n = 1:Nt

        ###### K-step ######
        K .= X*S;

        # Compute W basis on quadrature points
        WQuad = EvalAtQuad(obj.basis,W)';

        # impose BCs
        for i = 1:r
            K[1,i] = Integral(obj.q,uL.*WQuad[i,:].*fXi);
            K[2,i] = Integral(obj.q,uL.*WQuad[i,:].*fXi);
            K[end,i] = Integral(obj.q,uR.*WQuad[i,:].*fXi);
            K[end-1,i] = Integral(obj.q,uR.*WQuad[i,:].*fXi);
            KNew[1,i] = K[1,i]; KNew[2,i] = K[2,i];
            KNew[end,i] = K[end,i]; KNew[end-1,i] = K[end-1,i];
        end

        # Compute A_{i,j,m} = E[W_i W_j W_m]
        for i = 1:r
            for j = 1:r
                for m = 1:r
                    A[i,j,m] = Integral(obj.q,WQuad[i,:].*WQuad[j,:].*WQuad[m,:].*fXi);
                end
            end
        end

        for j = 3:(Nx-2)
            # compute numerical fluxes at cell j. Use Lax-Friedrichs
            KLeftMinus = K[j-1,:] .+ 0.5*Slope(obj,K[j-2,:],K[j-1,:],K[j,:])*dx;
            KLeftPlus = K[j,:] .- 0.5*Slope(obj,K[j-1,:],K[j,:],K[j+1,:])*dx;
            KRightMinus = K[j,:] .+ 0.5*Slope(obj,K[j-1,:],K[j,:],K[j+1,:])*dx;
            KRightPlus = K[j+1,:] .- 0.5*Slope(obj,K[j,:],K[j+1,:],K[j+2,:])*dx;
            for i = 1:r
                numFlux[i] = 0.25*(KRightMinus'*A[:,:,i]*KRightMinus .+ KRightPlus'*A[:,:,i]*KRightPlus).-0.5*dx/dt*(KRightPlus[i].-KRightMinus[i]); # g_{j+1/2}
                numFlux[i] -= 0.25*(KLeftMinus'*A[:,:,i]*KLeftMinus .+ KLeftPlus'*A[:,:,i]*KLeftPlus).-0.5*dx/dt*(KLeftPlus[i].-KLeftMinus[i]); # g_{j+1/2}-g_{j-1/2}
            end
            KNew[j,:] = K[j,:] .- dt/dx*numFlux;
        end
        K .= KNew;

        X,S = qrGramSchmidt(K); # optimize bei choosing XFull, SFull
        #println(size(K));
        #println(size(X));
        #println(size(S));
        #break;
        X = X[:, 1:obj.settings.r]; # remainder in S is zero, therefore we can throw away all columns after r (since in X*S, they will be multiplied by 0)
        S = S[1:obj.settings.r, 1:obj.settings.r]; # will be done by qr directly actually

        ###### S-step ######
        L .= Array((S*W')');
        # Compute A_{i,j,m} = E[L_i L_j W_m]
        LQuad = EvalAtQuad(obj.basis,L)';
        for i = 1:r
            for j = 1:r
                for m = 1:r
                    A[i,j,m] = Integral(obj.q,LQuad[i,:].*LQuad[j,:].*WQuad[m,:].*fXi);
                end
            end
        end

        # Compute Y_{l,q,k} = <\partial_x(X_l X_q) X_k>
        for l = 1:r
            for q = 1:r
                for k = 1:r
                    Y[l,q,k] = 0;
                    for j = 3:(Nx-2)
                        Y[l,q,k] += 0.5/dx*(-1/12*X[j+2,l]*X[j+2,q] + 2/3*X[j+1,l]*X[j+1,q] - 2/3*X[j-1,l]*X[j-1,q] + 1/12*X[j-2,l]*X[j-2,q])*X[j,k]; # maybe use trapezoidal rule?
                        #Y[l,q,k] += 0.25*(X[j+1,l]*X[j+1,q] - X[j-1,l]*X[j-1,q])*X[j,k];
                        #Y[l,q,k] += (0.25*(X[j+1,l]*X[j+1,q] + X[j,l]*X[j,q])-0.5*(X[j+1,q]-X[j,q]))*X[j,k];
                        #Y[l,q,k] -= (0.25*(X[j,l]*X[j-1,q] + X[j,l]*X[j,q])-0.5*(X[j,q]-X[j-1,q]))*X[j,k];
                    end
                end
            end
        end

        # compute Y*A
        for m = 1:r
            for k = 1:r
                SFlux[k,m] = 0.0;
                for l = 1:r
                    for q = 1:r
                        SFlux[k,m] += Y[l,q,k]*A[l,q,m];
                    end
                end
            end
        end

        S .= S .+ dt*SFlux;

        ###### L-step ######
        L = Array((S*W')');

        # Compute A_{i,j,m} = E[L_i L_j phi_m]
        LQuad = EvalAtQuad(obj.basis,L)';
        for i = 1:r
            for j = 1:r
                for m = 1:N^2
                    B[i,j,m] = Integral(obj.q,LQuad[i,:].*LQuad[j,:].*obj.basis.PhiQuad[:,m].*fXi);
                end
            end
        end

        # compute Y*A
        for m = 1:r
            for i = 1:(N^2)
                LFlux[i,m] = 0.0;
                for l = 1:r
                    for q = 1:r
                        LFlux[i,m] += Y[l,q,m]*B[l,q,i];
                    end
                end
            end
        end

        L .= L .- dt*LFlux;
        
        W,S = qrGramSchmidt(L);
        W = W[:, 1:obj.settings.r];
        S = S[1:obj.settings.r, 1:obj.settings.r];

        S .= S';

        # apply filter step
        Filter(obj,W);
        
        t = t+dt;

    end

    # return end time and solution
    return t, X, S, W;

end

function SolveBackward(obj::DLRSolver)
    println("Modal Backward")
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points

    fXi = 0.25;

    # Set up initial condition
    u,uCons = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u'); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    K1 = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N,r);
    L1 = zeros(N,r);
    S1 = zeros(r,r);
    
    Nt = Integer(round(tEnd/dt));

    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    #@gif 
    for n = 1:Nt

        ###### K-step ######
        K .= X*S;

        WQuad = EvalAtQuad(obj.basis,W)';

        # impose BCs
        for i = 1:r
            K[1,i] = Integral(obj.q,obj.uL.*WQuad[i,:].*fXi);
            K[end,i] = Integral(obj.q,obj.uR.*WQuad[i,:].*fXi);
        end
        
        if obj.settings.rkType == "Heun"
            K1 .= K .+ dt*RhsK(obj,K,W);
            K1 .= K1 .+ dt*RhsK(obj,K1,W);
            K .= 0.5.*(K.+K1);
        elseif obj.settings.rkType == "Euler"
            K .= K .+ dt*RhsK(obj,K,W);
        elseif obj.settings.rkType == "SSP"
            K .= UpdateK(obj.rkUpdate,obj,K,W);
        end

        X,S = qr(K); # optimize by choosing XFull, SFull
        X = X[:, 1:obj.settings.r]; 
        S = S[1:obj.settings.r, 1:obj.settings.r];

        ###### S-step ######

        L .= W*S';        
        if obj.settings.rkType == "Heun"
            S1 .= S .- dt.*RhsS(obj,X,W,L);
            L .= W*S';
            S1 .= S1 .- dt.*RhsS(obj,X,W,L);
            S .= 0.5.*(S.+S1);
        elseif obj.settings.rkType == "Euler"
            S .= S .- dt.*RhsS(obj,X,W,L);
        elseif obj.settings.rkType == "SSP"
            S .= UpdateS(obj.rkUpdate,obj,X,S,W);
        end

        ###### L-step ######
        L .= W*S';

        if obj.settings.rkType == "Heun"
            L1 .= L .+ dt*RhsL(obj,X,L,false)';
            L1 .= L1 .+ dt*RhsL(obj,X,L1,false)';
            L .= 0.5.*(L.+L1);
        elseif obj.settings.rkType == "Euler"
            L .= L .+ dt*RhsL(obj,X,L,false)';
        elseif obj.settings.rkType == "SSP"
            L .= UpdateL(obj.rkUpdate,obj,X,L,false);
        end
                
        W,S = qr(L);
        #W,S = qr(L);
        W = W[:, 1:obj.settings.r];
        S = S[1:obj.settings.r, 1:obj.settings.r];

        S .= S';

        # apply filter step
        Filter(obj,W);
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t, X,S,W;
end

function SolveForward(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points

    fXi = 0.25;

    # Set up initial condition
    u,uCons = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u'); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Array(Diagonal(S));
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    K1 = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N,r);
    L1 = zeros(N,r);
    S1 = zeros(r,r);

    PhiQuad = obj.basis.PhiQuad;
    
    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    #@gif 
    for n = 1:Nt

        ################## K-step ##################
        K .= X*S;

        # impose BCs
        WQuad = EvalAtQuad(obj.basis,W)';
        for i = 1:r
            K[1,i] = Integral(obj.q,obj.uL.*WQuad[i,:].*fXi);
            K[end,i] = Integral(obj.q,obj.uR.*WQuad[i,:].*fXi);
        end
        
        if obj.settings.rkType == "Heun"
            K1 .= K .+ dt*RhsK(obj,K,W);
            K1 .= K1 .+ dt*RhsK(obj,K1,W);
            K .= 0.5.*(K.+K1);
        elseif obj.settings.rkType == "Euler"
            K .= K .+ dt*RhsK(obj,K,W);
        elseif obj.settings.rkType == "SSP"
            K .= UpdateK(obj.rkUpdate,obj,K,W);
        end

        XNew,STmp = qr(K);
        XNew = XNew[:,1:r];

        MUp = XNew' * X;

        ################## L-step ##################
        L .= W*S';

        if obj.settings.rkType == "Heun"
            L1 .= L .+ dt*RhsL(obj,X,L)';
            L1 .= L1 .+ dt*RhsL(obj,X,L1)';
            L .= 0.5.*(L.+L1);
        elseif obj.settings.rkType == "Euler"
            L .= L .+ dt*RhsL(obj,X,L)';
        elseif obj.settings.rkType == "SSP"
            L .= UpdateL(obj.rkUpdate,obj,X,L);
        end
                
        WNew,STmp = qr(L);
        WNew = WNew[:,1:r];

        NUp = WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        L = W*S';

        if obj.settings.rkType == "Heun"
            S1 .= S .+ dt.*RhsS(obj,X,W,L);
            L .= W*S';
            S1 .= S1 .+ dt.*RhsS(obj,X,W,L);
            S .= 0.5.*(S.+S1);
        elseif obj.settings.rkType == "Euler"
            S .= S .+ dt.*RhsS(obj,X,W,L);
        elseif obj.settings.rkType == "SSP"
            S .= UpdateS(obj.rkUpdate,obj,X,S,W,false);
        end

        # apply filter step
        Filter(obj,W);
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    #println(norm(uHat))

    # return end time and solution
    return t, X,S,W;
end

function SolveForwardAdaptive(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points

    fXi = 0.25;

    # Set up initial condition
    u, uCons = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u'); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Array(Diagonal(S));
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    K1 = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N,r);
    L1 = zeros(N,r);
    S1 = zeros(r,r);

    PhiQuad = obj.basis.PhiQuad;
    
    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    #@gif 
    for n = 1:Nt

        ################## K-step ##################
        K .= X*S;

        # impose BCs
        WQuad = EvalAtQuad(obj.basis,W)';
        for i = 1:r
            K[1,i] = Integral(obj.q,obj.uL.*WQuad[i,:].*fXi);
            K[end,i] = Integral(obj.q,obj.uR.*WQuad[i,:].*fXi);
        end
        

        K1 .= K .+ dt*RhsK(obj,K,W);
        println(size(K))
        println(size(K1))
        K1 = [K1; K];
        println(size(K1))
        XNew,STmp = qr(K1);
        println(size(XNew))

        #XNew = XNew[:,1:r];

        MUp = XNew' * X;

        ################## L-step ##################
        L .= W*S';

        L1 .= L .+ dt*RhsL(obj,X,L)';
        L1 = [L1, L];
        WNew,STmp = qr(L1);
       
        #WNew = WNew[:,1:r];

        NUp = WNew' * W;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        L = WNew*S';

        S .= S .+ dt.*RhsS(obj,XNew,WNew,L);

        # Compute singular values of S1 and decide how to truncate:
        U,D,V = svd(S);
        tol = 1e-10;
        rmax = length(S[D>tol]);
        #safety check:
        #rmax = max(rmax,r);

        #cap to maximal rank of 20:
        rmax = min(rmax,20);


        X1 = XNew*X;
        W1 = WNew*W;

        # update solution with new rank
        S = S[1:rmax,1:rmax];
        X = XNew[:,1:rmax];
        W = WNew[:,1:rmax];

        # update rank
        r = rmax;

        # apply filter step
        Filter(obj,W);
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    #println(norm(uHat))

    # return end time and solution
    return t, X,S,W;
end

function RhsNodal(obj::DLRSolver,uQ::Array{Float64,2})
    rhs = zeros(size(uQ))
    Nx = obj.settings.Nx;
    for j = 2:(Nx-1)
        for k = 1:obj.settings.Nq^2
            rhs[j,k] = 0.5*(uQ[j+1,k]-2*uQ[j,k]+uQ[j-1,k])/obj.settings.dt - 0.25/obj.settings.dx*(uQ[j+1,k]^2-uQ[j-1,k]^2);
        end
    end
    return rhs;
end

function RhsNodalSplit(obj::DLRSolver,uQ::Array{Float64,2})
    rhs = zeros(size(uQ))
    Nx = obj.settings.Nx;
    for j = 2:(Nx-1)
        for k = 1:obj.settings.Nq^2
            rhs[j,k] = 0.5*(uQ[j+1,k]-2*uQ[j,k]+uQ[j-1,k])/obj.settings.dt - 0.25/obj.settings.dx*(uQ[j+1,k]^2-uQ[j-1,k]^2);
        end
    end
    return rhs;
end

function SolveNaiveSplit(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points

    fXi = 0.25;

    # save basis
    PhiQuad = Array(obj.basis.PhiQuad');
    PhiQuadCons = Array(obj.basis.PhiQuadCons');
    PhiQuadW = Array(obj.basis.PhiQuadW');
    PhiQuadWCons = Array(obj.basis.PhiQuadWCons');

    # Set up initial condition
    u, uCons = SetupIC(obj);
    u = Array(u');
    uQ = u*PhiQuad;
    #uNew = u;

    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    for n = 1:Nt

        uQ = u*PhiQuad + uCons*PhiQuadCons;

        println(maximum(dt*RhsNodal(obj,uQ)))

        #uQ = uQ .+ dt*RhsNodal(obj,uQ);

        uCons = uCons .+ dt*RhsNodalSplit(obj,uQ)*fXi*PhiQuadWCons;

        u = u .+ dt*RhsNodalSplit(obj,uQ)*fXi*PhiQuadW;

        next!(prog) # update progress bar

        t = t+dt;
    end

    # compute moments
    uQ = u*PhiQuad + uCons*PhiQuadCons;
    u = uQ*Array(obj.basis.PhiQuadWFull')*fXi;

    # return end time and solution
    return t, u;
    #return t, uQ*fXi*PhiQuadW;
end

function SolveNaiveSplitUnconventionalIntegrator(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points
    r = obj.settings.r;

    fXi = 0.25;

    # save basis
    PhiQuad = Array(obj.basis.PhiQuad');
    PhiQuadCons = Array(obj.basis.PhiQuadCons');
    PhiQuadW = Array(obj.basis.PhiQuadW');
    PhiQuadWCons = Array(obj.basis.PhiQuadWCons');

    # Set up initial condition
    u, uCons = SetupIC(obj);
    u = Array(u');
    uQ = u*PhiQuad;

    # Low-rank approx of init data:
    X,S,W = svd(u); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Array(Diagonal(S));
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    K1 = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N,r);
    L1 = zeros(N,r);
    S1 = zeros(r,r);

    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    for n = 1:Nt

        # reconstruct solution (inefficient)
        uQ = X*S*W'*PhiQuad + uCons*PhiQuadCons;

        # update conservative part (inefficient)
        uCons = uCons .+ dt*RhsNodalSplit(obj,uQ)*fXi*PhiQuadWCons;


        ###### K-step ######
        K .= X*S;
        WQuadW = PhiQuadW*W; # compute W_{ki}*w_k , where W \in R^{Nq x r}

        #K .= K .+ dt*Integral(obj.q,F(obj,K*EvalAtQuadDLR(obj.basis,W))*EvalAtQuadDLR(obj.basis,W).*fXi);
        K .= K .+ dt*RhsNodalSplit(obj,uQ)*WQuadW.*fXi;

        XNew,STmp = qrGramSchmidt(K); # optimize bei choosing XFull, SFull

        MUp = XNew' * X;

        ###### L-step ######
        L = W*S';

        #L .= L .+ dt*(X'*F(obj,X*EvalAtQuadDLR(obj.basis,L)))';
        L .= L .+ dt*(X'*RhsNodalSplit(obj,uQ)*PhiQuadW.*fXi)';
                
        WNew,STmp = qrGramSchmidt(L);

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')
        uQ = X*S*W'*PhiQuad + uCons*PhiQuadCons;
        WQuadW = PhiQuadW*W;

        S .= S .+ dt.*X'*RhsNodalSplit(obj,uQ)*WQuadW.*fXi;
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    # compute moments
    uQ = X*S*W'*PhiQuad + uCons*PhiQuadCons;
    u = uQ*Array(obj.basis.PhiQuadWFull')*fXi;

    # return end time and solution
    return t, u;
end

function SolveNaive(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points

    fXi = 0.25;

    # save basis
    PhiQuad = Array(obj.basis.PhiQuad');
    PhiQuadW = Array(obj.basis.PhiQuadW');

    # Set up initial condition
    u, uCons = SetupIC(obj);
    u = Array(u');
    uQ = u*PhiQuad;
    #uNew = u;

    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    for n = 1:Nt

        uQ = u*PhiQuad;

        println(maximum(dt*RhsNodal(obj,uQ)))

        #uQ = uQ .+ dt*RhsNodal(obj,uQ);

        u = u .+ dt*RhsNodal(obj,uQ)*fXi*PhiQuadW;
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t, u;
    #return t, uQ*fXi*PhiQuadW;
end

function SolveNaiveForward(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points

    fXi = 0.25;

    # Set up initial condition
    u, uCons = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u'); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Array(Diagonal(S));
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    K1 = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N,r);
    L1 = zeros(N,r);
    S1 = zeros(r,r);

    PhiQuad = obj.basis.PhiQuad;

    # set up conservative flux part
    for i = 1:N^2
        for j = 1:N^2
            obj.ACons[:,i,j] = ComputeMomentsCons(obj.basis,PhiQuad[:,i].*PhiQuad[:,j]*fXi);
        end
    end
    
    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    #@gif 
    for n = 1:Nt
        #uQNew .= uQ .+ dt*F(obj,uQ);
        #uQ .= uQNew;

        # update conservative flux (inefficient)
        uQ = X*S*EvalAtQuad(obj.basis,W);
        uQCons = EvalAtQuad(obj.basis,uCons);

        ###### K-step ######
        K .= X*S;

        K .= K .+ dt*Integral(obj.q,F(obj,K*EvalAtQuad(obj.basis,W))*EvalAtQuad(obj.basis,W).*fXi);

        XNew,STmp = qrGramSchmidt(K); # optimize bei choosing XFull, SFull

        MUp = XNew' * X;

        ###### L-step ######
        L = W*S';

        L .= L .+ dt*(X'*F(obj,X*EvalAtQuad(obj.basis,L)))';
                
        WNew,STmp = qrGramSchmidt(L);

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        S .= S .+ dt.*X'*F(obj,X*S*W')*W;
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    uQ = X*S*W';

    # compute moments of u for easier plotting
    uHat = zeros(obj.settings.Nx,obj.settings.N);
    for j = 1:obj.settings.Nx
        for i = 1:obj.settings.N
            uHat[j,i] = Integral(obj.q,uQ[j,:].*obj.basis.PhiQuad[:,i].*fXi);
        end
    end

    println(norm(uHat))


    X,S,W = svd(uHat);
    S = Diagonal(S);
    # return end time and solution
    return t, X,S,W;
end

function SolveForwardBC(obj::DLRSolver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq^2;
    N = obj.settings.N^2; # here, N is the number of quadrature points

    fXi = 0.25;

    # Set up initial condition
    u, uCons = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u'); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Array(Diagonal(S));
    S = S[1:r, 1:r]; 

    K = zeros(Nx,r);
    K1 = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N,r);
    L1 = zeros(N,r);
    S1 = zeros(r,r);

    PhiQuad = obj.basis.PhiQuad;

    # set up conservative flux part
    for i = 1:N^2
        for j = 1:N^2
            obj.ACons[:,i,j] = ComputeMomentsCons(obj.basis,PhiQuad[:,i].*PhiQuad[:,j]*fXi);
        end
    end
    
    Nt = Integer(round(tEnd/dt));
    
    # compute Dirichlet values if they are independent of time
    for k = 1:obj.settings.Nq
        for q = 1:obj.settings.Nq
            obj.uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            obj.uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    #@gif 
    for n = 1:Nt

        ################## K-step ##################
        K .= X*S;

        # impose BCs
        WQuad = EvalAtQuad(obj.basis,W)';
        for i = 1:r
            K[1,i] = Integral(obj.q,obj.uL.*WQuad[i,:].*fXi);
            K[end,i] = Integral(obj.q,obj.uR.*WQuad[i,:].*fXi);
        end
        
        if obj.settings.rkType == "Heun"
            K1 .= K .+ dt*RhsK(obj,K,W);
            K1 .= K1 .+ dt*RhsK(obj,K1,W);
            K .= 0.5.*(K.+K1);
        elseif obj.settings.rkType == "Euler"
            K .= K .+ dt*RhsK(obj,K,W);
        elseif obj.settings.rkType == "SSP"
            K .= UpdateK(obj.rkUpdate,obj,K,W);
        end

        XNew,STmp = qr(K);
        XNew = XNew[:,1:r];

        MUp = XNew' * X;

        ################## L-step ##################
        L .= W*S';

        if obj.settings.rkType == "Heun"
            L1 .= L .+ dt*RhsL(obj,X,L)';
            L1 .= L1 .+ dt*RhsL(obj,X,L1)';
            L .= 0.5.*(L.+L1);
        elseif obj.settings.rkType == "Euler"
            L .= L .+ dt*RhsL(obj,X,L)';
        elseif obj.settings.rkType == "SSP"
            L .= UpdateL(obj.rkUpdate,obj,X,L);
        end
                
        WNew,STmp = qr(L);
        WNew = WNew[:,1:r];

        NUp = WNew' * W;

        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        L = W*S';

        if obj.settings.rkType == "Heun"
            S1 .= S .+ dt.*RhsS(obj,X,W,L);
            L .= W*S';
            S1 .= S1 .+ dt.*RhsS(obj,X,W,L);
            S .= 0.5.*(S.+S1);
        elseif obj.settings.rkType == "Euler"
            S .= S .+ dt.*RhsS(obj,X,W,L);
        elseif obj.settings.rkType == "SSP"
            S .= UpdateS(obj.rkUpdate,obj,X,S,W,false);
        end

        # apply filter step
        Filter(obj,W);
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    #println(norm(uHat))

    # return end time and solution
    return t, X,S,W;
end

function SolveBackwardOld(obj::DLRSolver)
    t = 0.0;
    useStabilizingTerms = obj.settings.useStabilizingTerms;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq;
    N = obj.settings.N; # here, N is the number of quadrature points

    fXi = 0.25;
    dXi = obj.q.w[1];

    # Set up initial condition
    u = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u'); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r]; 

    A = zeros(r,r,r);
    B = zeros(r,r,Nq^2);
    Y = zeros(r,r,r);
    Y1 = zeros(r,r);
    SFlux = zeros(r,r);
    numFlux = zeros(r);
    K = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N^2,r);

    yK = zeros(Nx,r);
    fluxK = zeros(r);
    yS = zeros(r,r);
    fluxS = zeros(r,r);
    yL = zeros(r,N^2);
    fluxL = zeros(r,N^2);
    
    Nt = Integer(round(tEnd/dt));
    println(Nt);

    # compute Dirichlet values if they are independent of time
    uL = zeros(Nq^2);
    uR = zeros(Nq^2);
    for k = 1:Nq
        for q = 1:Nq
            uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    #@showprogress 0.1 "Progress " 
    #@gif 
    for n = 1:Nt

        ###### K-step ######
        K .= X*S;

        # Compute W basis on quadrature points
        WQuad = EvalAtQuad(obj.basis,W)';

        # impose BCs
        for i = 1:r
            K[1,i] = Integral(obj.q,uL.*WQuad[i,:].*fXi);
            K[end,i] = Integral(obj.q,uR.*WQuad[i,:].*fXi);
        end

        # Compute A_{i,j,m} = E[W_i W_j W_m]
        for i = 1:r
            for j = 1:r
                for m = 1:r
                    A[i,j,m] = Integral(obj.q,WQuad[i,:].*WQuad[j,:].*WQuad[m,:].*fXi);
                end
            end
        end

        for j = 2:(Nx-1)
            for p = 1:r
                fluxK[p] = 0.0;
                for l = 1:r
                    for m = 1:r
                        fluxK[p] += 1/4/dx * (K[j+1,l]*K[j+1,m]-K[j-1,l]*K[j-1,m])*A[l,m,p];
                    end
                end
            end
            for p = 1:r
                yK[j,p] = 1/2/dt .* (K[j+1,p]-2*K[j,p]+K[j-1,p]) - fluxK[p];
            end
        end

        K .= K .+ dt*yK;

        X,S = qr(K); # optimize by choosing XFull, SFull
        X = X[:, 1:obj.settings.r]; 
        S = S[1:obj.settings.r, 1:obj.settings.r];

        ###### S-step ######

        L = W*S';

        # Compute A_{i,j,m} = E[L_i L_j phi_m]
        LQuad = EvalAtQuad(obj.basis,L)';
        WQuad = EvalAtQuad(obj.basis,W)';
        for i = 1:r
            for j = 1:r
                for m = 1:r
                    A[i,j,m] = Integral(obj.q,LQuad[i,:].*LQuad[j,:].*WQuad[m,:].*fXi);
                end
            end
        end

        for p = 1:r
            for l = 1:r
                Y1[p,l] = 0.0;
                for j = 2:(Nx-1)
                    if useStabilizingTerms
                        Y1[p,l] += 1/2/dt * X[j,p]*(X[j+1,l]-2*X[j,l]+X[j-1,l]);
                    end
                end
            end
        end

        for p = 1:r
            for m = 1:r
                for l = 1:r
                    Y[m,l,p] = 0;
                    for j = 2:(Nx-1) 
                        Y[m,l,p] += 1/4/dx * X[j,p]*(X[j+1,l]*X[j+1,m]-X[j-1,l]*X[j-1,m]);
                    end
                end
            end
        end
    
        for q = 1:r
            for l = 1:r
                fluxS[q,l] = 0.0;
                for p = 1:r
                    for m = 1:r
                        fluxS[q,l] += Y[m,p,q]*A[m,p,l];
                    end
                end
            end
            for l = 1:r
                yS[q,l] = 0.0
                for m = 1:r
                    for i = 1:N^2
                        yS[q,l] += Y1[q,m]*L[i,m]*W[i,l];
                    end
                end
                yS[q,l] -= fluxS[q,l];
            end
        end

        S .= S .- dt.*yS;

        ###### L-step ######
        L = W*S';

        # Compute A_{i,j,m} = E[L_i L_j phi_m]
        LQuad = EvalAtQuad(obj.basis,L)';
        for i = 1:r
            for j = 1:r
                for m = 1:N^2
                    B[i,j,m] = Integral(obj.q,LQuad[i,:].*LQuad[j,:].*obj.basis.PhiQuad[:,m].*fXi);
                end
            end
        end
        
        for p = 1:r
            for i = 1:N^2
                fluxL[p,i] = 0.0;
                for l = 1:r
                    for m = 1:r
                        fluxL[p,i] += Y[m,l,p]*B[m,l,i];
                    end
                end
            end
        end

        for p = 1:r
            for i = 1:N^2
                yL[p,i] = 0.0;
                for l = 1:r
                    yL[p,i] += Y1[p,l]*L[i,l];
                end
                yL[p,i] -= fluxL[p,i];
            end
        end

        L .= L .+ dt*yL';
                
        W,S = qrGramSchmidt(L);
        #W,S = qr(L);
        W = W[:, 1:obj.settings.r];
        S = S[1:obj.settings.r, 1:obj.settings.r];

        S .= S';

        # apply filter step
        Filter(obj,W);
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    # return end time and solution
    return t, X,S,W;
end

function SolveForwardOld(obj::DLRSolver)
    t = 0.0;
    useStabilizingTerms = obj.settings.useStabilizingTerms;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    r = obj.settings.r; # DLR rank
    Nq = obj.settings.Nq;
    N = obj.settings.N; # here, N is the number of quadrature points

    fXi = 0.25;

    # Set up initial condition
    u = SetupIC(obj);

    # Low-rank approx of init data:
    X,S,W = svd(u'); 
    
    # rank-r truncation:
    X = X[:,1:r]; 
    W = W[:,1:r];
    S = Diagonal(S);
    S = S[1:r, 1:r]; 

    A = zeros(r,r,r);
    B = zeros(r,r,Nq^2);
    Y = zeros(r,r,r);
    Y1 = zeros(r,r);

    SFlux = zeros(r,r);
    numFlux = zeros(r);
    K = zeros(Nx,r);
    KNew = X*S;
    L = zeros(N^2,r);

    yK = zeros(Nx,r);
    fluxK = zeros(r);
    yS = zeros(r,r);
    fluxS = zeros(r,r);
    yL = zeros(r,N^2);
    fluxL = zeros(r,N^2);

    PhiQuad = obj.basis.PhiQuad;
    
    Nt = Integer(round(tEnd/dt));

    # compute Dirichlet values if they are independent of time
    uL = zeros(Nq^2);
    uR = zeros(Nq^2);
    for k = 1:Nq
        for q = 1:Nq
            uL[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[1],obj.q.xi[k],obj.q.xi[q])[1];
            uR[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[end],obj.q.xi[k],obj.q.xi[q])[1];
        end
    end

    prog = Progress(Nt,1)
    
    # time loop
    #@showprogress 0.1 "Progress " 
    #@gif 
    for n = 1:Nt

        ###### K-step ######
        K .= X*S;

        # Compute W basis on quadrature points
        WQuad = EvalAtQuad(obj.basis,W)';

        # impose BCs
        for i = 1:r
            K[1,i] = Integral(obj.q,uL.*WQuad[i,:].*fXi);
            K[end,i] = Integral(obj.q,uR.*WQuad[i,:].*fXi);
        end

        # Compute A_{i,j,m} = E[W_i W_j W_m]
        for i = 1:r
            for j = 1:r
                for m = 1:r
                    A[i,j,m] = Integral(obj.q,WQuad[i,:].*WQuad[j,:].*WQuad[m,:].*fXi);
                end
            end
        end

        for j = 2:(Nx-1)
            for p = 1:r
                fluxK[p] = 0.0;
                for l = 1:r
                    for m = 1:r
                        fluxK[p] += 1/4/dx * (K[j+1,l]*K[j+1,m]-K[j-1,l]*K[j-1,m])*A[l,m,p];
                    end
                end
            end
            for p = 1:r
                yK[j,p] = 1/2/dt .* (K[j+1,p]-2*K[j,p]+K[j-1,p]) - fluxK[p];
            end
        end

        K .= K .+ dt*yK;

        XNew,STmp = qrGramSchmidt(K); # optimize bei choosing XFull, SFull

        MUp = XNew' * X;

        ###### L-step ######
        L = W*S';

        for p = 1:r
            for m = 1:r
                for l = 1:r
                    Y[m,l,p] = 0;
                    for j = 2:(Nx-1) 
                        Y[m,l,p] += 1/4/dx * X[j,p]*(X[j+1,l]*X[j+1,m]-X[j-1,l]*X[j-1,m]);
                    end
                end
            end
        end

        # Compute A_{i,j,m} = E[L_i L_j phi_m]
        LQuad = EvalAtQuad(obj.basis,L)';
        for i = 1:r
            for j = 1:r
                for m = 1:N^2
                    B[i,j,m] = Integral(obj.q,LQuad[i,:].*LQuad[j,:].*obj.basis.PhiQuad[:,m].*fXi);
                end
            end
        end
        
        for p = 1:r
            for i = 1:N^2
                fluxL[p,i] = 0.0;
                for l = 1:r
                    for m = 1:r
                        fluxL[p,i] += Y[m,l,p]*B[m,l,i];
                    end
                end
            end
        end

        for p = 1:r
            for l = 1:r
                Y1[p,l] = 0.0;
                for j = 2:(Nx-1)
                    if useStabilizingTerms
                        Y1[p,l] += 1/2/dt * X[j,p]*(X[j+1,l]-2*X[j,l]+X[j-1,l]);
                    end
                end
            end
        end

        for p = 1:r
            for i = 1:N^2
                yL[p,i] = 0.0;
                for l = 1:r
                    yL[p,i] += Y1[p,l]*L[i,l];
                end
                yL[p,i] -= fluxL[p,i];
            end
        end

        L .= L .+ dt*yL';
                
        WNew,STmp = qrGramSchmidt(L);

        NUp = WNew' * W;
        W .= WNew;
        X .= XNew;

        ################## S-step ##################
        S .= MUp*S*(NUp')

        L = W*S';

        # Compute A_{i,j,m} = E[L_i L_j phi_m]
        LQuad = EvalAtQuad(obj.basis,L)';
        WQuad = EvalAtQuad(obj.basis,W)';
        for i = 1:r
            for j = 1:r
                for m = 1:r
                    A[i,j,m] = Integral(obj.q,LQuad[i,:].*LQuad[j,:].*WQuad[m,:].*fXi);
                end
            end
        end

        for p = 1:r
            for l = 1:r
                Y1[p,l] = 0.0;
                for j = 2:(Nx-1)
                    if useStabilizingTerms
                        Y1[p,l] += 1/2/dt * X[j,p]*(X[j+1,l]-2*X[j,l]+X[j-1,l]);
                    end
                end
            end
        end

        for p = 1:r
            for m = 1:r
                for l = 1:r
                    Y[m,l,p] = 0;
                    for j = 2:(Nx-1) 
                        Y[m,l,p] += 1/4/dx * X[j,p]*(X[j+1,l]*X[j+1,m]-X[j-1,l]*X[j-1,m]);
                    end
                end
            end
        end
    
        for q = 1:r
            for l = 1:r
                fluxS[q,l] = 0.0;
                for p = 1:r
                    for m = 1:r
                        fluxS[q,l] += Y[m,p,q]*A[m,p,l];
                    end
                end
            end
            for l = 1:r
                yS[q,l] = 0.0
                for m = 1:r
                    for i = 1:N^2
                        yS[q,l] += Y1[q,m]*L[i,m]*W[i,l];
                    end
                end
                yS[q,l] -= fluxS[q,l];
            end
        end

        S .= S .+ dt.*yS;

        # apply filter step
        Filter(obj,W);
        
        next!(prog) # update progress bar

        t = t+dt;
    end

    #println(norm(uHat))

    # return end time and solution
    return t, X,S,W;
end

function Filter(obj::DLRSolver,W::Array{Float64,2})
    lambda = obj.settings.lambda
    N = obj.settings.N;
    if obj.settings.filterType == "L2"
        for i = 1:N
            for l = 1:N
                W[(l-1)*N+i,:] .= W[(l-1)*N+i,:]/(1+lambda*(i-1)^2*i^2+lambda*(l-1)^2*l^2);
            end
        end
    elseif obj.settings.filterType == "EXP"
        epsilonM = eps(Float64);
        c = log( epsilonM );
        for i = 1:size(W,1)
            eta = i/(obj.settings.N+1)
            W[i,:] .= W[i,:]*exp( c * eta^obj.settings.filterOrder )^(lambda*obj.settings.dt);
        end
    end
end

# SSP Update function for K-step
function UpdateK(obj::TimeSolver,solver::DLRSolver,K::Array{Float64,2},W::Array{Float64,2})
    NCells = solver.settings.Nx;
    obj.KRK[1,:,:] .= K;
    for s = 1:obj.rkStages
        obj.Krhs[s,:,:] .= RhsK(solver,obj.KRK[s,:,:],W);;
        obj.KRK[s+1,:,:] .= zeros(NCells,solver.settings.r);
        for j = 1:s
            obj.KRK[s+1,:,:] = obj.KRK[s+1,:,:]+obj.alpha[s,j].*obj.KRK[j,:,:]+obj.dt*obj.beta[s,j].*obj.Krhs[j,:,:];
        end
    end

    return obj.KRK[obj.rkStages+1,:,:];
end

# SSP Update function for S-step
function UpdateS(obj::TimeSolver,solver::DLRSolver,X::Array{Float64,2},S::Array{Float64,2},W::Array{Float64,2},backward::Bool=true)
    obj.SRK[1,:,:] .= S;
    for s = 1:obj.rkStages
        obj.Srhs[s,:,:] .= RhsS(solver,X,W,W*obj.SRK[s,:,:]');
        if backward # if projector splitting integrator is used, sign in rhsS must be changed
            obj.Srhs[s,:,:] .= -obj.Srhs[s,:,:];
        end
        obj.SRK[s+1,:,:] .= zeros(solver.settings.r,solver.settings.r);
        for j = 1:s
            obj.SRK[s+1,:,:] = obj.SRK[s+1,:,:]+obj.alpha[s,j].*obj.SRK[j,:,:]+obj.dt*obj.beta[s,j].*obj.Srhs[j,:,:];
        end
    end

    return obj.SRK[obj.rkStages+1,:,:];
end

# SSP Update function for L-step
function UpdateL(obj::TimeSolver,solver::DLRSolver,X::Array{Float64,2},L::Array{Float64,2},recompute::Bool=true)
    N = solver.settings.N^2;
    obj.LRK[1,:,:] .= L;
    for s = 1:obj.rkStages
        obj.Lrhs[s,:,:] .= RhsL(solver,X,obj.LRK[s,:,:],recompute)';
        obj.LRK[s+1,:,:] .= zeros(N,solver.settings.r);
        for j = 1:s
            obj.LRK[s+1,:,:] = obj.LRK[s+1,:,:]+obj.alpha[s,j].*obj.LRK[j,:,:]+obj.dt*obj.beta[s,j].*obj.Lrhs[j,:,:];
        end
    end

    return obj.LRK[obj.rkStages+1,:,:];
end