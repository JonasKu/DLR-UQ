__precompile__

mutable struct Settings
    # grid settings
    # number spatial interfaces
    Nx::Int64;
    # start and end point
    a::Float64;
    b::Float64;
    # grid cell width
    dx::Float64

    # time settings
    # end time
    tEnd::Float64;
    # time increment
    dt::Float64;
    # cfl number
    cfl::Float64;
    
    # number of quadrature points
    Nq::Int64;
    # definition of quadrature type
    quadratureType::String;

    # maximal polynomial degree
    N::Int64;

    # DLR rank
    r::Int64

    # solution values
    uL::Float64;
    uR::Float64;

    # initial shock position at xi = 0
    x0::Float64;

    # grid
    x

    sigma::Float64;

    # spatial limiter
    limiterType::String;
    # time update type
    rkType::String;
    # number of RK stages
    rkStages::Int;
    # filter
    filterType::String;
    lambda::Float64;
    filterOrder::Int64;

    useStabilizingTermsS::Bool;
    useStabilizingTermsL::Bool;
    stabilization::Int;

    # conservation settings
    NCons::Int; # number of conserved basis functions
    iCons; #::Array{Int64,2}; # indices of conserved basis funtions

    # initial condition
    IC::Function;
    solutionExact::Function;

    function Settings()
        # define spatial grid
        Nx = 600;
        a = 0.0;
        b = 1.0;
        x = range(a,stop = b,length = Nx)
        dx = (b-a)/(Nx-1.0);

        # define number of stochastic moments
        N = 10; #10

        # define DLR rank (if DLR is chosen)
        r = 9; #25

        # define time settings
        tEnd = 0.01;#0.01;#0.01115;
        cfl = 0.5;

        # define test case settings
        x0 = 0.3; # 0.5
        x1 = 0.6;
        uL = 12.0;
        uM = 6.0;
        uR = 1.0;
        sigma0 = 0.2;
        sigma1 = 5.0;#5.0;

        quadratureType = "Gauss"; # Possibilities are Gauss and ClenshawCurtis

        # compute time step size
        dt = cfl*dx/uL;

        # determine quadrature points
        Nq = ceil(1.5*N+1);

        # filter parameters
        filterType = "L2" # L2, EXP
        lambda = 0.0;#0.00001
        filterOrder = 1;

        # limiter type
        limiterType = "None";#"Minmod"

        # Runge-Kutta type Euler, Heun, SSP
        rkType = "Euler"
        rkStages = 1;

        useStabilizingTermsS = true;
        useStabilizingTermsL = true;

        # stabilization method for PS: 0 - standard, 1 - Lax Wendroff in L and S, 2 - stabilization in all steps
        stabilization = 0;

        # conservation settings
        NCons = 2;#3; # number of conserved basis functions
        iCons = [1 1; 1 2];#[1 1;1 2;1 3]; # indices of conserved basis funtions

        #NCons = 0;#3; # number of conserved basis functions
        #iCons = 0;
        
        # build class 
        new(Nx,a,b,dx,tEnd,dt,cfl,Nq,quadratureType,N,r,uL,uR,x0,x,sigma0,limiterType,rkType,rkStages,filterType,lambda,filterOrder,useStabilizingTermsS,useStabilizingTermsL,stabilization,NCons,iCons,
            #(xV,xi,eta)->IC3(xV,xi,eta,sigma0,sigma1,uL,uM,uR,x0,x1),
            #(t,xV,xi,eta)->IC3Exact(t,xV,xi,eta,sigma0,sigma1,uL,uM,uR,x0,x1))
            (xV,xi,eta)->IC5(xV,xi,eta,sigma0,sigma1,uL,uR,x0),
            (t,xV,xi,eta)->IC5Exact(t,xV,xi,eta,sigma0,sigma1,uL,uR,x0))
    end

end

function IC1(x,xi,sigma,uL,uR,x0,x1)
    y = zeros(size(xi));
    for j = 1:length(y);
        if x < x0+sigma*xi[j]
            y[j] = uL;
        elseif x < x1+sigma*xi[j]
            y[j] = uL + (uR - uL)*(x-sigma*xi[j]-x0)/(x1-x0);
        else
            y[j] = uR;
        end
    end
    return y;
end

function IC1Exact(t::Float64,x,xi::Float64,sigma::Float64,uL::Float64,uR::Float64,x0::Float64,x1::Float64)
    y = zeros(length(x));

    if t >= (x1-x0)/(uL-uR);
        tS = (x1-x0)/(uL-uR);
        x0BeforeShock = x0+sigma*xi + tS*uL;
        x1BeforeShock = x1+sigma*xi + tS*uR;
        x0 = x0BeforeShock + (t-tS)*(uL+uR)*0.5;
        x1 = x0 - 1.0;
    else
        x0 = x0+sigma*xi + t*uL;
        x1 = x1+sigma*xi + t*uR;
    end

    for j = 1:length(y);
        if x[j] < x0
            y[j] = uL;
        elseif x[j] < x1
            y[j] = uL + (uR - uL)*(x[j]-x0)/(x1-x0);
        else
            y[j] = uR;
        end
    end

    return y;
end

function IC2(x,xi,sigma,uL,uR,x0,x1)
    y = zeros(size(xi));
    K0 = 12;
    K1 = 1;
    div = 1/(x0^3+3*x0*x1^2-x1^3-3*x1*x0^2);
    a = -2*(K0-K1)*div;
    b = 3*(K0-K1)*(x0+x1)*div;
    c = -6*(K0-K1)*x0*x1*div;
    d = (-K0*x1^3+3*K0*x0*x1^2+K1*x0^3-3*K1*x1*x0^2)*div;
    
    for k = 1:length(xi)
        if x-sigma*xi[k] < x0
            y[k] = K0;
        elseif x-sigma*xi[k] < x1
            y[k] = a*(x-sigma*xi[k])^3+b*(x-sigma*xi[k])^2+c*(x-sigma*xi[k])+d;
        else
            y[k] = K1;
        end
    end
    return y;
end

function IC2Exact(t::Float64,x::Float64,xi,sigma,uL,uR,x0,x1,problem,advectionSpeed)
    y = zeros(size(xi));
    K0 = 12;
    K1 = 1;
    div = 1/(x0^3+3*x0*x1^2-x1^3-3*x1*x0^2);
    a = -2*(K0-K1)*div;
    b = 3*(K0-K1)*(x0+x1)*div;
    c = -6*(K0-K1)*x0*x1*div;
    d = (-K0*x1^3+3*K0*x0*x1^2+K1*x0^3-3*K1*x1*x0^2)*div;
    
    for k = 1:length(xi)
        if x-sigma*xi[k] < x0+t*advectionSpeed
            y[k] = K0;
        elseif x-sigma*xi[k] < x1+t*advectionSpeed
            y[k] = a*(x-sigma*xi[k]-t*advectionSpeed)^3+b*(x-sigma*xi[k]-t*advectionSpeed)^2+c*(x-sigma*xi[k]-t*advectionSpeed)+d;
        else
            y[k] = K1;
        end
    end
    return y;
end

function IC3Exact(t::Float64,x,xi,eta,sigma0,sigma1,uL,uM,uR,x0,x1)
    y = 0.0
    println(x);
    for j = 1:length(y);
        tStar = 2.0*(x0-x1)/(uR-uL-sigma0*xi[j]);
        if t < tStar
            v0 = 0.5*(uL+uM+sigma0*xi[j]+sigma1*eta[j]);
            v1 = 0.5*(uR+uM+sigma1*eta[j]);
            x0T = x0+v0*t;
            x1T = x1+v1*t
            if x < x0T
                y[j] = uL+ sigma0*xi[j];
            elseif x < x1T
                y[j] = uM+ sigma1*eta[j];
            else
                y[j] = uR;
            end
        end
        if t >= tStar
            xStar = (x0-x1)*(uM-sigma1*eta+uR)/(uR-uL-sigma0*xi)+x1;
            vStar = 0.5*(uL + sigma0*xi +uR);
            if x < xStar + (t-tStar)*vStar
                y[j] = uL + sigma0*xi[j];
            else
                y[j] = uR;
            end
        end
    end
    return y;
end

function IC3(x,xi,eta,sigma0,sigma1,uL,uM,uR,x0,x1)
    y = zeros(length(xi));
    for j = 1:length(y);
        if x < x0
            y[j] = uL + sigma0*xi[j];
        elseif x < x1
            y[j] = uM + sigma1*eta[j];
        else
            y[j] = uR;
        end
    end
    return y;
end

function IC4(x,xi,eta,sigma0,sigma1,uL,uM,uR,x0,x1)
    y = zeros(length(xi));
    for j = 1:length(y);
        if x < x0+ sigma0*xi[j]
            y[j] = uL + sigma1*eta[j];
        elseif x < x1
            y[j] = uM;;
        else
            y[j] = uR;
        end
    end
    return y;
end

function IC4Exact(t::Float64,x,xi::Float64,eta::Float64,sigma0::Float64,sigma1::Float64,uL::Float64,uM::Float64,uR::Float64,x0::Float64,x1::Float64)
    y = zeros(length(x));
    uLXi = uL+eta*sigma1;
    tS = (x0+sigma0*xi-x1)/(uM-uLXi);
    #println("tS = ",tS)
    #println("x0 = ",x0+sigma0*xi + tS*uLXi+ (t-tS)*(uLXi+uR)*0.5)

    if t >= tS;
        x0BeforeShock = x0+sigma0*xi + tS*uLXi;
        x0 = x0BeforeShock + (t-tS)*(uLXi+uR)*0.5;
        x1 = x0 - 1.0;
    else
        x0 = x0+sigma0*xi + t*uLXi;
        x1 = x1 + t*uM;
    end

    for j = 1:length(y);
        if x[j] < x0
            y[j] = uLXi;
        elseif x[j] < x1
            y[j] = uM;
        else
            y[j] = uR;
        end
    end

    return y;
end

function IC5(x,xi,eta,sigma0::Float64,sigma1::Float64,uL::Float64,uR::Float64,x0::Float64)
    y = zeros(length(xi));
    for j = 1:length(y);
        if x < x0+ sigma0*xi[j]
            y[j] = uL;
        else
            y[j] = uR+ sigma1*(eta[j]+1)*0.5;
        end
    end
    return y;
end

function IC5Exact(t::Float64,x,xi::Float64,eta::Float64,sigma0::Float64,sigma1::Float64,uL::Float64,uR::Float64,x0::Float64)
    
    y = zeros(length(x));
    uRXi = uR+ sigma1*(eta+1)*0.5;

    x0 = x0+sigma0*xi + t*(uL+uRXi)/2;

    for j = 1:length(y);
        if x[j] < x0
            y[j] = uL;
        else
            y[j] = uRXi;
        end
    end

    return y;
end