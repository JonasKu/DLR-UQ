__precompile__
include("quadrature.jl")
include("Basis.jl")

using ProgressMeter

struct Solver
    # spatial grid of cell interfaces
    x;

    # quadrature
    q::Quadrature;

    # Solver settings
    settings::Settings;

    # spatial basis functions
    basis::Basis;

    # constructor
    function Solver(settings)
        x = settings.x;
        q = Quadrature(settings.Nq,"Gauss");
        basis = Basis(q,settings);

        new(x,q,settings,basis);
    end
end

# physical flux Scalar valued Burgers
function f(u::Float64)
    return 0.5*u^2.0;
end

# physical flux Vector valued Burgers
function f(u::Array{Float64,1})
    return 0.5*u.^2.0;
end

# Lax-Friedrichs (LF) flux for Burgers
function numFlux(beta::Float64,u::Float64,v::Float64)
    #beta = max(abs(u),abs(v)); # max of abs(f'(u)) 
    #beta = obj.dx/obj.dt/2;
    return 0.5*(f(u)+f(v))-beta.*(v-u);
end

function SetupIC(obj::Solver)
    u = zeros(obj.settings.N*obj.settings.N,obj.settings.Nx);
    uVals = zeros(obj.settings.Nq^2);
    for j = 1:obj.settings.Nx
        for q = 1:obj.settings.Nq
            for k = 1:obj.settings.Nq
                uVals[(k-1)*obj.settings.Nq+q] = obj.settings.IC(obj.x[j],obj.q.xi[k],obj.q.xi[q])[1];
            end
        end
        u[:,j] = ComputeMoments(obj.basis,uVals*0.25);
    end
    return u;
end

function minmod(a::Float64,b::Float64)
    y = 0;
    if ( abs(a) < abs(b)) && a*b > 0
        y = a;
    elseif ( abs(b) < abs(a)) && a*b > 0
        y = b;
    end
    return y;
end

function Slope(obj::Solver,u::Array{Float64,1},v::Array{Float64,1},w::Array{Float64,1})
    if obj.settings.limiterType == "Minmod"
        return minmod.(w.-v,v.-u)/obj.settings.dx;
    else 
        return 0.0;
    end
end

function Solve(obj::Solver)
    t = 0.0;
    dt = obj.settings.dt;
    dx = obj.x[2]-obj.x[1];
    tEnd = obj.settings.tEnd;
    Nx = obj.settings.Nx;
    N = obj.settings.N; # number of moments

    # Set up initial condition
    u = SetupIC(obj);
    uNew = deepcopy(u);
    
    Nt = round(tEnd/dt);
    
    # time loop
    @showprogress 0.1 "Progress " for n = 1:Nt

        # Update time by dt
        for j = 3:(Nx-2)
            uQjMM = EvalAtQuad(obj.basis,u[:,j-2]);
            uQjM = EvalAtQuad(obj.basis,u[:,j-1]);
            uQj = EvalAtQuad(obj.basis,u[:,j]);
            uQjP = EvalAtQuad(obj.basis,u[:,j+1]);
            uQjPP = EvalAtQuad(obj.basis,u[:,j+2]);
            numFRight = numFlux.( dx/dt/2,uQj.+0.5*dx*Slope(obj,uQjM,uQj,uQjP),uQjP.-0.5*dx*Slope(obj,uQj,uQjP,uQjPP));
            numFLeft = numFlux.( dx/dt/2,uQjM.+0.5*dx*Slope(obj,uQjMM,uQjM,uQj),uQj.-0.5*dx*Slope(obj,uQjM,uQj,uQjP));
            for i = 1:N
                uNew[:,j] = u[:,j].-dt/dx*ComputeMoments(obj.basis,(numFRight.-numFLeft)*0.25);
            end
        end
        u .= uNew;
        
        # apply filter step
        Filter(obj,u); 
        t = t+dt;
    end

    # return end time and solution
    return t, u;

end

function Filter(obj::Solver,u::Array{Float64,2})
    lambda = obj.settings.lambda
    N = obj.settings.N;
    if obj.settings.filterType == "L2"
        for j = 1:size(u,2)
            for i = 1:N
                for l = 1:N
                    #u[(l-1)*N+i,j] = u[(l-1)*N+i,j]/(1+lambda*(i-1)^2*i^2*(l-1)^2*l^2);
                    u[(l-1)*N+i,j] = u[(l-1)*N+i,j]/(1+lambda*(i-1)^2*i^2+lambda*(l-1)^2*l^2);
                end
            end
        end
    elseif obj.settings.filterType == "EXP"
        epsilonM = eps(Float64);
        c = log( epsilonM );
        for j = 1:size(u,2)
            for i = 1:size(u,1)
                eta = i/(obj.settings.N+1)
                u[i,j] = u[i,j]*exp( c * eta^obj.settings.filterOrder )^(lambda*obj.settings.dt);
            end
        end
    end
end