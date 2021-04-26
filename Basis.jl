__precompile__
import GSL

struct Basis
    # number of moments
    N::Int64;
    Nq::Int64;
    states::Int64;

    # precomputed Legendre Polynomials at quadrature points
    PhiQuad::Array{Float64,2};
    PhiQuadCons::Array{Float64,2};
    PhiQuadFull::Array{Float64,2};
    PhiQuadW::Array{Float64,2};
    PhiQuadWCons::Array{Float64,2};
    PhiQuadWFull::Array{Float64,2};

    function Basis(quadrature::Quadrature,settings::Settings)
        N = settings.N;
        Nq = quadrature.Nq;
        NCons = settings.NCons;

        # precompute Legendre basis functions at quad points
        PhiQuad = zeros(quadrature.Nq^2,N^2);
        PhiQuadFull = zeros(quadrature.Nq^2,N^2);
        PhiQuadW = zeros(N^2,quadrature.Nq^2);
        PhiQuadWFull = zeros(N^2,quadrature.Nq^2);
        for i = 1:N
            for j = 1:N
                for k = 1:Nq
                    for q = 1:Nq
                        PhiQuad[(q-1)*Nq+k,(j-1)*N+i] = Phi.(i-1,quadrature.xi[k])*Phi.(j-1,quadrature.xi[q]);
                        PhiQuadW[(j-1)*N+i,(q-1)*Nq+k] = Phi.(i-1,quadrature.xi[k])*Phi.(j-1,quadrature.xi[q])*quadrature.w[k]*quadrature.w[q];
                    end
                end
            end
        end
        PhiQuadFull .= PhiQuad;
        PhiQuadWFull .= PhiQuadW;
        
        # save conserved quantities
        PhiQuadCons = zeros(quadrature.Nq^2,NCons);
        PhiQuadWCons = zeros(NCons,quadrature.Nq^2);
        for i = 1:NCons
            for k = 1:Nq
                for q = 1:Nq
                    i1 = settings.iCons[i,1];
                    i2 = settings.iCons[i,2];
                    # set conserved quantities to zero
                    PhiQuad[(q-1)*Nq+k,(i2-1)*N+i1] = 0.0;
                    PhiQuadW[(i2-1)*N+i1,(q-1)*Nq+k] = 0.0;
                    # save conserved basis
                    PhiQuadCons[(q-1)*Nq+k,i] = Phi.(i1-1,quadrature.xi[k])*Phi.(i2-1,quadrature.xi[q]);
                    PhiQuadWCons[i,(q-1)*Nq+k] = PhiQuadCons[(q-1)*Nq+k,i]*quadrature.w[k]*quadrature.w[q];
                end
            end
        end

        new(N,Nq,1,PhiQuad,PhiQuadCons,PhiQuadFull,PhiQuadW,PhiQuadWCons,PhiQuadWFull);
    end
end

# Legendre Polynomials on [-1,1]
function Phi(n::Int64,xi::Float64) ## can I use this with input vectors xVal?
    return sqrt(2.0*n+1.0)*GSL.sf_legendre_Pl.(n,xi);
end

# evaluate polynomial with moments u at spatial position x
function Eval(obj::Basis,u::Array{Float64,1},xi)
    y=zeros(length(xi))
    for i = 1:obj.N
        y = y+u[i]*Phi.(i-1,xi);
    end
    return y;
end

# evalueates states at quadrature points. returns solution[quadpoints,states]
function Eval(obj::Basis,u::Array{Float64,2},xi)
    y=zeros(length(xi),obj.states)
    for s = 1:obj.states
        for i = 1:obj.N
            y[:,s] = y[:,s]+u[i,s]*Phi.(i-1,xi);
        end
    end
    return y;
end

function EvalAtQuad(obj::Basis,u::Array{Float64,1})
    return obj.PhiQuadFull*u;
end

function EvalAtQuadCons(obj::Basis,u::Array{Float64,2})
    return obj.PhiQuadCons*u;
end

function EvalAtQuadDLR(obj::Basis,u::Array{Float64,2})
    return obj.PhiQuad*u;
end

# evalueates states at quadrature points. returns solution[quadpoints,states]
function EvalAtQuad(obj::Basis,u::Array{Float64,2})
    return obj.PhiQuadFull*u;
end

# returns N moments of a function evaluated at the quadrature points and stored in uQ
function ComputeMomentsDLR(obj::Basis,uQ::Array{Float64,1})
    return obj.PhiQuadW*uQ;
end

# returns N moments of a function evaluated at the quadrature points and stored in uQ
function ComputeMoments(obj::Basis,uQ::Array{Float64,1})
    return obj.PhiQuadWFull*uQ;
end

# returns N moments of a function evaluated at the quadrature points and stored in uQ
function ComputeMomentsCons(obj::Basis,uQ::Array{Float64,1})
    return obj.PhiQuadWCons*uQ;
end

# returns N moments of all states evaluated at the quadrature points and stored in uQ
# uQ in [Nq,states], returns [N,states]
function ComputeMoments(obj::Basis,uQ::Array{Float64,2})
    return obj.PhiQuadWFull*uQ;
end

function ComputeMomentsDLR(obj::Basis,uQ::Array{Float64,2})
    println(size(obj.PhiQuadWFull*uQ))
    return obj.PhiQuadW*uQ;
end

# evaluate polynomial with moments u at spatial position x
function Eval(obj::Basis,u::Array{Float64,1},xi,eta)
    y=zeros(length(xi))
    for i = 1:obj.N
        for j = 1:obj.N
            y = y.+u[(j-1)*obj.N+i]*Phi.(i-1,xi).*Phi.(j-1,eta);
        end
    end
    return y;
end