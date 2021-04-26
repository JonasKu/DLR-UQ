__precompile__
using PyPlot

struct Plotting
    settings::Settings;
    basis::Basis;
    xQuad::Array{Float64,1};
    x::Array{Float64,1}
    tEnd::Float64;
    q::Quadrature;

    function Plotting(settings::Settings,basis::Basis,quadrature::Quadrature,tEnd::Float64)
        new(settings,basis,quadrature.xi,settings.x,tEnd,quadrature);
    end
end

function PlotInX(obj::Plotting,s::Int,u::Array{Float64,3},xi::Array{Float64,1})
    Nq = obj.settings.Nq;
    Nx = obj.settings.Nx;
    NxFine = 1000;
    xFine = range(obj.settings.a,obj.settings.b,length =NxFine)
    
    uExact = zeros(NxFine);
    uPlot = zeros(Nx);

    # start plot
    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
    for k = 1:length(xi)
        for j = 1:Nx
            tmp = Eval(obj.basis,u[:,s,j],xi[k]);
            uPlot[j] = tmp[1];
        end
        #for j = 1:NxFine
            #uExact[j] = obj.settings.solutionExact(obj.tEnd,xFine[j],xi[k])[1];
        #end
        ax.plot(obj.x,uPlot, "r--", linewidth=2, label=L"$u_{N}$", alpha=0.6)
        #ax.plot(xFine,uExact, "k-", linewidth=2, label=L"$u_{ex}$", alpha=0.6)
    end
    #ylimMinus = -2.0;
    #ylimPlus = 15.0
    #ax[:set_ylim]([ylimMinus,ylimPlus])
    ax.set_xlim([obj.settings.a,obj.settings.b])
    ax.set_xlabel("x", fontsize=20);

end

function meshgrid(x, y)
    X = [i for i in x, j in 1:length(y)]
    Y = [j for i in 1:length(x), j in y]
    return X, Y
end

function PlotInXi(obj::Plotting,u::Array{Float64,2},index::Int,info::String)
    Nq = obj.settings.Nq;
    Nx = obj.settings.Nx;
    NxiFine = 500;
    xiFine = range(-1,1,length =NxiFine);
    uExact = zeros(NxiFine,NxiFine);
    uPlot = zeros(NxiFine,NxiFine);
  
    for k = 1:NxiFine
        for q = 1:NxiFine
            tmp = Eval(obj.basis,u[:,index],xiFine[k],xiFine[q]);
            uPlot[k,q] = tmp[1];
            
            uExact[k,q] = obj.settings.solutionExact(obj.tEnd,obj.settings.x[index],xiFine[k],xiFine[q])[1];;#obj.settings.solutionExact(obj.tEnd,obj.x[index],xiFine[k])[1];
        end
    end

    # start plot
    fig = figure(figsize=(9,8)) # Create a figure and save its handle
    ax = gca()
    xgrid, ygrid = meshgrid(xiFine, xiFine)
    surf(xgrid, ygrid, uPlot', cmap=ColorMap("viridis"), alpha=0.7)
    tight_layout()

    PyPlot.savefig("results/PlotXi$(info)Nx$(Nx)N$(obj.settings.N)tEnd$(obj.settings.tEnd).png")

    fig = figure(figsize=(9,8)) # Create a figure and save its handle
    #fig = figure(figsize=PyPlot.figaspect(0.5))
    #ax = gca()
    #ax.set_xlabel(L"\xi_1", fontsize=20);
    #ax.set_ylabel(L"\xi_2")
    xlabel(L"\xi_1", fontsize=20)
    ylabel(L"\xi_2", fontsize=20)
    xgrid, ygrid = meshgrid(xiFine, xiFine)
    surf(xgrid, ygrid, uExact, rstride=2, cstride=2, cmap=ColorMap("viridis"), alpha=0.7)

    #ax.set_zlabel('u')
    tight_layout()
    plt.draw()
    PyPlot.savefig("results/PlotXiExactX$(obj.settings.x[index]).png")
    #zlim(-0.5, 1.0)

end

function ComparePlotInXi(obj::Plotting,s::Int,u::Array{Float64,3},uL1::Array{Float64,3},v::Array{Float64,3},index::Int)
    Nq = obj.settings.Nq;
    Nx = obj.settings.Nx;
    NxiFine = 100;
    xiFine = range(-1,1,length=NxiFine)
    uExact = zeros(NxiFine);
    uPlot = zeros(NxiFine);
    uL1Plot = zeros(NxiFine);
    vPlot = zeros(NxiFine);

    # start plot
    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
    if s == 1
        title(L"$\rho$", fontsize=20)
    elseif s == 2
        title(L"$\rho v$", fontsize=20)
    else
        title(L"$\rho e$", fontsize=20)
    end
    for k = 1:NxiFine
        vVals = Eval(obj.basis,v[:,:,index],xiFine[k])'
        tmp = UKin(obj.closure,vVals[:,1]);
        vPlot[k] = tmp[s];
        tmp = Eval(obj.basis,u[:,s,index],xiFine[k]);
        uPlot[k] = tmp[1];
        tmp = Eval(obj.basis,uL1[:,s,index],xiFine[k]);
        uL1Plot[k] = tmp[1];
        data_rho,data_u,data_P,data_e = obj.settings.solutionExact(obj.tEnd,obj.x[index],xiFine[k]);
        if s == 1
            uExact[k] = data_rho[1];
        elseif s == 2
            uExact[k] = data_rho[1]*data_u[1];
        else
            uExact[k] = data_rho[1]*data_e[1];
        end
    end

    ax.plot(xiFine,uPlot, "r:", linewidth=2, label="SG", alpha=0.6)
    ax.plot(xiFine,uL1Plot, "g--", linewidth=2, label="L1", alpha=0.6)
    ax.plot(xiFine,vPlot, "b-.", linewidth=2, label="IPM", alpha=0.6)
    ax.plot(xiFine,uExact, "k-", linewidth=2, label="exact", alpha=0.6)
    ax.set_xlim([-1.0,1.0])
    ax.tick_params("both",labelsize=20) 
    ax.set_xlabel(L"\xi", fontsize=20);
    ax.legend(loc="upper left", fontsize=20)
    PyPlot.savefig("results/PlotXiState$(s)Nx$(Nx)N$(obj.settings.N)tEnd$(obj.settings.tEnd)Sigma$(obj.settings.sigma).png")
end

function PlotExpectedValue(obj::Plotting,u::Array{Float64,2})
    Nq = obj.settings.Nq;
    Nx = obj.settings.Nx;
    NxFine = 1000;
    xFine = collect(range(obj.settings.a,obj.settings.b,length=NxFine))
    uExact = zeros(NxFine);
    varExact = zeros(NxFine);
    uPlot = zeros(Nx);
    varPlot = zeros(Nx);

    # start plot
    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung

    for j = 1:Nx
        uVals = EvalAtQuad(obj.basis,u[:,j]);
        uPlot[j] = Integral(obj.q,uVals*0.25);
        varPlot[j] = Integral(obj.q,0.25*(uVals.-uPlot[j]).^2);
    end
    varMax = maximum(varPlot);
    expMax = maximum(uPlot);
    qFine = Quadrature(200,"Gauss")
    exactState = zeros(NxFine,qFine.Nq,qFine.Nq);
    for j = 1:NxFine
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                exactState[j,k,l] = obj.settings.solutionExact(obj.tEnd,xFine[j],qFine.xi[k],qFine.xi[l])[1];
            end
        end
    end
    for j = 1:NxFine
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                uExact[j] += exactState[j,k,l]*0.25*qFine.w[k]*qFine.w[l];
            end
        end
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                varExact[j] += (exactState[j,k,l]-uExact[j])^2 * 0.25*qFine.w[k]*qFine.w[l];
            end
        end
    end
    ax.plot(obj.x,uPlot, "k--", linewidth=2, label=L"$E[u_{N}]$", alpha=0.6)
    ax.plot(xFine,uExact, "r-", linewidth=2, label=L"$E[u_{ex}]$", alpha=0.6)
    ylabel("Expectation Value", fontsize=20,color="red")
    ax2 = ax[:twinx]() # Create another axis on top of the current axis
    font2 = ["color"=>"blue"]
    ylabel("Standard deviation", fontsize=20,color="blue")
    ax2.plot(obj.x,sqrt.(varPlot), "k--", linewidth=2, label=L"$V[u_{N}]$", alpha=0.6)
    ax2.plot(xFine,sqrt.(varExact), "b-", linewidth=2, label=L"$V[u_{ex}]$", alpha=0.6)
    setp(ax2[:get_yticklabels](),color="blue") # Y Axis font formatting
    setp(ax[:get_yticklabels](),color="red")
    ax.set_xlim([obj.settings.a,obj.settings.b])
    ax.set_xlabel("x", fontsize=20);
    ax.tick_params("both",labelsize=20) 
    ax2[:tick_params]("both",labelsize=20)
    fig[:canvas][:draw]() # Update the figure
    PyPlot.savefig("test.png")
end

function PlotExpectedValue(obj::Plotting,u::Array{Float64,2},v::Array{Float64,2},info::String="")
    Nq = obj.settings.Nq;
    Nx = obj.settings.Nx;
    NxFine = 1000;
    xFine = collect(range(obj.settings.a,obj.settings.b,length=NxFine))
    uExact = zeros(NxFine);
    varExact = zeros(NxFine);
    uPlot = zeros(Nx);
    varPlot = zeros(Nx);
    vPlot = zeros(Nx);
    varVPlot = zeros(Nx);

    # start plot
    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung

    for j = 1:Nx
        uVals = EvalAtQuad(obj.basis,u[:,j]);
        uPlot[j] = Integral(obj.q,uVals*0.25);
        varPlot[j] = Integral(obj.q,0.25*(uVals.-uPlot[j]).^2);
        vVals = EvalAtQuad(obj.basis,v[:,j]);
        vPlot[j] = Integral(obj.q,vVals*0.25);
        varVPlot[j] = Integral(obj.q,0.25*(vVals.-vPlot[j]).^2);
    end
    varMax = maximum(varPlot);
    expMax = maximum(uPlot);
    qFine = Quadrature(200,"Gauss")
    exactState = zeros(NxFine,qFine.Nq,qFine.Nq);
    for j = 1:NxFine
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                exactState[j,k,l] = obj.settings.solutionExact(obj.tEnd,xFine[j],qFine.xi[k],qFine.xi[l])[1];
            end
        end
    end
    for j = 1:NxFine
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                uExact[j] += exactState[j,k,l]*0.25*qFine.w[k]*qFine.w[l];
            end
        end
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                varExact[j] += (exactState[j,k,l]-uExact[j])^2 * 0.25*qFine.w[k]*qFine.w[l];
            end
        end
    end
    ax.plot(obj.x,uPlot, "g-.", linewidth=2, label=L"DLRA$_9$", alpha=1.0)
    ax.plot(obj.x,vPlot, "m:", linewidth=2, label=L"unconventional DLRA$_9$", alpha=1.0)
    ylabel("Expectation", fontsize=20,color="red")
    ax.plot(xFine,uExact, "r-", linewidth=2, alpha=0.5)
    ax2 = ax[:twinx]() # Create another axis on top of the current axis
    ylabel("Standard deviation", fontsize=20,color="blue")
    ax2.plot(obj.x,sqrt.(varPlot), "g-.", linewidth=2, label="SG", alpha=1.0)
    ax2.plot(obj.x,sqrt.(varVPlot), "m:", linewidth=2, label="DLRA", alpha=1.0)
    #ax2[:set_position](new_position) # Position Method 2
    setp(ax2[:get_yticklabels](),color="blue") # Y Axis font formatting
    setp(ax[:get_yticklabels](),color="red")
    ax2.plot(xFine,sqrt.(varExact), "b-", linewidth=2, alpha=0.5)
    #ylimMinus = -0.5;
    #ylimPlus = 16.0
    #ax[:set_ylim]([ylimMinus,ylimPlus])
    ax.set_xlim([obj.settings.a,obj.settings.b])
    ax.set_xlabel("x", fontsize=20);
    ax.legend(loc="upper right", fontsize=20)
    ax.tick_params("both",labelsize=20) 
    ax2.tick_params("both",labelsize=20)
    fig.canvas.draw() # Update the figure
    PyPlot.savefig("results/ExpectedValueVar2D$(info)Nx$(Nx)N$(obj.settings.N)tEnd$(obj.settings.tEnd)r$(obj.settings.r)lambda$(s.lambda).png")
end

function PlotExpectedValue(obj::Plotting,u::Array{Float64,2},v::Array{Float64,2},w::Array{Float64,2},info::String="")
    Nq = obj.settings.Nq;
    Nx = obj.settings.Nx;
    NxFine = 1000;
    xFine = collect(range(obj.settings.a,obj.settings.b,length=NxFine))
    uExact = zeros(NxFine);
    varExact = zeros(NxFine);
    uPlot = zeros(Nx);
    varPlot = zeros(Nx);
    vPlot = zeros(Nx);
    varVPlot = zeros(Nx);
    wPlot = zeros(Nx);
    varWPlot = zeros(Nx);

    # start plot
    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung

    for j = 1:Nx
        uVals = EvalAtQuad(obj.basis,u[:,j]);
        uPlot[j] = Integral(obj.q,uVals*0.25);
        varPlot[j] = Integral(obj.q,0.25*(uVals.-uPlot[j]).^2);
        vVals = EvalAtQuad(obj.basis,v[:,j]);
        vPlot[j] = Integral(obj.q,vVals*0.25);
        varVPlot[j] = Integral(obj.q,0.25*(vVals.-vPlot[j]).^2);
        wVals = EvalAtQuad(obj.basis,w[:,j]);
        wPlot[j] = Integral(obj.q,wVals*0.25);
        varWPlot[j] = Integral(obj.q,0.25*(wVals.-wPlot[j]).^2);
    end
    varMax = maximum(varPlot);
    expMax = maximum(uPlot);
    qFine = Quadrature(200,"Gauss")
    exactState = zeros(NxFine,qFine.Nq,qFine.Nq);
    for j = 1:NxFine
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                exactState[j,k,l] = obj.settings.solutionExact(obj.tEnd,xFine[j],qFine.xi[k],qFine.xi[l])[1];
            end
        end
    end
    for j = 1:NxFine
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                uExact[j] += exactState[j,k,l]*0.25*qFine.w[k]*qFine.w[l];
            end
        end
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                varExact[j] += (exactState[j,k,l]-uExact[j])^2 * 0.25*qFine.w[k]*qFine.w[l];
            end
        end
    end
    if info == "noFilter"
        ax.plot(obj.x,uPlot, "k--", linewidth=2, label=L"SG$_{100}$", alpha=1.0)
        ax.plot(obj.x,vPlot, "g-.", linewidth=2, label=L"DLRA$_{9}$", alpha=1.0)
        ax.plot(obj.x,wPlot, "m:", linewidth=2, label=L"unconventional DLRA$_{9}$", alpha=1.0)
    else
        ax.plot(obj.x,uPlot, "k--", linewidth=2, label=L"fSG$_{100}$", alpha=1.0)
        ax.plot(obj.x,vPlot, "g-.", linewidth=2, label=L"fDLRA$_{9}$", alpha=1.0)
        ax.plot(obj.x,wPlot, "m:", linewidth=2, label=L"unconventional fDLRA$_{9}$", alpha=1.0)
    end
    
    ylabel("Expectation", fontsize=20,color="red")
    ax.plot(xFine,uExact, "r-", linewidth=2, alpha=0.5)
    ax2 = ax[:twinx]() # Create another axis on top of the current axis
    ylabel("Standard deviation", fontsize=20,color="blue")
    ax2.plot(obj.x,sqrt.(varPlot), "k--", linewidth=2, label="SG", alpha=1.0)
    ax2.plot(obj.x,sqrt.(varVPlot), "g-.", linewidth=2, label="DLRA", alpha=1.0)
    ax2.plot(obj.x,sqrt.(varWPlot), "m:", linewidth=2, label="unconventional DLRA", alpha=1.0)
    #ax2[:set_position](new_position) # Position Method 2
    setp(ax2[:get_yticklabels](),color="blue") # Y Axis font formatting
    setp(ax[:get_yticklabels](),color="red")
    ax2.plot(xFine,sqrt.(varExact), "b-", linewidth=2, alpha=0.5)
    #ylimMinus = -0.5;
    #ylimPlus = 16.0
    #ax[:set_ylim]([ylimMinus,ylimPlus])
    ax.set_xlim([obj.settings.a,obj.settings.b])
    ax.set_xlabel("x", fontsize=20);
    ax.legend(loc="upper right", fontsize=20)
    ax.tick_params("both",labelsize=20) 
    ax2.tick_params("both",labelsize=20)
    fig.canvas.draw() # Update the figure
    PyPlot.savefig("results/ExpectedValueVar2D$(info)Nx$(Nx)N$(obj.settings.N)tEnd$(obj.settings.tEnd)r$(obj.settings.r)lambda$(s.lambda).png")
end

function CompareExpectedValue(obj::Plotting,s::Int,u::Array{Float64,3},uL1::Array{Float64,3},v::Array{Float64,3},x)
    Nq = obj.settings.Nq;
    Nx = length(x);
    NxFine = 1000;
    xFine = range(obj.settings.a,obj.settings.b,length=NxFine)
    uExact = zeros(NxFine);
    varExact = zeros(NxFine);
    uPlot = zeros(Nx);
    varPlot = zeros(Nx);
    vPlot = zeros(Nx);
    varVPlot = zeros(Nx);
    uL1Plot = zeros(Nx);
    varL1Plot = zeros(Nx);

    # start plot
    fig, ax = subplots(figsize=(15, 8), dpi=100)#, facecolor='w', edgecolor='k') # dpi Aufloesung
    if s == 1
        title(L"$\rho$", fontsize=20)
    elseif s == 2
        title(L"$\rho v$", fontsize=20)
    else
        title(L"$\rho e$", fontsize=20)
    end
    
    for j = 1:Nx
        uPlot[j] = u[1,s,j];
        varPlot[j] = u[2:end,s,j]'u[2:end,s,j];
        uL1Plot[j] = uL1[1,s,j];
        varL1Plot[j] = uL1[2:end,s,j]'uL1[2:end,s,j];
        uVals = UKin(obj.closure,EvalAtQuad(obj.basis,v[:,:,j]));
        vPlot[j] = IntegralVec(obj.q,uVals[:,s]*0.5);
        varVPlot[j] = IntegralVec(obj.q,0.5*(uVals[:,s]-uPlot[j]).^2);
    end
    varMax = maximum(varPlot);
    expMax = maximum(uPlot);

    qFine = Quadrature(300,"Gauss")
    exactState = zeros(NxFine,qFine.Nq);
    for k = 1:qFine.Nq
        #data_rho,data_u,data_P,data_e = analytic_sod(obj.tEnd,obj.settings.x0,obj.settings.rhoL,obj.settings.pL,obj.settings.uL,obj.settings.rhoR,
        #                                obj.settings.pR,obj.settings.uR,obj.settings.gamma,xFine);
        data_rho,data_u,data_P,data_e = obj.settings.solutionExact(obj.tEnd,xFine,qFine.xi[k]);
        if s == 1
            exactState[:,k] = data_rho;
        elseif s == 2
            exactState[:,k] = data_rho.*data_u;
        else
            exactState[:,k] = data_rho.*data_e;
        end
    end
    for j = 1:NxFine
        uExact[j] = IntegralVec(qFine, exactState[j,:]*0.5,-1.0,1.0);
        varExact[j] = IntegralVec(qFine, (exactState[j,:].-uExact[j]).^2*0.5,-1.0,1.0)
    end
    
    ax.plot(x,uPlot, "y--", linewidth=2, label="SG", alpha=1.0)
    ax.plot(x,uL1Plot, "k-.", linewidth=2, label="L1", alpha=1.0)
    ax.plot(x,vPlot, "g-", linewidth=2, label="IPM", alpha=1.0)
    ylabel("Expectation", fontsize=20,color="red")
    ax.plot(xFine,uExact, "r:", linewidth=2, alpha=1.0)
    ax2 = ax[:twinx]() # Create another axis on top of the current axis
    ylabel("Standard deviation", fontsize=20,color="blue")
    ax2.plot(x,sqrt.(varPlot), "y--", linewidth=2, label="SG", alpha=1.0)
    ax2.plot(x,sqrt.(varL1Plot), "k-.", linewidth=2, label="SG", alpha=1.0)
    ax2.plot(x,sqrt.(varVPlot), "g-", linewidth=2, label="IPM", alpha=1.0)
    #ax2[:set_position](new_position) # Position Method 2
    setp(ax2[:get_yticklabels](),color="blue") # Y Axis font formatting
    setp(ax[:get_yticklabels](),color="red")
    ax2.plot(xFine,varExact, "b:", linewidth=2, alpha=1.0)
    #ylimMinus = -0.5;
    #ylimPlus = 16.0
    #ax[:set_ylim]([ylimMinus,ylimPlus])
    ax.set_xlim([obj.settings.a,obj.settings.b])
    ax.set_xlabel("x", fontsize=20);
    ax.legend(loc="upper right", fontsize=20)
    ax.tick_params("both",labelsize=20) 
    ax2[:tick_params]("both",labelsize=20)
    fig[:canvas][:draw]() # Update the figure
    PyPlot.savefig("results/PlotXiState$(s)Nx$(Nx)N$(obj.settings.N)tEnd$(obj.settings.tEnd)Sigma$(obj.settings.sigma).png")
end

function PlotExactSolution(obj::Plotting,s::Int,xi::Float64)
    NxFine = 1000;
    xFine = range(obj.settings.a,obj.settings.b,length=NxFine)
    uExact = zeros(NxFine);
    for j = 1:NxFine
        data_rho,data_u,data_P,data_e = obj.settings.solutionExact(obj.tEnd,xFine[j],xi);
        uExact[j] = data_rho[1];
    end
    fig, ax = subplots(figsize=(15, 8), dpi=100)
    ax.plot(xFine,uExact, "r-", linewidth=2, alpha=0.6)
end

function L2Error(obj::Plotting,u::Array{Float64,2},t::Float64)
    Nx = obj.settings.Nx;
    x = obj.settings.x;
    error = 0;
    for j = 1:Nx
        error = error + obj.settings.dx*Integral(obj.q, xi-> 0.5*( obj.settings.solutionExact(t,x[j],xi)-Eval(obj.basis,u[j,:],xi)).^2,-1.0,1.0)
    end
    return sqrt(error);
end

function L1Error(obj::Plotting,u::Array{Float64,2},t::Float64)
    Nx = obj.settings.Nx;
    x = obj.settings.x;
    error = 0;
    for j = 1:Nx
        error = error + obj.settings.dx*Integral(obj.q, xi-> 0.5*abs.( obj.settings.solutionExact(t,x[j],xi)-Eval(obj.basis,u[j,:],xi)),-1.0,1.0)
    end
    return error;
end

function L2ErrorExpVar(obj::Plotting,u::Array{Float64,2})
    Nx = obj.settings.Nx;
    x = obj.settings.x;
    errorExp = 0.0;
    errorVar = 0.0;

    qFine = Quadrature(200,"Gauss")
    exactState = zeros(Nx,qFine.Nq,qFine.Nq);
    uExact = zeros(Nx);
    varExact = zeros(Nx);
    for j = 1:Nx
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                exactState[j,k,l] = obj.settings.solutionExact(obj.tEnd,x[j],qFine.xi[k],qFine.xi[l])[1];
            end
        end
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                uExact[j] += exactState[j,k,l]*0.25*qFine.w[k]*qFine.w[l];
            end
        end
        for k = 1:qFine.Nq
            for l = 1:qFine.Nq
                varExact[j] += (exactState[j,k,l]-uExact[j])^2 * 0.25*qFine.w[k]*qFine.w[l];
            end
        end
    end

    for j = 1:Nx
        uVals = EvalAtQuad(obj.basis,u[:,j]);
        expN = Integral(obj.q,uVals*0.25);
        varN = Integral(obj.q,0.25*(uVals.-expN).^2);
        expEx = uExact[j];
        varEx = varExact[j];
        errorExp = errorExp + obj.settings.dx*(expEx-expN)^2;
        errorVar = errorVar + obj.settings.dx*(varEx-varN)^2;
    end
    return sqrt(errorExp),sqrt(errorVar);
end

function L1ErrorExpVar(obj::Plotting,u::Array{Float64,2},t::Float64)
    Nx = obj.settings.Nx;
    x = obj.settings.x;
    errorExp = 0.0;
    errorVar = 0.0;
    for j = 1:Nx
        expN = u[j,1];
        varN = u[j,2:end]'u[j,2:end];
        expEx = Integral(obj.q, xi-> obj.settings.solutionExact(obj.tEnd,x[j],xi)*0.5,-1.0,1.0)
        varEx = Integral(obj.q, xi-> (obj.settings.solutionExact(obj.tEnd,x[j],xi)-expEx).^2*0.5,-1.0,1.0);
        errorExp = errorExp + obj.settings.dx*abs(expEx-expN);
        errorVar = errorVar + obj.settings.dx*abs(varEx-varN);
    end
    return errorExp,errorVar;
end
