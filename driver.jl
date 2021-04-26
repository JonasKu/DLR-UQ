include("settings.jl")
include("SGSolver.jl")
include("DLRSolver.jl")
include("plotting.jl")

using NPZ;

close("all");

s = Settings();
N = s.N;

# trun on optimized used of stabilizing terms
optimized = false;
#s.N = s.r;

#r = [2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12]
r = [2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13; 14; 15; 16]
errorExpBack = zeros(length(r));
errorVarBack = zeros(length(r));
errorExpFor = zeros(length(r));
errorVarFor = zeros(length(r));
errorExpSG = zeros(length(r));
errorVarSG = zeros(length(r));

s.useStabilizingTermsS = true;
s.useStabilizingTermsL = true;

for k = 1:length(r)
    s.r = r[k];
    if optimized
        s.useStabilizingTermsS = false;
        s.useStabilizingTermsL = true;
    end
    solver = DLRSolver(s);
    @time tEnd, X, S, W = SolveBackward(solver);
    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpBack[k], errorVarBack[k] = L2ErrorExpVar(plotSolution,Array((X*S*W')'))
    println("rank ",r[k]," : ",errorExpBack[k]," ",errorVarBack[k])
end
println("-> DLR Backward DONE.")

s.useStabilizingTermsS = true;
s.useStabilizingTermsL = true;

for k = 1:length(r)
    s.r = r[k];
    if optimized
        s.useStabilizingTermsS = true;
        s.useStabilizingTermsL = false;
    end
    solver = DLRSolver(s);
    @time tEnd, X, S, W = SolveForward(solver);
    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpFor[k], errorVarFor[k] = L2ErrorExpVar(plotSolution,Array((X*S*W')'))
    println("rank ",r[k]," : ",errorExpFor[k]," ",errorVarFor[k])
end
println("-> DLR Forward DONE.")

for k = 1:length(r)
    s.N = r[k];
    solver = Solver(s);
    @time tEnd, u = Solve(solver);
    plotSolution = Plotting(s,solver.basis,solver.q,tEnd);
    errorExpSG[k], errorVarSG[k] = L2ErrorExpVar(plotSolution,u)
    println("rank ",r[k]," : ",errorExpSG[k]," ",errorVarSG[k])
end
println("-> SG DONE.")

fig, ax = subplots(figsize=(15, 8), dpi=100)
#ax.plot(r.^2,errorExpSG, "k--o", linewidth=2, label="SG", alpha=1.0)
ax.plot(r,errorExpBack, "g-.<", linewidth=2, label="DLR", alpha=1.0)
ax.plot(r,errorExpFor, "m:>", linewidth=2, label="unconventional DLR", alpha=1.0)
ylabel("Error Expectation", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.set_ylim([min(minimum(errorExpBack),minimum(errorExpFor)),max(errorExpBack[1],errorExpFor[1])])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/studyExp.png")

fig, ax = subplots(figsize=(15, 8), dpi=100)
#ax.plot(r.^2,errorVarSG, "k--o", linewidth=2, label="SG", alpha=1.0)
ax.plot(r,errorVarBack, "g-.<", linewidth=2, label="DLR", alpha=1.0)
ax.plot(r,errorVarFor, "m:>", linewidth=2, label="unconventional DLR", alpha=1.0)
ylabel("Error Variance", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.set_ylim([min(minimum(errorVarBack),minimum(errorVarFor)),max(errorVarBack[1],errorVarFor[1])])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/studyVar.png")

npzwrite("data/ErrorsExpForDLRNx$(s.Nx)N$(N)tEnd$(s.tEnd)lambda$(s.lambda)RKType$(s.rkType)cfl$(s.cfl)Optimized$(optimized)NCons$(s.NCons).npy", errorExpFor)
npzwrite("data/ErrorsVarForDLRNx$(s.Nx)N$(N)tEnd$(s.tEnd)lambda$(s.lambda)RKType$(s.rkType)cfl$(s.cfl)Optimized$(optimized)NCons$(s.NCons).npy", errorVarFor)
npzwrite("data/ErrorsExpBackDLRNx$(s.Nx)N$(N)tEnd$(s.tEnd)lambda$(s.lambda)RKType$(s.rkType)cfl$(s.cfl)Optimized$(optimized)NCons$(s.NCons).npy", errorExpBack)
npzwrite("data/ErrorsVarBackDLRNx$(s.Nx)N$(N)tEnd$(s.tEnd)lambda$(s.lambda)RKType$(s.rkType)cfl$(s.cfl)Optimized$(optimized)NCons$(s.NCons).npy", errorVarBack)
npzwrite("data/ErrorsExpSGNx$(s.Nx)N$(N)tEnd$(s.tEnd)lambda$(s.lambda)RKType$(s.rkType)cfl$(s.cfl).npy", errorExpSG)
npzwrite("data/ErrorsVarSGNx$(s.Nx)N$(N)tEnd$(s.tEnd)lambda$(s.lambda)RKType$(s.rkType)cfl$(s.cfl).npy", errorVarSG)