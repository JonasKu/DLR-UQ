using NPZ
using PyPlot

close("all")

rkMethod = "Euler"; # Heun, Euler, SSP

r = [2; 3; 4; 5; 6; 7; 8; 9; 10; 11; 12; 13; 14; 15; 16]

errorExpFor = npzread("data/ErrorsExpForDLRNx600N10tEnd0.01lambda0.0RKType$(rkMethod)cfl$(s.cfl)OptimizedfalseNCons0.npy")
errorVarFor = npzread("data/ErrorsVarForDLRNx600N10tEnd0.01lambda0.0RKType$(rkMethod)cfl$(s.cfl)OptimizedfalseNCons0.npy")
errorExpBack = npzread("data/ErrorsExpBackDLRNx600N10tEnd0.01lambda0.0RKType$(rkMethod)cfl$(s.cfl)OptimizedfalseNCons0.npy")
errorVarBack = npzread("data/ErrorsVarBackDLRNx600N10tEnd0.01lambda0.0RKType$(rkMethod)cfl$(s.cfl)OptimizedfalseNCons0.npy")
errorExpSG = npzread("data/ErrorsExpSGNx600N10tEnd0.01lambda0.0RKTypeEulercfl0.5.npy")
errorVarSG = npzread("data/ErrorsVarSGNx600N10tEnd0.01lambda0.0RKTypeEulercfl0.5.npy")

fig, ax = subplots(figsize=(10, 8), dpi=100)
ax.plot(r.^2,errorExpSG, "k--o", linewidth=2, label="SG", alpha=1.0)
ax.plot(r,errorExpBack, "g-.<", linewidth=2, label="DLRA", alpha=1.0)
ax.plot(r,errorExpFor, "m:>", linewidth=2, label="unconventional DLRA", alpha=1.0)
ylabel("Error Expectation", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/studyExpRKType$(rkMethod).png")

fig, ax = subplots(figsize=(10, 8), dpi=100)
ax.plot(r.^2,errorVarSG, "k--o", linewidth=2, label="SG", alpha=1.0)
ax.plot(r,errorVarBack, "g-.<", linewidth=2, label="DLRA", alpha=1.0)
ax.plot(r,errorVarFor, "m:>", linewidth=2, label="unconventional DLRA", alpha=1.0)
ylabel("Error Variance", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/studyVarRKType$(rkMethod).png")

if false
    errorExpFor = npzread("data/ErrorsExpForDLRNx600N20tEnd0.01lambda0.0RKType$(rkMethod).npy")
    errorVarFor = npzread("data/ErrorsVarForDLRNx600N20tEnd0.01lambda0.0RKType$(rkMethod).npy")
    errorExpBack = npzread("data/ErrorsExpBackDLRNx600N20tEnd0.01lambda0.0RKType$(rkMethod).npy")
    errorVarBack = npzread("data/ErrorsVarBackDLRNx600N20tEnd0.01lambda0.0RKType$(rkMethod).npy")

    fig, ax = subplots(figsize=(10, 8), dpi=100)
    ax.plot(r.^2,errorExpSG, "k--o", linewidth=2, label="SG", alpha=1.0)
    ax.plot(r,errorExpBack, "g-.<", linewidth=2, label=L"DLRA, $N=200$", alpha=1.0)
    ax.plot(r,errorExpFor, "m:>", linewidth=2, label=L"unconventional DLRA, $N=200$", alpha=1.0)
    ylabel("Error Expectation", fontsize=20)
    xlabel("rank/moments", fontsize=20)
    ax.set_xlim([r[1],r[end]])
    ax.legend(loc="upper right", fontsize=20)
    ax.tick_params("both",labelsize=20) 
    fig.canvas.draw() # Update the figure
    PyPlot.savefig("results/studyExpN200.png")

    fig, ax = subplots(figsize=(10, 8), dpi=100)
    ax.plot(r.^2,errorVarSG, "k--o", linewidth=2, label="SG", alpha=1.0)
    ax.plot(r,errorVarBack, "g-.<", linewidth=2, label=L"DLRA, $N=200$", alpha=1.0)
    ax.plot(r,errorVarFor, "m:>", linewidth=2, label=L"unconventional DLRA, $N=200$", alpha=1.0)
    ylabel("Error Variance", fontsize=20)
    xlabel("rank/moments", fontsize=20)
    ax.set_xlim([r[1],r[end]])
    ax.legend(loc="upper right", fontsize=20)
    ax.tick_params("both",labelsize=20) 
    fig.canvas.draw() # Update the figure
    PyPlot.savefig("results/studyVarN200.png")
end


#### FILTERED results
errorFilterExpFor = npzread("data/ErrorsExpForDLRNx600N10tEnd0.01lambda1.0e-5RKType$(rkMethod).npy")
errorFilterVarFor = npzread("data/ErrorsVarForDLRNx600N10tEnd0.01lambda1.0e-5RKType$(rkMethod).npy")
errorFilterExpBack = npzread("data/ErrorsExpBackDLRNx600N10tEnd0.01lambda1.0e-5RKType$(rkMethod).npy")
errorFilterVarBack = npzread("data/ErrorsVarBackDLRNx600N10tEnd0.01lambda1.0e-5RKType$(rkMethod).npy")
errorFilterExpSG = npzread("data/ErrorsExpSGDLRNx600tEnd0.01lambda1.0e-5Euler.npy")
errorFilterVarSG = npzread("data/ErrorsVarSGDLRNx600tEnd0.01lambda1.0e-5Euler.npy")

fig, ax = subplots(figsize=(10, 8), dpi=100)
ax.plot(r.^2,errorFilterExpSG, "k--o", linewidth=2, label="fSG", alpha=1.0)
ax.plot(r,errorFilterExpBack, "g-.<", linewidth=2, label="fDLR", alpha=1.0)
ax.plot(r,errorFilterExpFor, "m:>", linewidth=2, label="unconventional fDLR", alpha=1.0)
ylabel("Error Expectation", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/studyExpFilterRKType$(rkMethod).png")

fig, ax = subplots(figsize=(10, 8), dpi=100)
ax.plot(r.^2,errorFilterVarSG, "k--o", linewidth=2, label="fSG", alpha=1.0)
ax.plot(r,errorFilterVarBack, "g-.<", linewidth=2, label="fDLRA", alpha=1.0)
ax.plot(r,errorFilterVarFor, "m:>", linewidth=2, label="unconventional fDLRA", alpha=1.0)
ylabel("Error Variance", fontsize=20)
xlabel("rank/moments", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/studyVarFilterRKType$(rkMethod).png")

#### compare different rk schemes
errorExpForEuler = npzread("data/ErrorsExpForDLRNx600N10tEnd0.01lambda0.0RKTypeEuler.npy")
errorVarForEuler = npzread("data/ErrorsVarForDLRNx600N10tEnd0.01lambda0.0RKTypeEuler.npy")
errorExpBackEuler = npzread("data/ErrorsExpBackDLRNx600N10tEnd0.01lambda0.0RKTypeEuler.npy")
errorVarBackEuler = npzread("data/ErrorsVarBackDLRNx600N10tEnd0.01lambda0.0RKTypeEuler.npy")

errorExpForHeun = npzread("data/ErrorsExpForDLRNx600N10tEnd0.01lambda0.0RKTypeHeun.npy")
errorVarForHeun = npzread("data/ErrorsVarForDLRNx600N10tEnd0.01lambda0.0RKTypeHeun.npy")
errorExpBackHeun = npzread("data/ErrorsExpBackDLRNx600N10tEnd0.01lambda0.0RKTypeHeun.npy")
errorVarBackHeun = npzread("data/ErrorsVarBackDLRNx600N10tEnd0.01lambda0.0RKTypeHeun.npy")

errorExpForSSP = npzread("data/ErrorsExpForDLRNx600N10tEnd0.01lambda0.0RKTypeSSP.npy")
errorVarForSSP = npzread("data/ErrorsVarForDLRNx600N10tEnd0.01lambda0.0RKTypeSSP.npy")
errorExpBackSSP = npzread("data/ErrorsExpBackDLRNx600N10tEnd0.01lambda0.0RKTypeSSP.npy")
errorVarBackSSP = npzread("data/ErrorsVarBackDLRNx600N10tEnd0.01lambda0.0RKTypeSSP.npy")

fig, ax = subplots(figsize=(10, 8), dpi=100)
ax.plot(r,errorExpBackEuler, "g-.o", linewidth=2, label="Euler", alpha=1.0)
ax.plot(r,errorExpForEuler, "m-.>", linewidth=2, label="Euler unconventional", alpha=1.0)
ax.plot(r,errorExpBackHeun, "g--o", linewidth=2, label="Heun", alpha=1.0)
ax.plot(r,errorExpForHeun, "m-->", linewidth=2, label="Heun unconventional", alpha=1.0)
ax.plot(r,errorExpBackSSP, "g-o", linewidth=2, label="SSP3", alpha=1.0)
ax.plot(r,errorExpForSSP, "m->", linewidth=2, label="SSP3 unconventional", alpha=1.0)
ylabel("Error Expectation", fontsize=20)
xlabel("rank", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/studyExpRKTypeComparison.png")

fig, ax = subplots(figsize=(10, 8), dpi=100)
ax.plot(r,errorVarBackEuler, "g-.o", linewidth=2, label="Euler", alpha=1.0)
ax.plot(r,errorVarForEuler, "m-.>", linewidth=2, label="Euler unconventional", alpha=1.0)
ax.plot(r,errorVarBackHeun, "g--o", linewidth=2, label="Heun", alpha=1.0)
ax.plot(r,errorVarForHeun, "m-->", linewidth=2, label="Heun unconventional", alpha=1.0)
ax.plot(r,errorVarBackSSP, "g-o", linewidth=2, label="SSP3", alpha=1.0)
ax.plot(r,errorVarForSSP, "m->", linewidth=2, label="SSP3 unconventional", alpha=1.0)
ylabel("Error Variance", fontsize=20)
xlabel("rank", fontsize=20)
ax.set_xlim([r[1],r[end]])
ax.legend(loc="upper right", fontsize=20)
ax.tick_params("both",labelsize=20) 
fig.canvas.draw() # Update the figure
PyPlot.savefig("results/studyVarRKTypeComparison.png")
