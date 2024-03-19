#pragma once 
#include "AMDGPU.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"

using namespace llvm;

extern cl::opt<bool>
EnableEarlyIfConversion;

extern cl::opt<bool>
OptExecMaskPreRA;

extern cl::opt<bool>
    LowerCtorDtor;

// Option to disable vectorizer for tests.
extern cl::opt<bool> EnableLoadStoreVectorizer;

// Option to control global loads scalarization
extern cl::opt<bool> ScalarizeGlobal;

// Option to run internalize pass.
extern cl::opt<bool> InternalizeSymbols;

// Option to inline all early.
extern cl::opt<bool> EarlyInlineAll;

extern cl::opt<bool> RemoveIncompatibleFunctions;

extern cl::opt<bool> EnableSDWAPeephole;

extern cl::opt<bool> EnableDPPCombine;

// Enable address space based alias analysis
extern cl::opt<bool> EnableAMDGPUAliasAnalysis;

// Option to run late CFG structurizer
extern cl::opt<bool, true> LateCFGStructurize;

// Disable structurizer-based control-flow lowering in order to test convergence
// control tokens. This should eventually be replaced by the wave-transform.
extern cl::opt<bool, true> DisableStructurizer;

// Enable lib calls simplifications
extern cl::opt<bool> EnableLibCallSimplify;

extern cl::opt<bool> EnableLowerKernelArguments;

extern cl::opt<bool> EnableRegReassign;

extern cl::opt<bool> OptVGPRLiveRange;

extern cl::opt<ScanOptions> AMDGPUAtomicOptimizerStrategy;

// Enable Mode register optimization
extern cl::opt<bool> EnableSIModeRegisterPass;

// Enable GFX11.5+ s_singleuse_vdst insertion
extern cl::opt<bool>
    EnableInsertSingleUseVDST;

// Enable GFX11+ s_delay_alu insertion
extern cl::opt<bool>
    EnableInsertDelayAlu;

// Enable GFX11+ VOPD
extern cl::opt<bool>
    EnableVOPD;

// Option is used in lit tests to prevent deadcoding of patterns inspected.
extern cl::opt<bool>
EnableDCEInRA;

extern cl::opt<bool> EnableSetWavePriority;

extern cl::opt<bool> EnableScalarIRPasses;

extern cl::opt<bool> EnableStructurizerWorkarounds;

extern cl::opt<bool, true> EnableLowerModuleLDS;

extern cl::opt<bool> EnablePreRAOptimizations;

extern cl::opt<bool> EnablePromoteKernelArguments;

extern cl::opt<bool> EnableImageIntrinsicOptimizer;

extern cl::opt<bool>
    EnableLoopPrefetch;

extern cl::opt<bool> EnableMaxIlpSchedStrategy;

extern cl::opt<bool> EnableRewritePartialRegUses;

extern cl::opt<bool> EnableHipStdPar;

namespace llvm {


}