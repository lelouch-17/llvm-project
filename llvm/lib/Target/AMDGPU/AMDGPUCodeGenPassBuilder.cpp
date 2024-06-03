//===-- AMDGPUCodeGenPassBuilder.cpp ------------------------------*- C++ -*-=//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
/// \file
/// This file contains AMDGPU CodeGen pipeline builder.
/// TODO: Port CodeGen passes to new pass manager.
//===----------------------------------------------------------------------===//
#pragma once

#include "AMDGPUTargetMachine.h"

 #include "AMDGPUCodeGenPassBuilder.h"

#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/MC/MCStreamer.h"

#include "llvm/MC/MCStreamer.h"
#include "llvm/Transforms/Scalar/FlattenCFG.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h"
#include "llvm/Transforms/Scalar/Sink.h"
#include "llvm/Transforms/Scalar/StraightLineStrengthReduce.h"

#include "AMDGPU.h"
#include "AMDGPUAliasAnalysis.h"
#include "AMDGPUCtorDtorLowering.h"
#include "AMDGPUExportClustering.h"
#include "AMDGPUIGroupLP.h"
#include "AMDGPUMacroFusion.h"
#include "AMDGPURegBankSelect.h"
#include "AMDGPUTargetObjectFile.h"
#include "AMDGPUTargetTransformInfo.h"
#include "AMDGPUUnifyDivergentExitNodes.h"
#include "GCNIterativeScheduler.h"
#include "GCNSchedStrategy.h"
#include "GCNVOPDUtils.h"
#include "R600.h"
#include "R600MachineFunctionInfo.h"
#include "R600TargetMachine.h"
#include "SIMachineFunctionInfo.h"
#include "SIMachineScheduler.h"
#include "TargetInfo/AMDGPUTargetInfo.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/Analysis/CGSCCPassManager.h"
#include "llvm/CodeGen/GlobalISel/CSEInfo.h"
#include "llvm/CodeGen/GlobalISel/IRTranslator.h"
#include "llvm/CodeGen/GlobalISel/InstructionSelect.h"
#include "llvm/CodeGen/GlobalISel/Legalizer.h"
#include "llvm/CodeGen/GlobalISel/Localizer.h"
#include "llvm/CodeGen/GlobalISel/RegBankSelect.h"
#include "llvm/CodeGen/MIRParser/MIParser.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/IntrinsicsAMDGPU.h"
#include "llvm/IR/PassManager.h"
#include "llvm/IR/PatternMatch.h"
#include "llvm/InitializePasses.h"
#include "llvm/MC/TargetRegistry.h"
#include "llvm/Passes/PassBuilder.h"
#include "llvm/Transforms/HipStdPar/HipStdPar.h"
#include "llvm/Transforms/IPO.h"
#include "llvm/Transforms/IPO/AlwaysInliner.h"
#include "llvm/Transforms/IPO/GlobalDCE.h"
#include "llvm/Transforms/IPO/Internalize.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/Transforms/Scalar/LoopPassManager.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/Transforms/Utils/FixIrreducible.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/SimplifyLibCalls.h"
#include "llvm/Transforms/Utils/UnifyLoopExits.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"
#include "llvm/CodeGen/AtomicExpand.h"
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

extern cl::opt<bool> EnableEarlyIfConversion;

extern cl::opt<bool> OptExecMaskPreRA;

extern cl::opt<bool> LowerCtorDtor;

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
extern cl::opt<bool> EnableInsertSingleUseVDST;

// Enable GFX11+ s_delay_alu insertion
extern cl::opt<bool> EnableInsertDelayAlu;

// Enable GFX11+ VOPD
extern cl::opt<bool> EnableVOPD;

// Option is used in lit tests to prevent deadcoding of patterns inspected.
extern cl::opt<bool> EnableDCEInRA;

extern cl::opt<bool> EnableSetWavePriority;

extern cl::opt<bool> EnableScalarIRPasses;

extern cl::opt<bool> EnableStructurizerWorkarounds;

extern cl::opt<bool, true> EnableLowerModuleLDS;

extern cl::opt<bool> EnablePreRAOptimizations;

extern cl::opt<bool> EnablePromoteKernelArguments;

extern cl::opt<bool> EnableImageIntrinsicOptimizer;

extern cl::opt<bool> EnableLoopPrefetch;

extern cl::opt<bool> EnableMaxIlpSchedStrategy;

extern cl::opt<bool> EnableRewritePartialRegUses;

extern cl::opt<bool> EnableHipStdPar;

namespace {

#define DUMMY_MODULE_PASS(NAME, PASS_NAME, CONSTRUCTOR)                        \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Module &, ModuleAnalysisManager &) {                 \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_FUNCTION_PASS(NAME, PASS_NAME, CONSTRUCTOR)                      \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Function &, FunctionAnalysisManager &) {             \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_MACHINE_FUNCTION_PASS(NAME, PASS_NAME, CONSTRUCTOR)              \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(MachineFunction &,                                   \
                          MachineFunctionAnalysisManager &) {                  \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
    static AnalysisKey Key;                                                    \
  };                                                                           \
  AnalysisKey PASS_NAME::Key;
#include "AMDGPUPassRegistry.def"

class GCNCodeGenPassBuilder final
    : public AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder, GCNTargetMachine> {
public:
  GCNCodeGenPassBuilder(GCNTargetMachine &TM, CGPassBuilderOption Opts,
                        PassInstrumentationCallbacks *PIC)
      : AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder, GCNTargetMachine>(
            TM, Opts, PIC) {
    //  // It is necessary to know the register usage of the entire call graph.  We
    // // allow calls without EnableAMDGPUFunctionCalls if they are marked
    // // noinline, so this is always required.
    // FIXME: 
    // setRequiresCodeGenSCCOrder(true);
    // Target could set CGPassBuilderOption::MISchedPostRA to true to achieve
    //     substitutePass(&PostRASchedulerID, &PostMachineSchedulerID)
  }

  // FIXME: Port MachineScheduler Pass
  // ScheduleDAGInstrs *
  // createMachineScheduler(MachineSchedContext *C) const;

  // ScheduleDAGInstrs *
  // createPostMachineScheduler(MachineSchedContext *C) const;

  void addPreISel(AddIRPass &) const;
  void addMachineSSAOptimization(AddMachinePass &) const;
  void addILPOpts(AddMachinePass &) const;
  Error addInstSelector(AddMachinePass &) const;
  Error addIRTranslator(AddMachinePass &) const;
  void addPreLegalizeMachineIR(AddMachinePass &) const;
  Error addLegalizeMachineIR(AddMachinePass &) const;
  Error addPreRegBankSelect(AddMachinePass &) const;
  Error addRegBankSelect(AddMachinePass &) const;
  Error addPreGlobalInstructionSelect(AddMachinePass &) const;
  Error addGlobalInstructionSelect(AddMachinePass &) const;
  Error addFastRegAlloc(AddMachinePass &) const;
  void addOptimizedRegAlloc(AddMachinePass &) const;
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const;

  FunctionPass *createSGPRAllocPass(bool Optimized);
  FunctionPass *createVGPRAllocPass(bool Optimized);
  // FunctionPass *createRegAllocPass(bool Optimized) override;

  Error addRegAssignmentFast(AddMachinePass &) const;
  Error addRegAssignmentOptimized(AddMachinePass &) const;

  void addPreRegAlloc(AddMachinePass &) const;
  void addPreRewrite(AddMachinePass &) const;
  void addPostRegAlloc(AddMachinePass &) const;
  void addPreSched2(AddMachinePass &) const;
  void addPreEmitPass(AddMachinePass &) const;
};
} // namespace

void GCNCodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                          CreateMCStreamer MCStreamer) const {
  // TODO: Add AsmPrinter.
  
}

/// A dummy default pass factory indicates whether the register allocator is
/// overridden on the command line.
extern llvm::once_flag InitializeDefaultSGPRRegisterAllocatorFlag;
extern llvm::once_flag InitializeDefaultVGPRRegisterAllocatorFlag;

class SGPRRegisterRegAlloc : public RegisterRegAllocBase<SGPRRegisterRegAlloc> {
public:
  SGPRRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
      : RegisterRegAllocBase(N, D, C) {}
};

class VGPRRegisterRegAlloc : public RegisterRegAllocBase<VGPRRegisterRegAlloc> {
public:
  VGPRRegisterRegAlloc(const char *N, const char *D, FunctionPassCtor C)
      : RegisterRegAllocBase(N, D, C) {}
};

/// -{sgpr|vgpr}-regalloc=... command line option.
static FunctionPass *useDefaultRegisterAllocator() { return nullptr; }

extern cl::opt<SGPRRegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<SGPRRegisterRegAlloc>>
    SGPRRegAlloc;

extern cl::opt<VGPRRegisterRegAlloc::FunctionPassCtor, false,
               RegisterPassParser<VGPRRegisterRegAlloc>>
    VGPRRegAlloc;

static void initializeDefaultSGPRRegisterAllocatorOnce() {
  RegisterRegAlloc::FunctionPassCtor Ctor = SGPRRegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = SGPRRegAlloc;
    SGPRRegisterRegAlloc::setDefault(SGPRRegAlloc);
  }
}

static void initializeDefaultVGPRRegisterAllocatorOnce() {
  RegisterRegAlloc::FunctionPassCtor Ctor = VGPRRegisterRegAlloc::getDefault();

  if (!Ctor) {
    Ctor = VGPRRegAlloc;
    VGPRRegisterRegAlloc::setDefault(VGPRRegAlloc);
  }
}

FunctionPass *
GCNCodeGenPassBuilder::createSGPRAllocPass(bool Optimized) { // Newpm
  // Initialize the global default.
  llvm::call_once(InitializeDefaultSGPRRegisterAllocatorFlag,
                  initializeDefaultSGPRRegisterAllocatorOnce);

  RegisterRegAlloc::FunctionPassCtor Ctor = SGPRRegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  // FIXME_NEW : Port These Passes
  // if (Optimized)
  //   return createGreedyRegisterAllocator(onlyAllocateSGPRs);

  // return createFastRegisterAllocator(onlyAllocateSGPRs, false);
  return nullptr;
}

FunctionPass *
GCNCodeGenPassBuilder::createVGPRAllocPass(bool Optimized) { // Newpm
  // Initialize the global default.
  llvm::call_once(InitializeDefaultVGPRRegisterAllocatorFlag,
                  initializeDefaultVGPRRegisterAllocatorOnce);

  RegisterRegAlloc::FunctionPassCtor Ctor = VGPRRegisterRegAlloc::getDefault();
  if (Ctor != useDefaultRegisterAllocator)
    return Ctor();

  // FIXME_NEW : Port These Passes
  // if (Optimized)
  //   return createGreedyVGPRRegisterAllocator();

  // return createFastVGPRRegisterAllocator();
  return nullptr;
}

static const char RegAllocOptNotSupportedMessage[] =
    "-regalloc not supported with amdgcn. Use -sgpr-regalloc and "
    "-vgpr-regalloc";

Error GCNCodeGenPassBuilder::addRegAssignmentFast(
    AddMachinePass &addPass) const {
  // FIXME_NEW : RegAlloc CLI in TargetPassConfig
  // if (!usingDefaultRegAlloc())
  //   report_fatal_error(RegAllocOptNotSupportedMessage);

  addPass(GCNPreRALongBranchRegPass());

  //  addPass(createSGPRAllocPass(false)); FIXME_NEW

  // Equivalent of PEI for SGPRs.
  addPass(SILowerSGPRSpillsPass());
  addPass(SIPreAllocateWWMRegsPass());

  // addPass(createVGPRAllocPass(false)); FIXME_NEW

  addPass(SILowerWWMCopiesPass());
  return Error::success();
}

Error GCNCodeGenPassBuilder::addRegAssignmentOptimized(
    AddMachinePass &addPass) const {
  // FIXME_NEW : RegAlloc CLI in TargetPassConfig Remove implementation ? 
  // if (!usingDefaultRegAlloc())
  //   report_fatal_error(RegAllocOptNotSupportedMessage);

  addPass(GCNPreRALongBranchRegPass());

  //  addPass(createSGPRAllocPass(true)); FIXME_NEW

  // Commit allocated register changes. This is mostly necessary because too
  // many things rely on the use lists of the physical registers, such as the
  // verifier. This is only necessary with allocators which use LiveIntervals,
  // since FastRegAlloc does the replacements itself.
  // addPass(createVirtRegRewriter(false)); FIXME: Port to NPM

  // Equivalent of PEI for SGPRs.
  addPass(SILowerSGPRSpillsPass());
  addPass(SIPreAllocateWWMRegsPass());

  // addPass(createVGPRAllocPass(true));  FIXME_NEW

  addPreRewrite(addPass);
  // addPass(&VirtRegRewriterID); FIXME_NEW

  return Error::success();
}
void GCNCodeGenPassBuilder::addMachineSSAOptimization(
    AddMachinePass &addPass) const {
  CodeGenPassBuilder<GCNCodeGenPassBuilder,
                     GCNTargetMachine>::addMachineSSAOptimization(addPass);
  // We want to fold operands after PeepholeOptimizer has run (or as part of
  // it), because it will eliminate extra copies making it easier to fold the
  // real source operand. We want to eliminate dead instructions after, so that
  // we see fewer uses of the copies. We then need to clean up the dead
  // instructions leftover after the operands are folded as well.
  //
  // XXX - Can we get away without running DeadMachineInstructionElim again?
  addPass(SIFoldOperandsPass());
  if (EnableDPPCombine)
    addPass(GCNDPPCombinePass());
  addPass(SILoadStoreOptimizerPass());
  if (isPassEnabled(EnableSDWAPeephole)) {
    addPass(SIPeepholeSDWAPass());
    addPass(EarlyMachineLICMPass());
    addPass(MachineCSEPass());
    addPass(SIFoldOperandsPass());
  }
  addPass(DeadMachineInstructionElimPass());
  addPass(SIShrinkInstructionsPass());
}
void GCNCodeGenPassBuilder::addILPOpts(AddMachinePass &addPass) const {
  if (EnableEarlyIfConversion)
    addPass(EarlyIfConverterPass());

  CodeGenPassBuilder<GCNCodeGenPassBuilder, GCNTargetMachine>::addILPOpts(
      addPass);
}
Error GCNCodeGenPassBuilder::addIRTranslator(AddMachinePass &addPass) const {
  addPass(IRTranslatorPass());
  return Error::success();
}
void GCNCodeGenPassBuilder::addPreLegalizeMachineIR(AddMachinePass &addPass) const {
  bool IsOptNone = getOptLevel() == CodeGenOptLevel::None;
  addPass(AMDGPUPreLegalizerCombinerPass(IsOptNone));
  addPass(LocalizerPass()); 
  return ;
}
Error GCNCodeGenPassBuilder::addLegalizeMachineIR(
    AddMachinePass &addPass) const {
  addPass(LegalizerPass());
  return Error::success();
}
Error GCNCodeGenPassBuilder::addInstSelector(
    AddMachinePass &addPass) const {
  Error E =
      AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder,
                               GCNTargetMachine>::addInstSelector(addPass);
  addPass(SIFixSGPRCopiesPass());
  addPass(SILowerI1CopiesPass());
  return E;
}
Error GCNCodeGenPassBuilder::addPreRegBankSelect(AddMachinePass &addPass) const {
  bool IsOptNone = getOptLevel() == CodeGenOptLevel::None;
  addPass(AMDGPUPostLegalizeCombinerPass(IsOptNone)); 
   return Error::success();
}
Error GCNCodeGenPassBuilder::addRegBankSelect(AddMachinePass &addPass) const {
  addPass(RegBankSelectPass());
  return Error::success();
}
Error GCNCodeGenPassBuilder::addPreGlobalInstructionSelect(AddMachinePass &addPass) const {
  bool IsOptNone = getOptLevel() == CodeGenOptLevel::None;
  addPass(AMDGPURegBankCombinerPass(IsOptNone));
  return Error::success();
}
Error GCNCodeGenPassBuilder::addGlobalInstructionSelect(
    AddMachinePass &addPass) const {
  addPass(InstructionSelectPass(getOptLevel())); // Param
  return Error::success();
}
Error GCNCodeGenPassBuilder::addFastRegAlloc(AddMachinePass &addPass) const {
  // FIXME: We have to disable the verifier here because of PHIElimination +
  // TwoAddressInstructions disabling it.

  // This must be run immediately after phi elimination and before
  // TwoAddressInstructions, otherwise the processing of the tied operand of
  // SI_ELSE will introduce a copy of the tied operand source after the else.

  const_cast<GCNCodeGenPassBuilder*>(this)->insertPass<PHIEliminationPass,SILowerControlFlowPass>(SILowerControlFlowPass());


  const_cast<GCNCodeGenPassBuilder*>(this)->insertPass<TwoAddressInstructionPass,SIWholeQuadModePass>(SIWholeQuadModePass());

  return CodeGenPassBuilder<GCNCodeGenPassBuilder,
                            GCNTargetMachine>::addFastRegAlloc(addPass);
}
void GCNCodeGenPassBuilder::addOptimizedRegAlloc(AddMachinePass &addPass) const{
  // Allow the scheduler to run before SIWholeQuadMode inserts exec manipulation
  // instructions that cause scheduling barriers.
    const_cast<GCNCodeGenPassBuilder*>(this)->insertPass<MachineSchedulerPass,SIWholeQuadModePass>(SIWholeQuadModePass());
  

  if (OptExecMaskPreRA)
    const_cast<GCNCodeGenPassBuilder*>(this)->insertPass<MachineSchedulerPass,SIOptimizeExecMaskingPreRAPass>(SIOptimizeExecMaskingPreRAPass());

   if (EnableRewritePartialRegUses)
     const_cast<GCNCodeGenPassBuilder*>(this)->insertPass<RenameIndependentSubregsPass,GCNRewritePartialRegUsesPass>(GCNRewritePartialRegUsesPass());

   if (isPassEnabled(EnablePreRAOptimizations))
     const_cast<GCNCodeGenPassBuilder*>(this)->insertPass<RenameIndependentSubregsPass,GCNPreRAOptimizationsPass>(GCNPreRAOptimizationsPass());

  // // This is not an essential optimization and it has a noticeable impact on
  // // compilation time, so we only enable it from O2.
   if (getOptLevel() > CodeGenOptLevel::Less)
     const_cast<GCNCodeGenPassBuilder*>(this)->insertPass<MachineSchedulerPass,SIFormMemoryClausesPass>(SIFormMemoryClausesPass());

  // // FIXME: when an instruction has a Killed operand, and the instruction is
  // // inside a bundle, seems only the BUNDLE instruction appears as the Kills of
  // // the register in LiveVariables, this would trigger a failure in verifier,
  // // we should fix it and enable the verifier.
  //FIXME: check MschinePassRegistry
  // if (OptVGPRLiveRange)
  //   insertPass(&LiveVariablesID, &SIOptimizeVGPRLiveRangeID);
  // // This must be run immediately after phi elimination and before
  // // TwoAddressInstructions, otherwise the processing of the tied operand of
  // // SI_ELSE will introduce a copy of the tied operand source after the else.
   const_cast<GCNCodeGenPassBuilder*>(this)->insertPass<PHIEliminationPass,SILowerControlFlowPass>(SILowerControlFlowPass());

  if (EnableDCEInRA)
    const_cast<GCNCodeGenPassBuilder*>(this)->insertPass<DetectDeadLanesPass,DeadMachineInstructionElimPass>(DeadMachineInstructionElimPass());

  CodeGenPassBuilder<GCNCodeGenPassBuilder,
                     GCNTargetMachine>::addOptimizedRegAlloc(addPass);
}
void GCNCodeGenPassBuilder::addPreRegAlloc(AddMachinePass &addPass) const {
  if (LateCFGStructurize) {
    addPass(AMDGPUMachineCFGStructurizerPass());
  }
}
void GCNCodeGenPassBuilder::addPreRewrite(AddMachinePass &addPass) const {
  addPass(SILowerWWMCopiesPass());
  if (EnableRegReassign)
    addPass(GCNNSAReassignPass());
}
void GCNCodeGenPassBuilder::addPostRegAlloc(AddMachinePass &addPass) const {
  addPass(SIFixVGPRCopiesPass());
  if (getOptLevel() > CodeGenOptLevel::None)
    addPass(SIOptimizeExecMaskingPass());
  CodeGenPassBuilder<GCNCodeGenPassBuilder, GCNTargetMachine>::addPostRegAlloc(
      addPass);
}
void GCNCodeGenPassBuilder::addPreSched2(AddMachinePass &addPass) const {
  if (getOptLevel() > CodeGenOptLevel::None)
    addPass(SIShrinkInstructionsPass());
  addPass(SIPostRABundlerPass());
}
void GCNCodeGenPassBuilder::addPreEmitPass(AddMachinePass &addPass) const {
  if (isPassEnabled(EnableVOPD, CodeGenOptLevel::Less))
    addPass(GCNCreateVOPDPass());
  addPass(SIMemoryLegalizerPass());
  addPass(SIInsertWaitcntsPass());

  addPass(SIModeRegisterPass());

  if (getOptLevel() > CodeGenOptLevel::None)
    addPass(SIInsertHardClausesPass());

  addPass(SILateBranchLoweringPass());
  if (isPassEnabled(EnableSetWavePriority, CodeGenOptLevel::Less))
    addPass(AMDGPUSetWavePriorityPass());
  if (getOptLevel() > CodeGenOptLevel::None)
    addPass(SIPreEmitPeepholePass());
  // The hazard recognizer that runs as part of the post-ra scheduler does not
  // guarantee to be able handle all hazards correctly. This is because if there
  // are multiple scheduling regions in a basic block, the regions are scheduled
  // bottom up, so when we begin to schedule a region we don't know what
  // instructions were emitted directly before it.
  //
  // Here we add a stand-alone hazard recognizer pass which can handle all
  // cases.
  addPass(PostRAHazardRecognizerPass());

  if (isPassEnabled(EnableInsertSingleUseVDST, CodeGenOptLevel::Less))
    addPass(AMDGPUInsertSingleUseVDSTPass());

  if (isPassEnabled(EnableInsertDelayAlu, CodeGenOptLevel::Less))
    addPass(AMDGPUInsertDelayAluPass());

  addPass(BranchRelaxationPass());
}
void GCNCodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder, GCNTargetMachine>::addPreISel(
      addPass);

  if (getOptLevel() > CodeGenOptLevel::None)
    addPass(AMDGPULateCodeGenPreparePass());

  if (getOptLevel() > CodeGenOptLevel::None)
    addPass(SinkingPass());

  // Merge divergent exit nodes. StructurizeCFG won't recognize the multi-exit
  // regions formed by them.
  addPass(AMDGPUUnifyDivergentExitNodesPass());
  if (!LateCFGStructurize) {
    if (EnableStructurizerWorkarounds) {
      addPass(FixIrreduciblePass());
      addPass(UnifyLoopExitsPass());
    }
    addPass(StructurizeCFGPass()); // No SkipUniformRegions opn 
  }
  addPass(AMDGPUAnnotateUniformValuesPass());
  if (!LateCFGStructurize) {
    addPass(SIAnnotateControlFlowPass());
    // TODO: Move this right after structurizeCFG to avoid extra divergence
    // analysis. This depends on stopping SIAnnotateControlFlow from making
    // control flow modifications.
    addPass(AMDGPURewriteUndefForPHIPass());
  }
  addPass(LCSSAPass());
   

  // if (getOptLevel() > CodeGenOptLevel::Less)
   // FIXME_NEW : CGSCCPass ? 
  //   addPass(createModuleToPostOrderCGSCCPassAdaptor(AMDGPUPerfHintAnalysisPass())); 

}


template <typename DerivedT, typename TargetMachineT>
Error AMDGPUCodeGenPassBuilder<DerivedT, TargetMachineT>::addInstSelector(
    typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddMachinePass
        &addPass) const {
  addPass(AMDGPUDAGToDAGISelPass(
      getAMDGPUTargetMachine(),
      CodeGenPassBuilder<DerivedT, TargetMachineT>::getOptLevel()));
  return Error::success();
}

template <typename DerivedT, typename TargetMachineT>
void AMDGPUCodeGenPassBuilder<DerivedT, TargetMachineT>::addPreISel(
    typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddIRPass &addPass)
    const {
  if (CodeGenPassBuilder<DerivedT, TargetMachineT>::getOptLevel() >
      CodeGenOptLevel::None)
    addPass(FlattenCFGPass());
}

template <typename DerivedT, typename TargetMachineT>
void AMDGPUCodeGenPassBuilder<DerivedT, TargetMachineT>::
    addStraightLineScalarOptimizationPasses(
        typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddIRPass
            &addPass) const {
  addPass(SeparateConstOffsetFromGEPPass());
  // ReassociateGEPs exposes more opportunities for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addPass(StraightLineStrengthReducePass());
  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN or
  // EarlyCSE can reuse.
  addEarlyCSEOrGVNPass(addPass);
  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addPass(NaryReassociatePass());
  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addPass(EarlyCSEPass());
}

template <typename DerivedT, typename TargetMachineT>
void AMDGPUCodeGenPassBuilder<DerivedT, TargetMachineT>::addEarlyCSEOrGVNPass(
    typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddIRPass &addPass)
    const {
  if (CodeGenPassBuilder<DerivedT, TargetMachineT>::getOptLevel() == CodeGenOptLevel::Aggressive)
    addPass(GVNPass());
  else
    addPass(EarlyCSEPass());
}
template <typename DerivedT, typename TargetMachineT>
void AMDGPUCodeGenPassBuilder<DerivedT, TargetMachineT>::addCodeGenPrepare(
    typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddIRPass &addPass)
    const {
  if (AMDGPUCodeGenPassBuilder<DerivedT, TargetMachineT>::TM.getTargetTriple()
          .getArch() == Triple::amdgcn) {
    // FIXME: This pass adds 2 hacky attributes that can be replaced with an
    // analysis, and should be removed.
    addPass(AMDGPUAnnotateKernelFeaturesPass());
  }

  if (AMDGPUCodeGenPassBuilder<DerivedT, TargetMachineT>::TM.getTargetTriple()
              .getArch() == Triple::amdgcn &&
      EnableLowerKernelArguments)
    addPass(AMDGPULowerKernelArgumentsPass(
        AMDGPUCodeGenPassBuilder<DerivedT, TargetMachineT>::TM));

  CodeGenPassBuilder<DerivedT, TargetMachineT>::addCodeGenPrepare(addPass);

  if (isPassEnabled(EnableLoadStoreVectorizer))
    addPass(LoadStoreVectorizerPass());

  // LowerSwitch pass may introduce unreachable blocks that can
  // cause unexpected behavior for subsequent passes. Placing it
  // here seems better that these blocks would get cleaned up by
  // UnreachableBlockElim inserted next in the pass flow.
  addPass(LowerSwitchPass());
}
template <typename DerivedT, typename TargetMachineT>
void AMDGPUCodeGenPassBuilder<DerivedT, TargetMachineT>::addIRPasses(
    typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddIRPass &addPass)
    const {

  Triple::ArchType Arch =
      CodeGenPassBuilder<DerivedT, TargetMachineT>::TM.getTargetTriple()
          .getArch();
  if (RemoveIncompatibleFunctions && Arch == Triple::amdgcn)
    addPass(AMDGPURemoveIncompatibleFunctionsPass());

  // There is no reason to run these.
  const_cast<AMDGPUCodeGenPassBuilder*>(this)->template disablePass<StackMapLivenessPass,FuncletLayoutPass,PatchableFunctionPass>();

  addPass(AMDGPUPrintfRuntimeBindingPass());

  if (LowerCtorDtor)
    addPass(AMDGPUCtorDtorLoweringLegacyPass());

  if (isPassEnabled(EnableImageIntrinsicOptimizer))
    addPass(AMDGPUImageIntrinsicOptimizerPass(
        CodeGenPassBuilder<DerivedT, TargetMachineT>::TM));

  // Function calls are not supported, so make sure we inline everything.
  addPass(AMDGPUAlwaysInlinePass());
  addPass(AlwaysInlinerLegacyPass());

  // Handle uses of OpenCL image2d_t, image3d_t and sampler_t arguments.
  if (Arch == Triple::r600)
    addPass(R600OpenCLImageTypeLoweringPass());

  // Replace OpenCL enqueued block function pointers with global variables.
  addPass(AMDGPUOpenCLEnqueuedBlockLoweringPass());

  // Runs before PromoteAlloca so the latter can account for function uses
  // Removed Ref - mine
  if (EnableLowerModuleLDS) {
    addPass(AMDGPULowerModuleLDSLegacyPass(
        CodeGenPassBuilder<DerivedT, TargetMachineT>::TM));
  }

  // AMDGPUAttributor infers lack of llvm.amdgcn.lds.kernel.id calls, so run
  // after their introduction
  if (CodeGenPassBuilder<DerivedT, TargetMachineT>::getOptLevel() >
      CodeGenOptLevel::None)
    addPass(
        AMDGPUAttributorPass(CodeGenPassBuilder<DerivedT, TargetMachineT>::TM));

  if (CodeGenPassBuilder<DerivedT, TargetMachineT>::getOptLevel() >
      CodeGenOptLevel::None)
    addPass(InferAddressSpacesPass());

  // Run atomic optimizer before Atomic Expand
  if ((CodeGenPassBuilder<DerivedT, TargetMachineT>::TM.getTargetTriple().getArch() == Triple::amdgcn) &&
      (CodeGenPassBuilder<DerivedT, TargetMachineT>::getOptLevel() >= CodeGenOptLevel::Less) &&
      (AMDGPUAtomicOptimizerStrategy != ScanOptions::None)) {
    addPass(AMDGPUAtomicOptimizerPass(CodeGenPassBuilder<DerivedT, TargetMachineT>::TM, AMDGPUAtomicOptimizerStrategy));
  }

  TargetMachineT* PtrTm = &(CodeGenPassBuilder<DerivedT, TargetMachineT>::TM);  
  addPass(AtomicExpandPass(PtrTm));

  if (CodeGenPassBuilder<DerivedT, TargetMachineT>::getOptLevel() >
      CodeGenOptLevel::None) {
    addPass(AMDGPUPromoteAllocaPass(
        CodeGenPassBuilder<DerivedT, TargetMachineT>::TM));

    if (isPassEnabled(EnableScalarIRPasses))
      addStraightLineScalarOptimizationPasses(addPass);

    if (EnableAMDGPUAliasAnalysis) {
      // FIXME_NEW: 
      //addPass(AMDGPUAA());
      // addPass(createExternalAAWrapperPass([](Pass &P, Function &,
      //                                        AAResults &AAR) {
      //   if (auto *WrapperPass =
      //   P.getAnalysisIfAvailable<AMDGPUAAWrapperPass>())
      //     AAR.addAAResult(WrapperPass->getResult());
      //   }));
    }

    if (CodeGenPassBuilder<DerivedT, TargetMachineT>::TM.getTargetTriple()
            .getArch() == Triple::amdgcn) {
      // TODO: May want to move later or split into an early and late one.
      addPass(AMDGPUCodeGenPreparePass(
          CodeGenPassBuilder<DerivedT, TargetMachineT>::TM));
    }

    // Try to hoist loop invariant parts of divisions AMDGPUCodeGenPrepare may
    // have expanded.
    if (CodeGenPassBuilder<DerivedT, TargetMachineT>::getOptLevel() >
        CodeGenOptLevel::Less) {
       addPass(createFunctionToLoopPassAdaptor(LICMPass()));
    }
  }

  CodeGenPassBuilder<DerivedT, TargetMachineT>::addIRPasses(addPass);

  // EarlyCSE is not always strong enough to clean up what LSR produces. For
  // example, GVN can combine
  //
  //   %0 = add %a, %b
  //   %1 = add %b, %a
  //
  // and
  //
  //   %0 = shl nsw %a, 2
  //   %1 = shl %a, 2
  //
  // but EarlyCSE can do neither of them.
  if (isPassEnabled(EnableScalarIRPasses))
    addEarlyCSEOrGVNPass(addPass);
}
Error GCNTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType,const CGPassBuilderOption &Opt,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = GCNCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, Out, DwoOut, FileType);
}