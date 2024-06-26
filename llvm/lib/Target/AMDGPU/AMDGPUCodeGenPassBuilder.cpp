//===- lib/Target/AMDGPU/AMDGPUCodeGenPassBuilder.cpp ---------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//<<<<<<< HEAD
/// \file
/// This file contains AMDGPU CodeGen pipeline builder.
/// TODO: Port CodeGen passes to new pass manager.
#pragma once

#include "AMDGPUTargetMachine.h"
 #include "AMDGPUCodeGenPassBuilder.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/MC/MCStreamer.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/Sink.h"
#include "AMDGPU.h"
#include "AMDGPUUnifyDivergentExitNodes.h"
#include "R600.h"
#include "llvm/CodeGen/Passes.h"
#include "llvm/CodeGen/RegAllocRegistry.h"
#include "llvm/CodeGen/TargetPassConfig.h"
#include "llvm/IR/PassManager.h"
#include "llvm/Transforms/Scalar.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
#include "llvm/Transforms/Utils/FixIrreducible.h"
#include "llvm/Transforms/Utils.h"
#include "llvm/Transforms/Utils/UnifyLoopExits.h"
//#include "llvm/CodeGen/AtomicExpand.h"
#include <optional>

using namespace llvm;

extern cl::opt<bool> EnableEarlyIfConversion;

extern cl::opt<bool> OptExecMaskPreRA;



// Option to control global loads scalarization
extern cl::opt<bool> ScalarizeGlobal;

// Option to run internalize pass.
extern cl::opt<bool> InternalizeSymbols;

// Option to inline all early.
extern cl::opt<bool> EarlyInlineAll;


extern cl::opt<bool> EnableSDWAPeephole;

extern cl::opt<bool> EnableDPPCombine;


// Option to run late CFG structurizer
extern cl::opt<bool, true> LateCFGStructurize;

// Disable structurizer-based control-flow lowering in order to test convergence
// control tokens. This should eventually be replaced by the wave-transform.
extern cl::opt<bool, true> DisableStructurizer;

// Enable lib calls simplifications
extern cl::opt<bool> EnableLibCallSimplify;


extern cl::opt<bool> EnableRegReassign;

extern cl::opt<bool> OptVGPRLiveRange;


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


extern cl::opt<bool> EnableStructurizerWorkarounds;


extern cl::opt<bool> EnablePreRAOptimizations;

extern cl::opt<bool> EnablePromoteKernelArguments;


extern cl::opt<bool> EnableLoopPrefetch;

extern cl::opt<bool> EnableMaxIlpSchedStrategy;

extern cl::opt<bool> EnableRewritePartialRegUses;

void GCNCodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                          CreateMCStreamer MCStreamer) const {
  // TODO: Add AsmPrinter.
  
}

FunctionPass *
GCNCodeGenPassBuilder::createSGPRAllocPass(bool Optimized) { 
  return nullptr;
}

FunctionPass *
GCNCodeGenPassBuilder::createVGPRAllocPass(bool Optimized) { 
  return nullptr;
}


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
   addPass(VirtRegRewriterPass()); 

  // Equivalent of PEI for SGPRs.
  addPass(SILowerSGPRSpillsPass());
  addPass(SIPreAllocateWWMRegsPass());

  // addPass(createVGPRAllocPass(true));  FIXME_NEW

  addPreRewrite(addPass);
   addPass(VirtRegRewriterPass());

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
  Error Error =
      AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder,
                               GCNTargetMachine>::addInstSelector(addPass);
  addPass(SIFixSGPRCopiesPass());
  addPass(SILowerI1CopiesPass());
  return Error;
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
    addPass(StructurizeCFGPass());  
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
Error GCNTargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType,const CGPassBuilderOption &Opt,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = GCNCodeGenPassBuilder(*this, Opt, PIC);
  return CGPB.buildPipeline(MPM, Out, DwoOut, FileType);
}
