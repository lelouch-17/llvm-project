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

#include "llvm/Transforms/Scalar/FlattenCFG.h"
#include "llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h"
#include "llvm/Transforms/Scalar/Sink.h"
#include "llvm/Transforms/Scalar/StraightLineStrengthReduce.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/MC/MCStreamer.h"

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
#include <optional>

using namespace llvm;
using namespace llvm::PatternMatch;

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

template<typename DerivedT>
class AMDGPUCodeGenPassBuilder
    : public CodeGenPassBuilder<DerivedT> {
public:
  explicit AMDGPUCodeGenPassBuilder(LLVMTargetMachine &TM,
                                    CGPassBuilderOption Opts,
                                    PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder<DerivedT>(TM, Opts, PIC) {}

    bool isPassEnabled(const cl::opt<bool> &Opt,
                     CodeGenOptLevel Level = CodeGenOptLevel::Default) const {
    if (Opt.getNumOccurrences())
      return Opt;
    if (CodeGenPassBuilder<DerivedT>::TM.getOptLevel() < Level)
      return false;
    return Opt;
  }
  
  void addIRPasses(typename CodeGenPassBuilder<DerivedT>::AddIRPass &) const;
  void addCodeGenPrepare(typename CodeGenPassBuilder<DerivedT>::AddIRPass &) const;
  void addPreISel(typename CodeGenPassBuilder<DerivedT>::AddIRPass &) const;
  Error addInstSelector(typename CodeGenPassBuilder<DerivedT>::AddMachinePass &) const; 
  void addEarlyCSEOrGVNPass(typename CodeGenPassBuilder<DerivedT>::AddIRPass &) const;
  void addStraightLineScalarOptimizationPasses(typename CodeGenPassBuilder<DerivedT>::AddIRPass &) const;
};

class GCNCodeGenPassBuilder final : public AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder> {
public:
  GCNCodeGenPassBuilder(LLVMTargetMachine &TM,
                                    CGPassBuilderOption Opts,
                                    PassInstrumentationCallbacks *PIC)
      : AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder>(TM, Opts, PIC){
    //  // It is necessary to know the register usage of the entire call graph.  We
    // // allow calls without EnableAMDGPUFunctionCalls if they are marked
    // // noinline, so this is always required.
    // setRequiresCodeGenSCCOrder(true);
    // substitutePass(&PostRASchedulerID, &PostMachineSchedulerID);
      }
  void addPreISel(AddIRPass &) const;
  void addMachineSSAOptimization(AddMachinePass &) const;
  void addILPOpts(AddMachinePass &) const;
//  Error addInstSelector(AddMachinePass &) const; //TODO
  Error addIRTranslator(AddMachinePass &) const;
 // void addPreLegalizeMachineIR(AddMachinePass &) const;  //TODO
  Error addLegalizeMachineIR(AddMachinePass &) const;
  Error addPreRegBankSelect(AddMachinePass &) const;
  Error addRegBankSelect(AddMachinePass &) const;
  Error addPreGlobalInstructionSelect(AddMachinePass &) const;
  Error addGlobalInstructionSelect(AddMachinePass &) const;
  Error addFastRegAlloc(AddMachinePass &) const;
  void addOptimizedRegAlloc(AddMachinePass &) const;

  
  // FunctionPass *createSGPRAllocPass(bool Optimized);
  // FunctionPass *createVGPRAllocPass(bool Optimized);
  // FunctionPass *createRegAllocPass(bool Optimized) override;

  // bool addRegAssignAndRewriteFast() override; //
  // bool addRegAssignAndRewriteOptimized() override; //


  void addPreRegAlloc(AddMachinePass &) const;
  void addPreRewrite(AddMachinePass &) const;
  void addPostRegAlloc(AddMachinePass &) const;
  void addPreSched2(AddMachinePass &) const;
  void addPreEmitPass(AddMachinePass &) const;
};
} // namespace

void GCNCodeGenPassBuilder::addMachineSSAOptimization(
    AddMachinePass &addPass) const {
  CodeGenPassBuilder<GCNCodeGenPassBuilder>::addMachineSSAOptimization(addPass);
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

  CodeGenPassBuilder<GCNCodeGenPassBuilder>::addILPOpts(addPass);
}
Error GCNCodeGenPassBuilder::addIRTranslator(AddMachinePass &addPass) const {
  addPass(IRTranslatorPass());
  return Error::success();
}
Error GCNCodeGenPassBuilder::addLegalizeMachineIR(
    AddMachinePass &addPass) const {
  addPass(LegalizerPass());
  return Error::success();
}
Error GCNCodeGenPassBuilder::addPreRegBankSelect(AddMachinePass &addPass) const {
  bool IsOptNone = getOptLevel() == CodeGenOptLevel::None;
  addPass(AMDGPUPostLegalizeCombinerPass()); // Add Param Isoptnone. 
}
Error GCNCodeGenPassBuilder::addRegBankSelect(AddMachinePass &addPass) const {
  addPass(RegBankSelectPass());
  return Error::success();
}
Error GCNCodeGenPassBuilder::addPreGlobalInstructionSelect(AddMachinePass &addPass) const {
  bool IsOptNone = getOptLevel() == CodeGenOptLevel::None;
  addPass(AMDGPURegBankCombinerPass());
}
Error GCNCodeGenPassBuilder::addGlobalInstructionSelect(
    AddMachinePass &addPass) const {
  addPass(InstructionSelectPass()); // Param
  return Error::success();
}
Error GCNCodeGenPassBuilder::addFastRegAlloc(AddMachinePass &addPass) const {
  // FIXME: We have to disable the verifier here because of PHIElimination +
  // TwoAddressInstructions disabling it.

  // This must be run immediately after phi elimination and before
  // TwoAddressInstructions, otherwise the processing of the tied operand of
  // SI_ELSE will introduce a copy of the tied operand source after the else.

  // FIXME: insertPass api is broken
  // insertPass(&PHIEliminationID, &SILowerControlFlowID);

  // insertPass(&TwoAddressInstructionPassID, &SIWholeQuadModeID);

  return CodeGenPassBuilder<GCNCodeGenPassBuilder>::addFastRegAlloc(addPass);
}
void GCNCodeGenPassBuilder::addOptimizedRegAlloc(AddMachinePass &addPass) const{
  // Allow the scheduler to run before SIWholeQuadMode inserts exec manipulation
  // instructions that cause scheduling barriers.
  // insertPass(&MachineSchedulerID, &SIWholeQuadModeID);

  // FIXME:: insertPass API is not working unable to find the root cause of the const error 
  // I get after below line is uncommented.
  // addPass.insertPass(&MachineSchedulerPass::Key, SIFixSGPRCopiesPass());

  // if (OptExecMaskPreRA)
  //   insertPass(&MachineSchedulerID, &SIOptimizeExecMaskingPreRAID);

  // if (EnableRewritePartialRegUses)
  //   insertPass(&RenameIndependentSubregsID, &GCNRewritePartialRegUsesID);

  // if (isPassEnabled(EnablePreRAOptimizations))
  //   insertPass(&RenameIndependentSubregsID, &GCNPreRAOptimizationsID);

  // // This is not an essential optimization and it has a noticeable impact on
  // // compilation time, so we only enable it from O2.
  // if (TM->getOptLevel() > CodeGenOptLevel::Less)
  //   insertPass(&MachineSchedulerID, &SIFormMemoryClausesID);

  // // FIXME: when an instruction has a Killed operand, and the instruction is
  // // inside a bundle, seems only the BUNDLE instruction appears as the Kills of
  // // the register in LiveVariables, this would trigger a failure in verifier,
  // // we should fix it and enable the verifier.
  // if (OptVGPRLiveRange)
  //   insertPass(&LiveVariablesID, &SIOptimizeVGPRLiveRangeID);
  // // This must be run immediately after phi elimination and before
  // // TwoAddressInstructions, otherwise the processing of the tied operand of
  // // SI_ELSE will introduce a copy of the tied operand source after the else.
  // insertPass(&PHIEliminationID, &SILowerControlFlowID);

  // if (EnableDCEInRA)
  //   insertPass(&DetectDeadLanesID, &DeadMachineInstructionElimID);

  CodeGenPassBuilder<GCNCodeGenPassBuilder>::addOptimizedRegAlloc(addPass);
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
  CodeGenPassBuilder<GCNCodeGenPassBuilder>::addPostRegAlloc(addPass);
}
void GCNCodeGenPassBuilder::addPreSched2(AddMachinePass &addPass) const {
  if (TM.getOptLevel() > CodeGenOptLevel::None)
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
  AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder>::addPreISel(addPass);

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(AMDGPULateCodeGenPreparePass());

  if (TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(SinkingPass());

  // Merge divergent exit nodes. StructurizeCFG won't recognize the multi-exit
  // regions formed by them.
  addPass(AMDGPUUnifyDivergentExitNodesPass());
  if (!LateCFGStructurize) {
    if (EnableStructurizerWorkarounds) {
      addPass(FixIrreduciblePass());
      addPass(UnifyLoopExitsPass());
    }
    addPass(StructurizeCFGPass()); // true/ -> SkipUniformRegions
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

  // if (TM.getOptLevel() > CodeGenOptLevel::Less)
  //   addPass(AMDGPUPerfHintAnalysis()); 

}
// FIXME: port this function and see the split between AMDGPUPassconfig::addInstSelector
// and GCNPassConfig::addInstSelector
template <typename DerivedT>
Error AMDGPUCodeGenPassBuilder<DerivedT>::addInstSelector(typename CodeGenPassBuilder<DerivedT>::AddMachinePass &addPass) const {
  // Install an instruction selector.

  return Error::success();
}

template <typename DerivedT>
void AMDGPUCodeGenPassBuilder<DerivedT>::addPreISel(typename CodeGenPassBuilder<DerivedT>::AddIRPass &addPass) const {
  if (AMDGPUCodeGenPassBuilder<DerivedT>::TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(FlattenCFGPass());
}

template <typename DerivedT>
void AMDGPUCodeGenPassBuilder<DerivedT>::addStraightLineScalarOptimizationPasses(typename CodeGenPassBuilder<DerivedT>::AddIRPass &addPass) const {
  addPass(SeparateConstOffsetFromGEPPass());
  // ReassociateGEPs exposes more opportunities for SLSR. See
  // the example in reassociate-geps-and-slsr.ll.
  addPass(StraightLineStrengthReducePass());
  // SeparateConstOffsetFromGEP and SLSR creates common expressions which GVN or
  // EarlyCSE can reuse.
  // FIXME: Port this
  AMDGPUCodeGenPassBuilder::addEarlyCSEOrGVNPass(addPass);
  // Run NaryReassociate after EarlyCSE/GVN to be more effective.
  addPass(NaryReassociatePass());
  // NaryReassociate on GEPs creates redundant common expressions, so run
  // EarlyCSE after it.
  addPass(EarlyCSEPass());
}

template <typename DerivedT>
void AMDGPUCodeGenPassBuilder<DerivedT>::addEarlyCSEOrGVNPass(typename CodeGenPassBuilder<DerivedT>::AddIRPass &addPass) const {
   if(1)
    addPass(GVNPass());
  else
    addPass(EarlyCSEPass());
}
template <typename DerivedT>
void AMDGPUCodeGenPassBuilder<DerivedT>::addCodeGenPrepare(typename CodeGenPassBuilder<DerivedT>::AddIRPass &addPass) const {
  if (AMDGPUCodeGenPassBuilder<DerivedT>::TM.getTargetTriple().getArch() == Triple::amdgcn) {
    // FIXME: This pass adds 2 hacky attributes that can be replaced with an
    // analysis, and should be removed.
    addPass(AMDGPUAnnotateKernelFeaturesPass());
  }

  if (AMDGPUCodeGenPassBuilder<DerivedT>::TM.getTargetTriple().getArch() == Triple::amdgcn &&
      EnableLowerKernelArguments)
    addPass(AMDGPULowerKernelArgumentsPass(AMDGPUCodeGenPassBuilder<DerivedT>::TM));

  CodeGenPassBuilder<DerivedT>::addCodeGenPrepare(addPass);

  if (isPassEnabled(EnableLoadStoreVectorizer))
    addPass(LoadStoreVectorizerPass());

  // LowerSwitch pass may introduce unreachable blocks that can
  // cause unexpected behavior for subsequent passes. Placing it
  // here seems better that these blocks would get cleaned up by
  // UnreachableBlockElim inserted next in the pass flow.
  addPass(LowerSwitchPass());
}
template <typename DerivedT>
void AMDGPUCodeGenPassBuilder<DerivedT>::addIRPasses(typename CodeGenPassBuilder<DerivedT>::AddIRPass &addPass) const {

  Triple::ArchType Arch = CodeGenPassBuilder<DerivedT>::TM.getTargetTriple().getArch();
  if (RemoveIncompatibleFunctions && Arch == Triple::amdgcn)
    addPass(AMDGPURemoveIncompatibleFunctionsPass());

  // FIXME: Find a way to disable these passes
  // There is no reason to run these.
  // disablePass(&StackMapLivenessID);
  // disablePass(&FuncletLayoutID);
  // disablePass(&PatchableFunctionID);

  addPass(AMDGPUPrintfRuntimeBindingPass());

  if (LowerCtorDtor)
    addPass(AMDGPUCtorDtorLoweringLegacyPass());

  if (isPassEnabled(EnableImageIntrinsicOptimizer))
    addPass(AMDGPUImageIntrinsicOptimizerPass(CodeGenPassBuilder<DerivedT>::TM));

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
    addPass(AMDGPULowerModuleLDSLegacyPass(CodeGenPassBuilder<DerivedT>::TM));
  }

  // AMDGPUAttributor infers lack of llvm.amdgcn.lds.kernel.id calls, so run
  // after their introduction
  if (CodeGenPassBuilder<DerivedT>::TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(AMDGPUAttributorPass(CodeGenPassBuilder<DerivedT>::TM));

  if (CodeGenPassBuilder<DerivedT>::TM.getOptLevel() > CodeGenOptLevel::None)
    addPass(InferAddressSpacesPass());

  // Run atomic optimizer before Atomic Expand
  // if ((TM.getTargetTriple().getArch() == Triple::amdgcn) &&
  //     (TM.getOptLevel() >= CodeGenOptLevel::Less) &&
  //     (AMDGPUAtomicOptimizerStrategy != ScanOptions::None)) {
  //   addPass(AMDGPUAtomicOptimizerPass(TM, AMDGPUAtomicOptimizerStrategy));
  // }

  addPass(AtomicExpandPass());

  if (CodeGenPassBuilder<DerivedT>::TM.getOptLevel() > CodeGenOptLevel::None) {
    addPass(AMDGPUPromoteAllocaPass(CodeGenPassBuilder<DerivedT>::TM));

    if (isPassEnabled(EnableScalarIRPasses))
      addStraightLineScalarOptimizationPasses(addPass);

    if (EnableAMDGPUAliasAnalysis) {
      addPass(AMDGPUAAWrapperPass());
      // FIXME: Port these
      // addPass(createExternalAAWrapperPass([](Pass &P, Function &,
      //                                        AAResults &AAR) {
      //   if (auto *WrapperPass = P.getAnalysisIfAvailable<AMDGPUAAWrapperPass>())
      //     AAR.addAAResult(WrapperPass->getResult());
      //   }));
    }

    if (CodeGenPassBuilder<DerivedT>::TM.getTargetTriple().getArch() == Triple::amdgcn) {
      // TODO: May want to move later or split into an early and late one.
      addPass(AMDGPUCodeGenPreparePass(CodeGenPassBuilder<DerivedT>::TM));
    }

    // Try to hoist loop invariant parts of divisions AMDGPUCodeGenPrepare may
    // have expanded.
    if (CodeGenPassBuilder<DerivedT>::TM.getOptLevel() > CodeGenOptLevel::Less){
      // FIXME
      // addPass(createLICMPass());
    }
  }


  CodeGenPassBuilder<DerivedT>::addIRPasses(addPass);

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