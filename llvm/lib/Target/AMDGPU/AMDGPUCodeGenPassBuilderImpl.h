#include "AMDGPU.h"
#include "AMDGPUTargetMachine.h"
#include "llvm/Passes/CodeGenPassBuilder.h"
#include "llvm/Transforms/Scalar/EarlyCSE.h"
#include "llvm/Transforms/Scalar/FlattenCFG.h"
#include "llvm/Transforms/Scalar/GVN.h"
#include "llvm/Transforms/Scalar/NaryReassociate.h"
#include "llvm/Transforms/Scalar/SeparateConstOffsetFromGEP.h"
#include "llvm/Transforms/Scalar/StraightLineStrengthReduce.h"
#include "llvm/Transforms/Utils/LowerSwitch.h"
#include "llvm/Transforms/Vectorize/LoadStoreVectorizer.h"
#include "llvm/Transforms/Scalar/InferAddressSpaces.h"
#include "llvm/CodeGen/AtomicExpand.h"

using namespace llvm;


extern cl::opt<bool> EnableLowerKernelArguments;

extern cl::opt<bool> RemoveIncompatibleFunctions;

extern cl::opt<bool> LowerCtorDtor;


extern cl::opt<bool> EnableImageIntrinsicOptimizer;

extern cl::opt<bool, true> EnableLowerModuleLDS;

extern cl::opt<ScanOptions> AMDGPUAtomicOptimizerStrategy;


extern cl::opt<bool> EnableScalarIRPasses;

// Enable address space based alias analysis
extern cl::opt<bool> EnableAMDGPUAliasAnalysis;

// Option to disable vectorizer for tests.
extern cl::opt<bool> EnableLoadStoreVectorizer;

namespace {
#define DUMMY_MODULE_PASS(NAME, PASS_NAME)                        \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Module &, ModuleAnalysisManager &) {                 \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_LOOP_PASS(NAME, PASS_NAME)                          \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Loop &, LoopAnalysisManager &) {                     \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
  #define DUMMY_CGSCC_PASS(NAME, PASS_NAME)                    \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(LazyCallGraph::SCC &,CGSCCAnalysisManager &,LazyCallGraph &, CGSCCUpdateResult &) {             \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
  #define DUMMY_FUNCTION_PASS(NAME, PASS_NAME)                    \
  struct PASS_NAME : public PassInfoMixin<PASS_NAME> {                         \
    template <typename... Ts> PASS_NAME(Ts &&...) {}                           \
    PreservedAnalyses run(Function &, FunctionAnalysisManager &) {             \
      return PreservedAnalyses::all();                                         \
    }                                                                          \
  };
#define DUMMY_MACHINE_FUNCTION_PASS(NAME, PASS_NAME)              \
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
}

template <typename DerivedT, typename TargetMachineT>
class AMDGPUCodeGenPassBuilder
    : public CodeGenPassBuilder<DerivedT, TargetMachineT> {
public:
  explicit AMDGPUCodeGenPassBuilder(TargetMachineT &TM,
                                    CGPassBuilderOption Opt,
                                    PassInstrumentationCallbacks *PIC)
      : CodeGenPassBuilder<DerivedT, TargetMachineT>(TM, Opt, PIC) {}

  bool isPassEnabled(const cl::opt<bool> &Opt,
                     CodeGenOptLevel Level = CodeGenOptLevel::Default) const {
    if (Opt.getNumOccurrences())
      return Opt;
    if (CodeGenPassBuilder<DerivedT, TargetMachineT>::TM.getOptLevel() < Level)
      return false;
    return Opt;
  }

  AMDGPUTargetMachine &getAMDGPUTargetMachine() const {
    return CodeGenPassBuilder<DerivedT, TargetMachineT>::template getTM<
        AMDGPUTargetMachine>();
  }
  void addIRPasses(
      typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddIRPass &) const;
  void addCodeGenPrepare(
      typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddIRPass &) const;
  void addPreISel(
      typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddIRPass &) const;
  Error addInstSelector(
      typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddMachinePass &)
      const;
  void addEarlyCSEOrGVNPass(
      typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddIRPass &) const;
  void addStraightLineScalarOptimizationPasses(
      typename CodeGenPassBuilder<DerivedT, TargetMachineT>::AddIRPass &) const;
};

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
     //  addPass(createFunctionToLoopPassAdaptor(LICMPass())); NEW_ERROR
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