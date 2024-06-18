#include "AMDGPUTargetMachine.h"
#include "llvm/Passes/CodeGenPassBuilder.h"

using namespace llvm;

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

  //    FIXME: Port MachineScheduler Pass
  //    ScheduleDAGInstrs *
  //   createMachineScheduler(MachineSchedContext *C) const ;

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
