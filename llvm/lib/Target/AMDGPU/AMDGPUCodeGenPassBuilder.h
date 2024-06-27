#include "AMDGPUCodeGenPassBuilderImpl.h"
class GCNCodeGenPassBuilder final
    : public AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder, GCNTargetMachine> {
public:
  GCNCodeGenPassBuilder(GCNTargetMachine &TM, CGPassBuilderOption Opt,
                        PassInstrumentationCallbacks *PIC)
      : AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder, GCNTargetMachine>(
            TM, Opt, PIC) {
    Opt.RequiresCodeGenSCCOrder = true;
  // Exceptions and StackMaps are not supported, so these passes will never do
  // anything.
  // Garbage collection is not supported.
   disablePass<StackMapLivenessPass, FuncletLayoutPass, ShadowStackGCLoweringPass>();
  }

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
  Error addFastRegAlloc(AddMachinePass &);
  void addOptimizedRegAlloc(AddMachinePass &);
  void addAsmPrinter(AddMachinePass &, CreateMCStreamer) const;

  FunctionPass *createSGPRAllocPass(bool Optimized);
  FunctionPass *createVGPRAllocPass(bool Optimized);
  FunctionPass *createRegAllocPass(bool Optimized);

  Error addRegAssignmentFast(AddMachinePass &) const;
  Error addRegAssignmentOptimized(AddMachinePass &) const;

  void addPreRegAlloc(AddMachinePass &) const;
  void addPreRewrite(AddMachinePass &) const;
  void addPostRegAlloc(AddMachinePass &) const;
  void addPreSched2(AddMachinePass &) const;
  void addPreEmitPass(AddMachinePass &) const;
};