

 #include "AMDGPUCodeGenPassBuilderImpl.h"
class GCNCodeGenPassBuilder final
    : public AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder, GCNTargetMachine> {
public:
  GCNCodeGenPassBuilder(GCNTargetMachine &TM, CGPassBuilderOption Opt,
                        PassInstrumentationCallbacks *PIC)
      : AMDGPUCodeGenPassBuilder<GCNCodeGenPassBuilder, GCNTargetMachine>(
            TM, Opt, PIC) {
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
  Error addFastRegAlloc(AddMachinePass &);
  void addOptimizedRegAlloc(AddMachinePass &);
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