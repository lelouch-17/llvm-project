; RUN: llc -enable-new-pm -O0 -mtriple=amdgcn-amd-amdhsa  -debug-pass-manager %s -o /dev/null 2>&1 | FileCheck -check-prefix=O0-PIPELINE %s

; O0-PIPELINE: Running pass: RequireAnalysisPass<llvm::ProfileSummaryAnalysis, llvm::Module, llvm::AnalysisManager<llvm::Module>> on [module]
; O0-PIPELINE-NEXT: Running analysis: ProfileSummaryAnalysis on [module]
; O0-PIPELINE-NEXT: Running pass: RequireAnalysisPass<llvm::CollectorMetadataAnalysis, llvm::Module, llvm::AnalysisManager<llvm::Module>> on [module]
; O0-PIPELINE-NEXT: Running analysis: CollectorMetadataAnalysis on [module]
; O0-PIPELINE-NEXT: Running pass: PreISelIntrinsicLoweringPass on [module]
; O0-PIPELINE-NEXT: Running analysis: InnerAnalysisManagerProxy<llvm::FunctionAnalysisManager, llvm::Module> on [module]
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPURemoveIncompatibleFunctionsPass on [module]
; O0-PIPELINE-NEXT: Running pass: AMDGPUPrintfRuntimeBindingPass on [module]
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPUCtorDtorLoweringLegacyPass on [module]
; O0-PIPELINE-NEXT: Running pass: AMDGPUAlwaysInlinePass on [module]
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::AlwaysInlinerLegacyPass on [module]
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPUOpenCLEnqueuedBlockLoweringPass on [module]
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPULowerModuleLDSLegacyPass on [module]
; O0-PIPELINE-NEXT: Running pass: VerifierPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running analysis: VerifierAnalysis on f
; O0-PIPELINE-NEXT: Running pass: ShadowStackGCLoweringPass on [module]
; O0-PIPELINE-NEXT: Running pass: LowerConstantIntrinsicsPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running analysis: TargetLibraryAnalysis on f
; O0-PIPELINE-NEXT: Running pass: UnreachableBlockElimPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running pass: EntryExitInstrumenterPass on f (1 instruction)
; O0-PIPELINE-NEXT: Invalidating analysis: VerifierAnalysis on f
; O0-PIPELINE-NEXT: Running pass: ScalarizeMaskedMemIntrinPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running analysis: TargetIRAnalysis on f
; O0-PIPELINE-NEXT: Running pass: ExpandReductionsPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running pass: EarlyCSEPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running analysis: DominatorTreeAnalysis on f
; O0-PIPELINE-NEXT: Running analysis: AssumptionAnalysis on f
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPUAnnotateKernelFeaturesPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running pass: AMDGPULowerKernelArgumentsPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running pass: LowerSwitchPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running analysis: LazyValueAnalysis on f
; O0-PIPELINE-NEXT: Running pass: LowerInvokePass on f (1 instruction)
; O0-PIPELINE-NEXT: Running pass: UnreachableBlockElimPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running pass: CallBrPreparePass on f (1 instruction)
; O0-PIPELINE-NEXT: Running pass: SafeStackPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running pass: StackProtectorPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running analysis: SSPLayoutAnalysis on f
; O0-PIPELINE-NEXT: Running pass: VerifierPass on f (1 instruction)
; O0-PIPELINE-NEXT: Running analysis: VerifierAnalysis on f
; O0-PIPELINE-NEXT: Running analysis: MachineModuleAnalysis on [module]
; O0-PIPELINE-NEXT: Running analysis: InnerAnalysisManagerProxy<llvm::MachineFunctionAnalysisManager, llvm::Module> on [module]
; O0-PIPELINE-NEXT: Running analysis: OuterAnalysisManagerProxy<llvm::ModuleAnalysisManager, llvm::MachineFunction> on f
; O0-PIPELINE-NEXT: Running pass: FinalizeISelPass on f
; O0-PIPELINE-NEXT: Running pass: LocalStackSlotPass on f
; O0-PIPELINE-NEXT: Running pass: PHIEliminationPass on f
; O0-PIPELINE-NEXT: Running pass: TwoAddressInstructionPass on f
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIFixVGPRCopiesPass on f
; O0-PIPELINE-NEXT: Running pass: RemoveRedundantDebugValuesPass on f
; O0-PIPELINE-NEXT: Running pass: PrologEpilogInserterPass on f
; O0-PIPELINE-NEXT: Running pass: ExpandPostRAPseudosPass on f
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIPostRABundlerPass on f
; O0-PIPELINE-NEXT: Running pass: FEntryInserterPass on f
; O0-PIPELINE-NEXT: Running pass: XRayInstrumentationPass on f
; O0-PIPELINE-NEXT: Running pass: PatchableFunctionPass on f
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIMemoryLegalizerPass on f
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIInsertWaitcntsPass on f
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIModeRegisterPass on f
; O0-PIPELINE-NEXT: Running pass: (anonymous namespace)::SILateBranchLoweringPass on f
; O0-PIPELINE-NEXT: Running pass: PostRAHazardRecognizerPass on f
; O0-PIPELINE-NEXT: Running pass: BranchRelaxationPass on f
; O0-PIPELINE-NEXT: Running pass: FuncletLayoutPass on f
; O0-PIPELINE-NEXT: Running pass: StackMapLivenessPass on f
; O0-PIPELINE-NEXT: Running pass: LiveDebugValuesPass on f
; O0-PIPELINE-NEXT: Running pass: MachineSanitizerBinaryMetadata on f
; O0-PIPELINE-NEXT: Running pass: FreeMachineFunctionPass on f
; O0-PIPELINE-NEXT: Clearing all analysis results for: f
; O0-PIPELINE-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<llvm::FunctionAnalysisManager, llvm::Module> on [module]
; O0-PIPELINE-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<llvm::MachineFunctionAnalysisManager, llvm::Module> on [module]

; O3-PIPELINE: Running pass: RequireAnalysisPass<llvm::ProfileSummaryAnalysis, llvm::Module, llvm::AnalysisManager<llvm::Module>> on [module]
; O3-PIPELINE-NEXT: Running analysis: ProfileSummaryAnalysis on [module]
; O3-PIPELINE-NEXT: Running pass: RequireAnalysisPass<llvm::CollectorMetadataAnalysis, llvm::Module, llvm::AnalysisManager<llvm::Module>> on [module]
; O3-PIPELINE-NEXT: Running analysis: CollectorMetadataAnalysis on [module]
; O3-PIPELINE-NEXT: Running pass: PreISelIntrinsicLoweringPass on [module]
; O3-PIPELINE-NEXT: Running analysis: InnerAnalysisManagerProxy<llvm::FunctionAnalysisManager, llvm::Module> on [module]
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPURemoveIncompatibleFunctionsPass on [module]
; O3-PIPELINE-NEXT: Running pass: AMDGPUPrintfRuntimeBindingPass on [module]
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPUCtorDtorLoweringLegacyPass on [module]
; O3-PIPELINE-NEXT: Running pass: AMDGPUImageIntrinsicOptimizerPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: AMDGPUAlwaysInlinePass on [module]
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::AlwaysInlinerLegacyPass on [module]
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPUOpenCLEnqueuedBlockLoweringPass on [module]
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPULowerModuleLDSLegacyPass on [module]
; O3-PIPELINE-NEXT: Running pass: AMDGPUAttributorPass on [module]
; O3-PIPELINE-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<llvm::FunctionAnalysisManager, llvm::Module> on [module]
; O3-PIPELINE-NEXT: Running analysis: InnerAnalysisManagerProxy<llvm::FunctionAnalysisManager, llvm::Module> on [module]
; O3-PIPELINE-NEXT: Running pass: InferAddressSpacesPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: AssumptionAnalysis on f
; O3-PIPELINE-NEXT: Running analysis: TargetIRAnalysis on f
; O3-PIPELINE-NEXT: Running pass: AMDGPUPromoteAllocaPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: SeparateConstOffsetFromGEPPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: DominatorTreeAnalysis on f
; O3-PIPELINE-NEXT: Running analysis: LoopAnalysis on f
; O3-PIPELINE-NEXT: Running analysis: TargetLibraryAnalysis on f
; O3-PIPELINE-NEXT: Running pass: StraightLineStrengthReducePass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: ScalarEvolutionAnalysis on f
; O3-PIPELINE-NEXT: Running pass: GVNPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: AAManager on f
; O3-PIPELINE-NEXT: Running analysis: BasicAA on f
; O3-PIPELINE-NEXT: Running analysis: ScopedNoAliasAA on f
; O3-PIPELINE-NEXT: Running analysis: TypeBasedAA on f
; O3-PIPELINE-NEXT: Running analysis: OuterAnalysisManagerProxy<llvm::ModuleAnalysisManager, llvm::Function> on f
; O3-PIPELINE-NEXT: Running analysis: AMDGPUAA on f
; O3-PIPELINE-NEXT: Running analysis: MemoryDependenceAnalysis on f
; O3-PIPELINE-NEXT: Running analysis: OptimizationRemarkEmitterAnalysis on f
; O3-PIPELINE-NEXT: Running pass: NaryReassociatePass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: EarlyCSEPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPUAAWrapperPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: AMDGPUCodeGenPreparePass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: UniformityInfoAnalysis on f
; O3-PIPELINE-NEXT: Running analysis: CycleAnalysis on f
; O3-PIPELINE-NEXT: Running pass: VerifierPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: VerifierAnalysis on f
; O3-PIPELINE-NEXT: Running pass: LoopSimplifyPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: LCSSAPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: MergeICmpsPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: ExpandMemCmpPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: ShadowStackGCLoweringPass on [module]
; O3-PIPELINE-NEXT: Running pass: LowerConstantIntrinsicsPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: UnreachableBlockElimPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: ConstantHoistingPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: BlockFrequencyAnalysis on f
; O3-PIPELINE-NEXT: Running analysis: BranchProbabilityAnalysis on f
; O3-PIPELINE-NEXT: Running analysis: PostDominatorTreeAnalysis on f
; O3-PIPELINE-NEXT: Running pass: ReplaceWithVeclib on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: PartiallyInlineLibCallsPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: EntryExitInstrumenterPass on f (1 instruction)
; O3-PIPELINE-NEXT: Invalidating analysis: ScalarEvolutionAnalysis on f
; O3-PIPELINE-NEXT: Invalidating analysis: MemoryDependenceAnalysis on f
; O3-PIPELINE-NEXT: Invalidating analysis: CycleAnalysis on f
; O3-PIPELINE-NEXT: Invalidating analysis: UniformityInfoAnalysis on f
; O3-PIPELINE-NEXT: Invalidating analysis: VerifierAnalysis on f
; O3-PIPELINE-NEXT: Running pass: ScalarizeMaskedMemIntrinPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: ExpandReductionsPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: GVNPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: MemoryDependenceAnalysis on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPUAnnotateKernelFeaturesPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: AMDGPULowerKernelArgumentsPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: CodeGenPreparePass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: LoadStoreVectorizerPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: ScalarEvolutionAnalysis on f
; O3-PIPELINE-NEXT: Running pass: LowerSwitchPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: LazyValueAnalysis on f
; O3-PIPELINE-NEXT: Running pass: LowerInvokePass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: UnreachableBlockElimPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: CallBrPreparePass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: SafeStackPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running pass: StackProtectorPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: SSPLayoutAnalysis on f
; O3-PIPELINE-NEXT: Running pass: VerifierPass on f (1 instruction)
; O3-PIPELINE-NEXT: Running analysis: VerifierAnalysis on f
; O3-PIPELINE-NEXT: Running analysis: MachineModuleAnalysis on [module]
; O3-PIPELINE-NEXT: Running analysis: InnerAnalysisManagerProxy<llvm::MachineFunctionAnalysisManager, llvm::Module> on [module]
; O3-PIPELINE-NEXT: Running analysis: OuterAnalysisManagerProxy<llvm::ModuleAnalysisManager, llvm::MachineFunction> on f
; O3-PIPELINE-NEXT: Running pass: FinalizeISelPass on f
; O3-PIPELINE-NEXT: Running pass: EarlyTailDuplicatePass on f
; O3-PIPELINE-NEXT: Running pass: OptimizePHIsPass on f
; O3-PIPELINE-NEXT: Running pass: StackColoringPass on f
; O3-PIPELINE-NEXT: Running pass: LocalStackSlotPass on f
; O3-PIPELINE-NEXT: Running pass: EarlyMachineLICMPass on f
; O3-PIPELINE-NEXT: Running pass: MachineCSEPass on f
; O3-PIPELINE-NEXT: Running pass: MachineSinkingPass on f
; O3-PIPELINE-NEXT: Running pass: PeepholeOptimizerPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIFoldOperandsPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::GCNDPPCombinePass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SILoadStoreOptimizerPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIPeepholeSDWAPass on f
; O3-PIPELINE-NEXT: Running pass: EarlyMachineLICMPass on f
; O3-PIPELINE-NEXT: Running pass: MachineCSEPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIFoldOperandsPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIShrinkInstructionsPass on f
; O3-PIPELINE-NEXT: Running pass: DetectDeadLanesPass on f
; O3-PIPELINE-NEXT: Running pass: InitUndefPass on f
; O3-PIPELINE-NEXT: Running pass: ProcessImplicitDefsPass on f
; O3-PIPELINE-NEXT: Running pass: PHIEliminationPass on f
; O3-PIPELINE-NEXT: Running pass: TwoAddressInstructionPass on f
; O3-PIPELINE-NEXT: Running pass: RegisterCoalescerPass on f
; O3-PIPELINE-NEXT: Running pass: RenameIndependentSubregsPass on f
; O3-PIPELINE-NEXT: Running pass: MachineSchedulerPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SILowerWWMCopiesPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::GCNNSAReassignPass on f
; O3-PIPELINE-NEXT: Running pass: VirtRegRewriterPass on f
; O3-PIPELINE-NEXT: Running pass: StackSlotColoringPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIFixVGPRCopiesPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIOptimizeExecMaskingPass on f
; O3-PIPELINE-NEXT: Running pass: RemoveRedundantDebugValuesPass on f
; O3-PIPELINE-NEXT: Running pass: PostRAMachineSinkingPass on f
; O3-PIPELINE-NEXT: Running pass: ShrinkWrapPass on f
; O3-PIPELINE-NEXT: Running pass: PrologEpilogInserterPass on f
; O3-PIPELINE-NEXT: Running pass: BranchFolderPass on f
; O3-PIPELINE-NEXT: Running pass: TailDuplicatePass on f
; O3-PIPELINE-NEXT: Running pass: MachineLateInstrsCleanupPass on f
; O3-PIPELINE-NEXT: Running pass: MachineCopyPropagationPass on f
; O3-PIPELINE-NEXT: Running pass: ExpandPostRAPseudosPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIShrinkInstructionsPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIPostRABundlerPass on f
; O3-PIPELINE-NEXT: Running pass: PostRASchedulerPass on f
; O3-PIPELINE-NEXT: Running pass: MachineBlockPlacementPass on f
; O3-PIPELINE-NEXT: Running pass: FEntryInserterPass on f
; O3-PIPELINE-NEXT: Running pass: XRayInstrumentationPass on f
; O3-PIPELINE-NEXT: Running pass: PatchableFunctionPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::GCNCreateVOPDPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIMemoryLegalizerPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIInsertWaitcntsPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIModeRegisterPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIInsertHardClausesPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SILateBranchLoweringPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::SIPreEmitPeepholePass on f
; O3-PIPELINE-NEXT: Running pass: PostRAHazardRecognizerPass on f
; O3-PIPELINE-NEXT: Running pass: (anonymous namespace)::AMDGPUInsertDelayAluPass on f
; O3-PIPELINE-NEXT: Running pass: BranchRelaxationPass on f
; O3-PIPELINE-NEXT: Running pass: FuncletLayoutPass on f
; O3-PIPELINE-NEXT: Running pass: StackMapLivenessPass on f
; O3-PIPELINE-NEXT: Running pass: LiveDebugValuesPass on f
; O3-PIPELINE-NEXT: Running pass: MachineSanitizerBinaryMetadata on f
; O3-PIPELINE-NEXT: Running pass: FreeMachineFunctionPass on f
; O3-PIPELINE-NEXT: Clearing all analysis results for: f
; O3-PIPELINE-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<llvm::FunctionAnalysisManager, llvm::Module> on [module]
; O3-PIPELINE-NEXT: Invalidating analysis: InnerAnalysisManagerProxy<llvm::MachineFunctionAnalysisManager, llvm::Module> on [module]

define void @f() {
  ret void
}