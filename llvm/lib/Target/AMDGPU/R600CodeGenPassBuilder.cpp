//===-- R600CodeGenPassBuilder.cpp ------ Build R600 CodeGen pipeline -----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
#include "R600CodeGenPassBuilder.h"
#include "llvm/Transforms/Scalar/StructurizeCFG.h"
using namespace llvm; 

extern cl::opt<bool>
    EnableR600StructurizeCFG;

extern cl::opt<bool> EnableR600IfConvert;

R600CodeGenPassBuilder::R600CodeGenPassBuilder(
    R600TargetMachine &TM, const CGPassBuilderOption &Opts,
    PassInstrumentationCallbacks *PIC)
    :AMDGPUCodeGenPassBuilder<R600CodeGenPassBuilder, R600TargetMachine>(TM, Opts, PIC) {
  Opt.RequiresCodeGenSCCOrder = true;
}

void R600CodeGenPassBuilder::addPreISel(AddIRPass &addPass) const {
  // TODO: Add passes pre instruction selection.
  AMDGPUCodeGenPassBuilder<R600CodeGenPassBuilder, R600TargetMachine>::addPreISel(addPass);

  if (EnableR600StructurizeCFG)
    addPass(StructurizeCFGPass());
}

void R600CodeGenPassBuilder::addAsmPrinter(AddMachinePass &addPass,
                                           CreateMCStreamer) const {
  // TODO: Add AsmPrinter.
}

Error R600CodeGenPassBuilder::addInstSelector(AddMachinePass &addPass) const {
  // TODO: Add instruction selector.
 // addPass(createR600ISelDag(getAMDGPUTargetMachine(), getOptLevel()));
  return Error::success();
}

void R600CodeGenPassBuilder::addPreRegAlloc(AddMachinePass &addPass) const {
  addPass(R600VectorRegMerger());
};

  void R600CodeGenPassBuilder::addPreSched2(AddMachinePass &addPass) const {
 addPass(R600EmitClauseMarkers());
  if (EnableR600IfConvert)
    addPass(IfConverterPass()); 
  addPass(R600ClauseMergePass());
  };
  void R600CodeGenPassBuilder::addPreEmitPass(AddMachinePass &addPass) const {
  addPass(R600MachineCFGStructurizerPass());
  addPass(R600ExpandSpecialInstrsPass());
  addPass(FinalizeMachineBundlesPass());
  addPass(R600Packetizer());
  addPass(R600ControlFlowFinalizer());
  };

Error R600TargetMachine::buildCodeGenPipeline(
    ModulePassManager &MPM, raw_pwrite_stream &Out, raw_pwrite_stream *DwoOut,
    CodeGenFileType FileType, const CGPassBuilderOption &Opts,
    PassInstrumentationCallbacks *PIC) {
  auto CGPB = R600CodeGenPassBuilder(*this, Opts, PIC);
  return CGPB.buildPipeline(MPM, Out, DwoOut, FileType);
}