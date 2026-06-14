//===- TopDownVec.h ---------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A Top-Down Vectorizer pass.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_TOPDOWNVEC_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_TOPDOWNVEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"

namespace llvm::sandboxir {

class LLVM_ABI TopDownVec final : public RegionPass {
  bool Change = false;
  DenseSet<Instruction *> DeadInstrCandidates;
  std::unique_ptr<InstrMaps> IMaps;
  unsigned long TopDownInvocationCnt = 0;



  class ActionsVector {
  private:
    SmallVector<std::unique_ptr<Action>, 16> Actions;

  public:
    auto begin() const { return Actions.begin(); }
    auto end() const { return Actions.end(); }
    auto rbegin() const { return Actions.rbegin(); }
    auto rend() const { return Actions.rend(); }
    void push_back(std::unique_ptr<Action> &&ActPtr) {
      ActPtr->Idx = Actions.size();
      Actions.push_back(std::move(ActPtr));
    }
    void clear() { Actions.clear(); }
#ifndef NDEBUG
    void print(raw_ostream &OS) const;
    void dump() const;
#endif // NDEBUG
  };
  ActionsVector Actions;
  unsigned DebugBndlCnt = 0;

  Action *vectorizeRec(ArrayRef<Value *> Bndl, unsigned Depth,
                       LegalityAnalysis &Legality);
  Value *emitVectors();
  bool tryVectorize(ArrayRef<Value *> Seeds, LegalityAnalysis &Legality);

public:
  TopDownVec() : RegionPass("top-down-vec") {}
  bool runOnRegion(Region &Rgn, const Analyses &A) final;
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_TOPDOWNVEC_H
