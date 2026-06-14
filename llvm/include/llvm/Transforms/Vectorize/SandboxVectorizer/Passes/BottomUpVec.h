//===- BottomUpVec.h --------------------------------------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// A Bottom-Up Vectorizer pass.
//

#ifndef LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H
#define LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H

#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/SandboxIR/Constant.h"
#include "llvm/SandboxIR/Pass.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"

namespace llvm::sandboxir {

/// This is a simple bottom-up vectorizer Region pass.
/// It expects a "seed slice" as an input in the Region's Aux vector.
/// The "seed slice" is a vector of instructions that can be used as a starting
/// point for vectorization, like stores to consecutive memory addresses.
/// Starting from the seed instructions, it walks up the def-use chain looking
/// for more instructions that can be vectorized. This pass will generate vector
/// code if it can legally vectorize the code, regardless of whether it is
/// profitable or not. For now profitability is checked at the end of the region
/// pass pipeline by a dedicated pass that accepts or rejects the IR
/// transaction, depending on the cost.
class LLVM_ABI BottomUpVec final : public RegionPass {
  bool Change = false;
  /// The original instructions that are potentially dead after vectorization.
  DenseSet<Instruction *> DeadInstrCandidates;
  /// Maps scalars to vectors.
  std::unique_ptr<InstrMaps> IMaps;
  /// Counter used for force-stopping the vectorizer after this many
  /// invocations. Used for debugging miscompiles.
  unsigned long BottomUpInvocationCnt = 0;

  /// Helper class describing how(if) to vectorize the code.
  class ActionsVector {
  private:
    SmallVector<std::unique_ptr<Action>, 16> Actions;

  public:
    auto begin() const { return Actions.begin(); }
    auto end() const { return Actions.end(); }
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
  /// Helper counter for debugging. It counts the bundles that we attempt to
  /// vectorize in vectorizeRec().
  unsigned DebugBndlCnt = 0;

  /// Recursively try to vectorize \p Bndl and its operands. This populates the
  /// `Actions` vector.
  Action *vectorizeRec(ArrayRef<Value *> Bndl, ArrayRef<Value *> UserBndl,
                       unsigned Depth, LegalityAnalysis &Legality);
  /// Generate vector instructions based on `Actions` and return the last vector
  /// created.
  Value *emitVectors();
  /// Entry point for vectorization starting from \p Seeds.
  bool tryVectorize(ArrayRef<Value *> Seeds, LegalityAnalysis &Legality);

public:
  BottomUpVec() : RegionPass("bottom-up-vec") {}
  bool runOnRegion(Region &Rgn, const Analyses &A) final;
};

} // namespace llvm::sandboxir

#endif // LLVM_TRANSFORMS_VECTORIZE_SANDBOXVECTORIZER_PASSES_BOTTOMUPVEC_H
