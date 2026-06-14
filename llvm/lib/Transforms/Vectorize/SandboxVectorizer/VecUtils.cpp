//===- VecUtils.cpp -------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "llvm/Transforms/Vectorize/SandboxVectorizer/VecUtils.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/Legality.h"
#include "llvm/Transforms/Vectorize/SandboxVectorizer/InstrMaps.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "sandbox-vectorizer"

namespace llvm::sandboxir {

unsigned VecUtils::getFloorPowerOf2(unsigned Num) {
  if (Num == 0)
    return Num;
  unsigned Mask = Num;
  Mask >>= 1;
  for (unsigned ShiftBy = 1; ShiftBy < sizeof(Num) * 8; ShiftBy <<= 1)
    Mask |= Mask >> ShiftBy;
  return Num & ~Mask;
}

Value *VecUtils::createVectorInstr(ArrayRef<Value *> Bndl,
                                   ArrayRef<Value *> Operands) {
  assert(all_of(Bndl, [](auto *V) { return isa<Instruction>(V); }) &&
         "Expect Instructions!");
  auto &Ctx = Bndl[0]->getContext();

  Type *ScalarTy = VecUtils::getElementType(Utils::getExpectedType(Bndl[0]));
  auto *VecTy = VecUtils::getWideType(ScalarTy, VecUtils::getNumLanes(Bndl));

  BasicBlock::iterator WhereIt = VecUtils::getInsertPointAfterInstrs(
      Bndl, cast<Instruction>(Bndl[0])->getParent());

  auto Opcode = cast<Instruction>(Bndl[0])->getOpcode();
  Value *NewI = nullptr;
  switch (Opcode) {
  case Instruction::Opcode::ZExt:
  case Instruction::Opcode::SExt:
  case Instruction::Opcode::FPToUI:
  case Instruction::Opcode::FPToSI:
  case Instruction::Opcode::FPExt:
  case Instruction::Opcode::PtrToInt:
  case Instruction::Opcode::IntToPtr:
  case Instruction::Opcode::SIToFP:
  case Instruction::Opcode::UIToFP:
  case Instruction::Opcode::Trunc:
  case Instruction::Opcode::FPTrunc:
  case Instruction::Opcode::BitCast: {
    assert(Operands.size() == 1u && "Casts are unary!");
    NewI = CastInst::create(VecTy, Opcode, Operands[0], WhereIt, Ctx,
                            "VCast");
    break;
  }
  case Instruction::Opcode::FCmp:
  case Instruction::Opcode::ICmp: {
    auto Pred = cast<CmpInst>(Bndl[0])->getPredicate();
    assert(all_of(drop_begin(Bndl),
                  [Pred](auto *SBV) {
                    return cast<CmpInst>(SBV)->getPredicate() == Pred;
                  }) &&
           "Expected same predicate across bundle.");
    NewI = CmpInst::create(Pred, Operands[0], Operands[1], WhereIt, Ctx,
                           "VCmp");
    break;
  }
  case Instruction::Opcode::Select: {
    NewI = SelectInst::create(Operands[0], Operands[1], Operands[2], WhereIt,
                              Ctx, "Vec");
    break;
  }
  case Instruction::Opcode::FNeg: {
    auto *UOp0 = cast<UnaryOperator>(Bndl[0]);
    auto OpC = UOp0->getOpcode();
    NewI = UnaryOperator::createWithCopiedFlags(OpC, Operands[0], UOp0,
                                                WhereIt, Ctx, "Vec");
    break;
  }
  case Instruction::Opcode::Add:
  case Instruction::Opcode::FAdd:
  case Instruction::Opcode::Sub:
  case Instruction::Opcode::FSub:
  case Instruction::Opcode::Mul:
  case Instruction::Opcode::FMul:
  case Instruction::Opcode::UDiv:
  case Instruction::Opcode::SDiv:
  case Instruction::Opcode::FDiv:
  case Instruction::Opcode::URem:
  case Instruction::Opcode::SRem:
  case Instruction::Opcode::FRem:
  case Instruction::Opcode::Shl:
  case Instruction::Opcode::LShr:
  case Instruction::Opcode::AShr:
  case Instruction::Opcode::And:
  case Instruction::Opcode::Or:
  case Instruction::Opcode::Xor: {
    auto *BinOp0 = cast<BinaryOperator>(Bndl[0]);
    auto *LHS = Operands[0];
    auto *RHS = Operands[1];
    NewI = BinaryOperator::createWithCopiedFlags(
        BinOp0->getOpcode(), LHS, RHS, BinOp0, WhereIt, Ctx, "Vec");
    break;
  }
  case Instruction::Opcode::Load: {
    auto *Ld0 = cast<LoadInst>(Bndl[0]);
    Value *Ptr = Ld0->getPointerOperand();
    NewI = LoadInst::create(VecTy, Ptr, Ld0->getAlign(), WhereIt, Ctx,
                            "VecL");
    break;
  }
  case Instruction::Opcode::Store: {
    auto Align = cast<StoreInst>(Bndl[0])->getAlign();
    Value *Val = Operands[0];
    Value *Ptr = Operands[1];
    NewI = StoreInst::create(Val, Ptr, Align, WhereIt, Ctx);
    break;
  }
  case Instruction::Opcode::UncondBr:
  case Instruction::Opcode::CondBr:
  case Instruction::Opcode::Ret:
  case Instruction::Opcode::PHI:
  case Instruction::Opcode::AddrSpaceCast:
  case Instruction::Opcode::Call:
  case Instruction::Opcode::GetElementPtr:
    llvm_unreachable("Unimplemented");
    break;
  default:
    llvm_unreachable("Unimplemented");
    break;
  }
  return NewI;
}

void VecUtils::tryEraseDeadInstrs(DenseSet<Instruction *> &DeadInstrCandidates, StringRef PassName) {
  DenseMap<BasicBlock *, SmallVector<Instruction *>> SortedDeadInstrCandidates;
  // The dead instrs could span BBs, so we need to collect and sort them per BB.
  for (auto *DeadI : DeadInstrCandidates)
    SortedDeadInstrCandidates[DeadI->getParent()].push_back(DeadI);
  for (auto &Pair : SortedDeadInstrCandidates)
    sort(Pair.second,
         [](Instruction *I1, Instruction *I2) { return I1->comesBefore(I2); });
  for (const auto &Pair : SortedDeadInstrCandidates) {
    for (Instruction *I : reverse(Pair.second)) {
      if (I->hasNUses(0)) {
        // Erase the dead instructions bottom-to-top.
        LLVM_DEBUG(dbgs() << PassName << "Erase dead: " << *I << "\n");
        I->eraseFromParent();
      }
    }
  }
  DeadInstrCandidates.clear();
}

Value *VecUtils::createShuffle(Value *VecOp, const ShuffleMask &Mask,
                               BasicBlock *UserBB) {
  BasicBlock::iterator WhereIt = VecUtils::getInsertPointAfterInstrs({VecOp}, UserBB);
  return ShuffleVectorInst::create(VecOp, VecOp, Mask, WhereIt,
                                   VecOp->getContext(), "VShuf");
}

Value *VecUtils::createPack(ArrayRef<Value *> ToPack, BasicBlock *UserBB) {
  BasicBlock::iterator WhereIt = VecUtils::getInsertPointAfterInstrs(ToPack, UserBB);

  Type *ScalarTy = VecUtils::getCommonScalarType(ToPack);
  unsigned Lanes = VecUtils::getNumLanes(ToPack);
  Type *VecTy = VecUtils::getWideType(ScalarTy, Lanes);

  // Create a series of pack instructions.
  Value *LastInsert = PoisonValue::get(VecTy);

  Context &Ctx = ToPack[0]->getContext();

  unsigned InsertIdx = 0;
  for (Value *Elm : ToPack) {
    // An element can be either scalar or vector. We need to generate different
    // IR for each case.
    if (Elm->getType()->isVectorTy()) {
      unsigned NumElms =
          cast<FixedVectorType>(Elm->getType())->getNumElements();
      for (auto ExtrLane : seq<int>(0, NumElms)) {
        // We generate extract-insert pairs, for each lane in `Elm`.
        Constant *ExtrLaneC =
            ConstantInt::getSigned(Type::getInt32Ty(Ctx), ExtrLane);
        // This may return a Constant if Elm is a Constant.
        auto *ExtrI =
            ExtractElementInst::create(Elm, ExtrLaneC, WhereIt, Ctx, "VPack");
        if (!isa<Constant>(ExtrI))
          WhereIt = std::next(cast<Instruction>(ExtrI)->getIterator());
        Constant *InsertLaneC =
            ConstantInt::getSigned(Type::getInt32Ty(Ctx), InsertIdx++);
        // This may also return a Constant if ExtrI is a Constant.
        LastInsert = InsertElementInst::create(LastInsert, ExtrI, InsertLaneC,
                                               WhereIt, Ctx, "VPack");
        if (!isa<Constant>(LastInsert))
          WhereIt = std::next(cast<Instruction>(LastInsert)->getIterator());
      }
    } else {
      Constant *InsertLaneC =
          ConstantInt::getSigned(Type::getInt32Ty(Ctx), InsertIdx++);
      LastInsert = InsertElementInst::create(LastInsert, Elm, InsertLaneC,
                                             WhereIt, Ctx, "VPack");
      if (!isa<Constant>(LastInsert))
        WhereIt = std::next(cast<Instruction>(LastInsert)->getIterator());
    }
  }
  return LastInsert;
}

void VecUtils::collectPotentiallyDeadInstrs(ArrayRef<Value *> Bndl, DenseSet<Instruction *> &DeadInstrCandidates) {
  for (Value *V : Bndl)
    DeadInstrCandidates.insert(cast<Instruction>(V));
  // Also collect the GEPs of vectorized loads and stores.
  auto Opcode = cast<Instruction>(Bndl[0])->getOpcode();
  switch (Opcode) {
  case Instruction::Opcode::Load: {
    for (Value *V : drop_begin(Bndl))
      if (auto *Ptr =
              dyn_cast<Instruction>(cast<LoadInst>(V)->getPointerOperand()))
        DeadInstrCandidates.insert(Ptr);
    break;
  }
  case Instruction::Opcode::Store: {
    for (Value *V : drop_begin(Bndl))
      if (auto *Ptr =
              dyn_cast<Instruction>(cast<StoreInst>(V)->getPointerOperand()))
        DeadInstrCandidates.insert(Ptr);
    break;
  }
  default:
    break;
  }
}

void VecUtils::emitUnpacksForExternalUses(ArrayRef<Value *> Bndl, Value *Vec,
                                          const InstrMaps &IMaps) {
  // Find where we should emit the unpacks.
  BasicBlock::iterator WhereIt;
  if (auto *VecI = dyn_cast<Instruction>(Vec)) {
    WhereIt = std::next(VecI->getIterator());
  } else {
    // If Vec is a constant then it should be safe to emit the unpacks at the
    // top of the block.
    // Note: Extracts from constants are usually folded to constants.
    assert(isa<Constant>(Vec) && "Expected constant!");
    assert(isa<Instruction>(Bndl[0]) &&
           "A widened Bndl should contain instrs!");
    BasicBlock *BB = cast<Instruction>(Bndl[0])->getParent();
    WhereIt =
        BB->empty()
            ? BB->begin()
            : std::next(
                  VecUtils::getLastPHIOrSelf(&*BB->begin())->getIterator());
  }

  for (auto [Lane, Elm] : VecUtils::enumerateLanes(Bndl)) {
    for (User *U : Elm->users()) {
      // Skip users that we just vectorized.
      if (IMaps.isVectorized(U))
        continue;
      auto *LastUnpackV = VecUtils::unpack(Vec, Elm->getType(), Lane, WhereIt);
      Elm->replaceAllUsesWith(LastUnpackV);
    }
  }
}

#ifndef NDEBUG
template <typename T> static void dumpImpl(ArrayRef<T *> Bndl) {
  for (auto [Idx, V] : enumerate(Bndl))
    dbgs() << Idx << "." << *V << "\n";
}
void VecUtils::dump(ArrayRef<Value *> Bndl) { dumpImpl(Bndl); }
void VecUtils::dump(ArrayRef<Instruction *> Bndl) { dumpImpl(Bndl); }
#endif // NDEBUG

} // namespace llvm::sandboxir
