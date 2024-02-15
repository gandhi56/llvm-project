//===- AMDGPUSplitModule.cpp ----------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
/// \file Implements a module splitting algorithm designed to support the
/// FullLTO --lto-partitions option for parallel codegen. This is completely
/// different from the common SplitModule pass, as this pass is designed with
/// AMDGPU in mind.
///
/// The core idea behind the pass is the same as SplitModule: load-balance the
/// module's functions across a set of N partitions to allow parallel codegen.
/// However, it does so differently:
///   - Kernels are used as the module's "roots".
///     They're known entry points on AMDGPU, and everything else is often
///     internal only.
///   - Each kernel has a set of dependencies, and when a kernel and its
///     dependencies is considered "big", we try to put it in a partition where
///     most dependencies are already imported, to avoid duplicating large
///     amounts of code.
///   - There's special care for indirect calls in order to ensure
///     AMDGPUResourceUsageAnalysis can work correctly.
///
/// Improvement ideas for this pass:
///   - Be more careful with expensive global variables. Some libraries ship
///   static variables with precomputed data, and they can sometimes be huge.
///   This pass just aggressively copies those everywhere and prunes the ones
///   that aren't needed after. This is suboptimal, and if multiple copies of
///   those variables are made, it's even worse because with private linkage,
///   all copies will be kept by the linker.
///   - Be more careful with code size in general - keep track of how much
///   copies of each function are present in each partition.
///   - Smarter indirect call handling, e.g. if a function is known to have low
///   SGPR/VGPR/LDS usage, we can probably ignore it.
///   - Smarter dependencies collection: if some attributes can be used to infer
///   resource usage of a cluster of functions, maybe we don't need to consider
///   those functions "dependencies" of the kernel. We could externalize
///   some functions to enable better partitioning.
///   - Allow flexible partitioning: pass a min/max number of partitions and let
///   the pass create new partitions only if needed. With the current impl we
///   can easily get modules with only one small function, or even with no
///   functions at all. This could also elegantly handle the "no splitting
///   possible" case by just returning one partition (= the input module).
//
//===----------------------------------------------------------------------===//

#include "AMDGPUSplitModule.h"
#include "AMDGPUTargetMachine.h"
#include "Utils/AMDGPUBaseInfo.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Analysis/CallGraph.h"
#include "llvm/Analysis/TargetTransformInfo.h"
#include "llvm/IR/Function.h"
#include "llvm/IR/Instruction.h"
#include "llvm/IR/Module.h"
#include "llvm/IR/User.h"
#include "llvm/IR/Value.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/raw_ostream.h"
#include "llvm/Transforms/Utils/Cloning.h"
#include <algorithm>
#include <cassert>
#include <iterator>
#include <memory>
#include <utility>
#include <vector>

using namespace llvm;

#define DEBUG_TYPE "amdgpu-split-module"

namespace {

static cl::opt<float> LargeKernelFactor(
    "amdgpu-module-splitting-large-kernel-threshold", cl::init(2.0), cl::Hidden,
    cl::desc(
        "consider a kernel as large and needing special treatment when it "
        "exceeds the average cost of a partition by this factor; e;g. 2.0 "
        "means if the kernel and its dependencies is 2 times bigger than "
        "an average partition; 0 disables large kernels handling entirely"));

static cl::opt<float> LargeKernelOverlapForMerge(
    "amdgpu-module-splitting-large-kernel-merge-overlap", cl::init(0.8),
    cl::Hidden,
    cl::desc("defines how much overlap between two large kernel's dependencies "
             "is needed to put them in the same partition"));

using CostType = InstructionCost::CostType;
using PartitionID = unsigned;

/// Calculate the cost of each function in \p M
/// \param TM TargetMachine instance used to retrieve TargetTransformInfo.
/// \param M Module to analyze.
/// \param CostMap[out] Resulting Function -> Cost map.
/// \return The module's total cost.
static CostType
calculateFunctionCosts(const AMDGPUTargetMachine &TM, Module &M,
                       DenseMap<const Function *, CostType> &CostMap) {
  CostType ModuleCost = 0;
  CostType KernelCost = 0;

  for (auto &Fn : M) {
    if (Fn.isDeclaration())
      continue;

    CostType FnCost = 0;
    auto TTI = TM.getTargetTransformInfo(Fn);

    for (auto &BB : Fn) {
      for (auto &I : BB) {
        auto Cost =
            TTI.getInstructionCost(&I, TargetTransformInfo::TCK_CodeSize);
        assert(Cost != InstructionCost::getMax());
        // Assume expensive if we can't tell the cost of an instruction.
        CostType CostVal =
            Cost.getValue().value_or(TargetTransformInfo::TCC_Expensive);
        assert((FnCost + CostVal) >= FnCost && "Overflow!");
        FnCost += CostVal;
      }
    }

    assert(FnCost != 0);

    CostMap[&Fn] = FnCost;
    assert((ModuleCost + FnCost) >= ModuleCost && "Overflow!");
    ModuleCost += FnCost;

    if (AMDGPU::isKernelCC(&Fn))
      KernelCost += FnCost;
  }

  LLVM_DEBUG(CostType FnCost = (ModuleCost - KernelCost);
             dbgs() << "=> Total Module Cost: " << ModuleCost << "\n"
                    << "  => KernelCost: " << KernelCost << " ("
                    << format("%0.2f", (float(KernelCost) / ModuleCost) * 100)
                    << "%)\n"
                    << "  => FnsCost: " << FnCost << " ("
                    << format("%0.2f", (float(FnCost) / ModuleCost) * 100)
                    << "%)\n");

  return ModuleCost;
}

/// When a kernel or any of its callees performs an indirect call, this function
/// takes over \ref addAllDependencies and adds all potentially callable
/// functions to \p Fns so they can be counted as dependencies of the kernel.
///
/// This is needed due to how AMDGPUResourceUsageAnalysis operates: in the
/// presence of an indirect call, the function's resource usage is the same as
/// the most expensive function in the module.
/// \param M    The module.
/// \param Fns[out] Resulting list of functions.
static void addAllIndirectCallDependencies(const Module &M,
                                           DenseSet<const Function *> &Fns) {
  for (const auto &Fn : M) {
    if (!AMDGPU::isEntryFunctionCC(Fn.getCallingConv()))
      Fns.insert(&Fn);
  }
}

/// Adds the functions that \p Fn may call to \p Fns, then recurses into each
/// callee until all reachable functions have been gathered.
///
/// \param CG Call graph for \p Fn's module.
/// \param Fn Current function to look at.
/// \param Fns[out] Resulting list of functions.
/// \param HadIndirectCall[out] Set to true if an indirect call was seen at some
/// point, either in \p Fn or in one of the function it calls. When that
/// happens, we fall back to adding all callable functions inside \p Fn's module
/// to \p Fns.
/// \param HadExternalCall[out] Set to true if a call to an external function
/// was seen at some point, either in \p Fn or in one of the function it calls
static void addAllDependencies(const CallGraph &CG, const Function &Fn,
                               DenseSet<const Function *> &Fns,
                               bool &HadIndirectCall, bool &HadExternalCall) {
  assert(!Fn.isDeclaration());

  const Module &M = *Fn.getParent();
  SmallVector<const Function *> WorkList({&Fn});
  while (!WorkList.empty()) {
    const auto &CurFn = *WorkList.pop_back_val();

    // Scan for an indirect call. If such a call is found, we have to
    // conservatively assume this can call all non-entrypoint functions in the
    // module.
    for (const auto &BB : CurFn) {
      for (const auto &I : BB) {
        const auto *CB = dyn_cast<CallBase>(&I);
        if (!CB || !CB->isIndirectCall())
          continue;

        LLVM_DEBUG(dbgs() << "Indirect call detected in " << CurFn.getName()
                          << " - treating all non-entrypoint functions as "
                             "potential dependencies\n");
        // TODO: Print an ORE as well.
        addAllIndirectCallDependencies(M, Fns);
        HadIndirectCall = true;
        return;
      }
    }

    for (auto &CGEntry : *CG[&CurFn]) {
      auto *Callee = CGEntry.second->getFunction();
      if (!Callee)
        continue;

      assert(!AMDGPU::isKernelCC(Callee));

      if (Callee->isDeclaration())
        continue;

      if (Callee->hasExternalLinkage())
        HadExternalCall = true;

      auto [It, Inserted] = Fns.insert(Callee);
      if (Inserted)
        WorkList.push_back(Callee);
    }
  }
}

/// Contains information about a kernel and its dependencies.
struct KernelWithDependencies {
  KernelWithDependencies(CallGraph &CG,
                         const DenseMap<const Function *, CostType> &FnCosts,
                         const Function *Fn)
      : Fn(Fn) {
    addAllDependencies(CG, *Fn, Dependencies, HasIndirectCall, HasExternalCall);
    TotalCost = FnCosts.at(Fn);
    for (const auto *Dep : Dependencies)
      TotalCost += FnCosts.at(Dep);
  }

  const Function *Fn = nullptr;
  DenseSet<const Function *> Dependencies;
  /// Whether \p Fn or any of its \ref Dependencies contains an indirect call.
  bool HasIndirectCall = false;
  /// Whether \p Fn or any of its \ref Dependencies contains a call to a
  /// function with external linkage.
  bool HasExternalCall = false;

  CostType TotalCost = 0;
};

/// Calculates how much overlap there is between \p A and \p B.
/// \return A number between 0.0 and 1.0, where 1.0 means A == B and 0.0 means A
/// and B have no shared elements. Kernels do not count in overlap calculation.
static float calculateOverlap(const DenseSet<const Function *> &A,
                              const DenseSet<const Function *> &B) {
  DenseSet<const Function *> Total;
  for (const auto *F : A) {
    if (!AMDGPU::isKernelCC(F))
      Total.insert(F);
  }

  unsigned NumCommon = 0;
  for (const auto *F : B) {
    if (AMDGPU::isKernelCC(F))
      continue;

    auto [It, Inserted] = Total.insert(F);
    if (!Inserted)
      ++NumCommon;
  }

  return float(NumCommon) / Total.size();
}

/// Performs all of the partitioning work on \p M.
/// \param M Module to partition.
/// \param NumParts Number of partitions to create.
/// \param ModuleCost Total cost of all functions in \p M.
/// \param FnCosts Map of Function -> Cost
/// \param WorkList Kernels and their dependencies to process in order.
/// \returns The created partitions (a vector of size \p NumParts )
static std::vector<DenseSet<const Function *>>
doPartitioning(Module &M, unsigned NumParts, CostType ModuleCost,
               const DenseMap<const Function *, CostType> &FnCosts,
               const SmallVector<KernelWithDependencies> &WorkList) {

  // Calculate a "large kernel threshold". When more than one kernel's total
  // import cost exceeds this value, we will try to merge it with other,
  // similarly large kernels.
  //
  // e.g. let two kernels X and Y have a import cost of ~10% of the module, we
  // assign X to a partition as usual, but when we get to Y, we check if it's
  // worth also putting it in Y's partition.
  const CostType LargeKernelThreshold =
      LargeKernelFactor ? ((ModuleCost / NumParts) * LargeKernelFactor)
                        : std::numeric_limits<CostType>::max();

  std::vector<DenseSet<const Function *>> Partitions;
  Partitions.resize(NumParts);

  // Assign a partition to each kernel, and try to keep the partitions more or
  // less balanced. We do that through a priority queue sorted in reverse, so we
  // can always look at the partition with the least content.
  //
  // There are some cases where we will be deliberately unbalanced though.
  //  - Large kernels: we try to merge with existing partitions to reduce code
  //  duplication.
  //  - Kernels with indirect or external calls always go in the first partition
  //  (P0).
  auto ComparePartitions = [](const std::pair<PartitionID, CostType> &a,
                              const std::pair<PartitionID, CostType> &b) {
    // When two partitions have the same cost, assign to the one with the
    // biggest ID first. This allows us to put things in P0 last, because P0 may
    // have other stuff added later.
    if (a.second == b.second)
      return a.first < b.first;
    return a.second > b.second;
  };

  // We can't use priority_queue here because we need to be able to access any
  // element. This makes this a bit inefficient as we need to sort it again
  // everytime we change it, but it's a very small array anyway (likely under 64
  // partitions) so it's a cheap operation.
  std::vector<std::pair<PartitionID, CostType>> BalancingQueue;
  for (unsigned I = 0; I < NumParts; ++I)
    BalancingQueue.push_back(std::make_pair(I, 0));

  // Helper function to handle assigning a kernel to a partition. This takes
  // care of updating the balancing queue.
  const auto AssignToPartition = [&](PartitionID PID,
                                     const KernelWithDependencies &KWD) {
    auto &FnsInPart = Partitions[PID];
    FnsInPart.insert(KWD.Fn);
    FnsInPart.insert(KWD.Dependencies.begin(), KWD.Dependencies.end());
    // clang-format off
    LLVM_DEBUG(dbgs() << "assign " << KWD.Fn->getName() << " to P" << PID
                      << "\n  ->  ";
               if(!KWD.Dependencies.empty()) {
                dbgs() << KWD.Dependencies.size() << " dependencies added\n";
               });
    // clang-format on

    // Update the balancing queue. we scan backwards because in the common case
    // the partition is at the end.
    for (auto &[QueuePID, Cost] : reverse(BalancingQueue)) {
      if (QueuePID == PID) {
        CostType NewCost = 0;
        for (auto *Fn : Partitions[PID])
          NewCost += FnCosts.at(Fn);

        // clang-format off
        LLVM_DEBUG(dbgs() << "[Updating P" << PID << " Cost]:" << Cost << " -> "
                          << NewCost;
                   if (Cost) {
                     dbgs() << " ("
                            << unsigned(((float(NewCost) / Cost) - 1) * 100)
                            << "% increase)";
                   }
                   dbgs() << "\n");
        // clang-format on

        Cost = NewCost;
      }
    }

    sort(BalancingQueue, ComparePartitions);
  };

  for (auto &CurKernel : WorkList) {
    // When a kernel has indirect calls, it must stay in the first partition
    // alongside every reachable non-entry function. This is a nightmare case
    // for splitting as it severely limits what we can do.
    if (CurKernel.HasIndirectCall) {
      LLVM_DEBUG(dbgs() << "Kernel with indirect call(s): "
                        << CurKernel.Fn->getName() << " defaulting to P0\n");
      AssignToPartition(0, CurKernel);
      continue;
    }

    // When a kernel calls external functions, we have to keep it in the first
    // partition as well. This is because we cannot duplicate external functions
    // into multiple modules. To avoid duplicating accidentally, we
    // conservatively put every external function in P0.
    if (CurKernel.HasExternalCall) {
      LLVM_DEBUG(dbgs() << "Kernel with external call(s): "
                        << CurKernel.Fn->getName() << " defaulting to P0\n");
      AssignToPartition(0, CurKernel);
      continue;
    }

    // Take care of large kernels now. Those should ideally be merged with an
    // existing partition to avoid duplicating a lot of code.
    if (CurKernel.TotalCost > LargeKernelThreshold) {
      assert(LargeKernelOverlapForMerge >= 0.0f &&
             LargeKernelOverlapForMerge <= 1.0f);
      LLVM_DEBUG(dbgs() << "Large Kernel: " << CurKernel.Fn->getName()
                        << " - looking for partition with at least "
                        << format("%0.2f", LargeKernelOverlapForMerge * 100)
                        << "% overlap\n");

      bool Assigned = false;
      for (const auto &[PID, Fns] : enumerate(Partitions)) {
        float Overlap = calculateOverlap(CurKernel.Dependencies, Fns);
        LLVM_DEBUG(dbgs() << "  => " << format("%0.2f", Overlap * 100)
                          << "% overlap with P" << PID << "\n");
        if (Overlap > LargeKernelOverlapForMerge) {
          LLVM_DEBUG(dbgs() << "  selecting P" << PID << "\n");
          AssignToPartition(PID, CurKernel);
          Assigned = true;
        }
      }

      if (Assigned)
        continue;
    }

    // Normal "load-balancing", assign to partition with least pressure.
    auto [PID, CurCost] = BalancingQueue.back();
    AssignToPartition(PID, CurKernel);
  }

  // Work is mostly done now, verify the partioning and add all functions we may
  // have missed (= unreachable, or we don't understand how they're reached) to
  // P0.
  DenseSet<const Function *> AllFunctions;
  for (const auto &[Idx, Part] : enumerate(Partitions)) {
    [[maybe_unused]] CostType Cost = 0;
    for (auto *Fn : Part) {
      // external linkage functions should exclusively be in the first partition
      // at this stage. In theory, we should only ever see external linkage
      // functions here if they're kernels, or if they've been added due to a
      // kernel using indirect calls somewhere in its CallGraph.
      assert(Idx == 0 || (!Fn->hasExternalLinkage() || AMDGPU::isKernelCC(Fn)));
      LLVM_DEBUG(Cost += FnCosts.at(Fn));
    }
    LLVM_DEBUG(dbgs() << "P" << Idx << " has a total cost of " << Cost << " ("
                      << format("%0.2f", (float(Cost) / ModuleCost) * 100)
                      << "% of source module)\n");
    AllFunctions.insert(Part.begin(), Part.end());
  }

  // Add missed functions to P0. This will take care of adding things like
  // external functions with no callers in the module to P0. This should be
  // fairly rare as AMDGPU internalizes everything in most cases, so unused
  // internal functions would get removed.
  for (auto &Fn : M) {
    if (!Fn.isDeclaration() && !AllFunctions.contains(&Fn)) {
      LLVM_DEBUG(dbgs() << Fn.getName()
                        << " has no partition assigned, defaulting to P0\n");
      Partitions[0].insert(&Fn);
    }
  }

  return Partitions;
}
} // end anonymous namespace

void llvm::splitAMDGPUModule(
    const AMDGPUTargetMachine &TM, Module &M, unsigned N,
    function_ref<void(std::unique_ptr<Module> MPart)> ModuleCallback) {
  CallGraph CG(M);

  // Start by calculating the cost of every function in the module, as well as
  // the module's overall cost.
  DenseMap<const Function *, CostType> FnCosts;
  const CostType ModuleCost = calculateFunctionCosts(TM, M, FnCosts);

  // Gather every kernel into a WorkList, then sort it by descending total cost
  // of the kernel so the biggest kernels are seen first.
  SmallVector<KernelWithDependencies> WorkList;
  for (auto &Fn : M) {
    if (AMDGPU::isKernelCC(&Fn) && !Fn.isDeclaration())
      WorkList.emplace_back(CG, FnCosts, &Fn);
  }
  sort(WorkList, [&](auto &A, auto &B) {
    // Sort by total cost, and if the total cost is identical, sort
    // alphabetically.
    if (A.TotalCost == B.TotalCost)
      return A.Fn->getName() < B.Fn->getName();
    return A.TotalCost > B.TotalCost;
  });

  // This performs all of the partitioning work.
  auto Partitions = doPartitioning(M, N, ModuleCost, FnCosts, WorkList);
  assert(Partitions.size() == N);

  // Some GVs need to be conservatively imported into every module (including
  // their initializers), and then cleaned up afterwards.
  //
  // FIXME: Should large GVs (with big initializers) be an extra splitting
  // constraint? This could duplicate huge GVs (containing stuff like
  // precomputed datasets into) into multiple modules, bloating up the code.
  const auto NeedsConservativeImport = [&](const GlobalValue *GV) {
    // We conservatively import private/internal GVs into every module and clean
    // them up afterwards.
    const auto *Var = dyn_cast<GlobalVariable>(GV);
    return Var && (Var->hasPrivateLinkage() || Var->hasInternalLinkage());
  };

  LLVM_DEBUG(dbgs() << "Creating " << N << " modules...\n");
  for (unsigned I = 0; I < N; ++I) {
    const auto &FnsInPart = Partitions[I];

    ValueToValueMapTy VMap;
    std::unique_ptr<Module> MPart(
        CloneModule(M, VMap, [&](const GlobalValue *GV) {
          // Functions go in their assigned partition.
          if (const auto *Fn = dyn_cast<Function>(GV)) {
// Check we don't import an external linkage function in any
// partition other than P0.
#ifndef NDEBUG
            if (Fn->hasExternalLinkage() && !AMDGPU::isKernelCC(Fn)) {
              assert((I == 0) == FnsInPart.contains(Fn));
            }
#endif
            return FnsInPart.contains(Fn);
          }

          if (NeedsConservativeImport(GV))
            return true;

          // Everything else goes in the first partition.
          return I == 0;
        }));
    if (I != 0)
      MPart->setModuleInlineAsm("");

    // Clean-up conservatively imported GVs without any users.
    for (auto &GV : make_early_inc_range(MPart->globals())) {
      if (NeedsConservativeImport(&GV) && GV.use_empty())
        GV.eraseFromParent();
    }

    unsigned NumAllFns = 0, NumKernels = 0;
    for (auto &Cur : *MPart) {
      if (!Cur.isDeclaration()) {
        ++NumAllFns;
        if (AMDGPU::isKernelCC(&Cur))
          ++NumKernels;
      }
    }
    LLVM_DEBUG(dbgs() << "  - Module " << I << " with " << NumAllFns
                      << " functions (" << NumKernels << " kernels)\n");
    ModuleCallback(std::move(MPart));
  }
}
