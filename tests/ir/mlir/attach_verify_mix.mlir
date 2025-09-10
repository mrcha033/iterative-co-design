// RUN: icd-mlir-opt --icd-attach-metadata --icd-verify %s | FileCheck %s
module {
  %0 = "stablehlo.dot"(%a, %b) : (tensor<2x4xf16>, tensor<4x8xf16>) -> tensor<2x8xf16>
  // CHECK: icd.layout_perm
  // CHECK: icd.metrics
}

