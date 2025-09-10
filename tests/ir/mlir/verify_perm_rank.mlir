// RUN: icd-mlir-opt --icd-attach-metadata --icd-verify %s | FileCheck %s
module {
  %0 = "stablehlo.add"(%a, %b) : (tensor<3x5xf16>, tensor<3x5xf16>) -> tensor<3x5xf16>
  // CHECK: icd.layout_perm
}
// CHECK: icd.metrics

