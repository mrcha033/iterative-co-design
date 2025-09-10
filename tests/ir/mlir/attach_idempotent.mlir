// RUN: icd-mlir-opt --icd-attach-metadata --icd-attach-metadata %s | FileCheck %s
module {
  %0 = "stablehlo.constant"() {value = dense<0.0> : tensor<2x4xf16>} : () -> tensor<2x4xf16>
  // CHECK: icd.layout_tag = "icd/v1"
  // CHECK-NOT: icd.layout_tag = "icd/v1".*icd.layout_tag
}

