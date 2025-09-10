// RUN: icd-mlir-opt --icd-attach-metadata %s | FileCheck %s
module {
  %0 = "stablehlo.constant"() {value = dense<0.0> : tensor<1xf32>} : () -> tensor<1xf32>
  // CHECK: icd.layout_tag = "icd/v1"
  // CHECK: icd.layout_perm
}

