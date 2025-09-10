// RUN: icd-mlir-opt --icd-attach-metadata --icd-verify %s | FileCheck %s
module attributes {sym_name = "noop"} {
}
// CHECK: module

