git clone https://github.com/tensorflow/tensorflow.git
pushd tensorflow
tensorflow/contrib/makefile/download_dependencies.sh
tensorflow/contrib/makefile/compile_linux_protobuf.sh
# incude operations so lstm block is available
cat <<EOF >> tensorflow/BUILD

# Added build rule
cc_binary(
    name = "libtensorflow_all.so",
    linkshared = 1,
    linkopts = ["-Wl,--version-script=tensorflow/tf_version_script.lds"], # Remove this line if you are using MacOS
    deps = [
        "//tensorflow/core:framework_internal",
        "//tensorflow/core:tensorflow",
        "//tensorflow/cc:cc_ops",
        "//tensorflow/cc:client_session",
        "//tensorflow/cc:scope",
        "//tensorflow/c:c_api",
        "//tensorflow/contrib/rnn:gru_ops_kernels",
        "//tensorflow/contrib/rnn:lstm_ops_kernels",
    ],
)
EOF
# include to contrib/rnn builds
patch -p 1 <<EOF
diff --git a/tensorflow/contrib/rnn/BUILD b/tensorflow/contrib/rnn/BUILD
index 43c0f75..9b45fa8 100644
--- a/tensorflow/contrib/rnn/BUILD
+++ b/tensorflow/contrib/rnn/BUILD
@@ -333,6 +333,8 @@ tf_kernel_library(
     srcs = [
         "kernels/blas_gemm.cc",
         "kernels/blas_gemm.h",
+        "kernels/gru_ops.h",
+        "ops/gru_ops.cc",
     ],
     gpu_srcs = [
         "kernels/blas_gemm.h",
@@ -351,6 +353,8 @@ tf_kernel_library(
     srcs = [
         "kernels/blas_gemm.cc",
         "kernels/blas_gemm.h",
+        "kernels/lstm_ops.h",
+        "ops/lstm_ops.cc",
     ],
     gpu_srcs = [
         "kernels/blas_gemm.h",
EOF

# build TF library
bazel build //tensorflow:libtensorflow_all.so

popd
