// JarvisOS: tells Java whether this CPU can run libllama_jarvis.
//
// On x86_64 the engine is built with AVX2/FMA/F16C (see Android.bp), which
// the Android x86_64 baseline does not guarantee: the emulator only has them
// when started with `-qemu -cpu host`. Loading the engine on a CPU without
// them would kill system_server with SIGILL, so LlamaEngine loads this small
// library first (built for the baseline) and asks before loading the other.

#include <jni.h>

extern "C" JNIEXPORT jboolean JNICALL
Java_com_android_server_jarvis_inference_LlamaEngine_nativeCpuSupported(JNIEnv*, jclass) {
#if defined(__x86_64__)
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")
            && __builtin_cpu_supports("f16c") ? JNI_TRUE : JNI_FALSE;
#else
    return JNI_TRUE;
#endif
}
