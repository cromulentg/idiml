#include "../public/compact_lang_det.h"
#include "include/com_idibon_ml_cld2_CLD2.h"

extern "C" {

JNIEXPORT jint JNICALL Java_com_idibon_ml_cld2_CLD2_cld2_1Detect
  (JNIEnv *env, jclass klass, jbyteArray content, jboolean plainText) {

    jbyte *bytes = env->GetByteArrayElements(content, NULL);
    jint length = env->GetArrayLength(content);
    bool reliable;


    CLD2::Language language = CLD2::DetectLanguage(
        reinterpret_cast<const char *>(bytes), length, !!plainText, &reliable);

    env->ReleaseByteArrayElements(content, bytes, JNI_ABORT);
    if (reliable)
        return language;
    else
        return -1;
}

}
