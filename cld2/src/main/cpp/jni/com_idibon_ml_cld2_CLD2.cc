#include "../public/compact_lang_det.h"
#include "../public/encodings.h"
#include "include/com_idibon_ml_cld2_CLD2.h"

extern "C" {

JNIEXPORT jint JNICALL Java_com_idibon_ml_cld2_CLD2_cld2_1Detect
  (JNIEnv *env, jclass klass, jbyteArray content, jboolean plainText) {

    jbyte *bytes = env->GetByteArrayElements(content, NULL);
    jint length = env->GetArrayLength(content);
    bool reliable;

    CLD2::Language language[3];
    int percents[3];
    double scores[3];
    int text_bytes;
    CLD2::CLDHints hints = { NULL, NULL,
        CLD2::UNKNOWN_ENCODING, CLD2::UNKNOWN_LANGUAGE };

    CLD2::ExtDetectLanguageSummary(reinterpret_cast<const char *>(bytes),
        length, !!plainText, &hints, CLD2::kCLDFlagBestEffort, language,
        percents, scores, NULL, &text_bytes, &reliable);

    env->ReleaseByteArrayElements(content, bytes, JNI_ABORT);
    return reliable ? language[0] : -language[0];
}

}
