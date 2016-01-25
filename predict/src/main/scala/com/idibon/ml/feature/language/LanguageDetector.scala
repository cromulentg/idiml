package com.idibon.ml.feature.language

import scala.util.Try
import com.idibon.ml.feature.{Feature, FeatureTransformer}
import com.idibon.ml.cld2._
import com.ibm.icu.util.ULocale
import org.json4s._

/** Generates a feature based on the primary language used in a document.
  */
class LanguageDetector extends FeatureTransformer {

  /** Uses language metadata or language detection to create features
    *
    * @param document - a parsed JSON document
    * @return - a LanguageCode feature
    */
  def apply(document: JObject): LanguageCode = {
    (document \ "metadata" \ "iso_639_1").toOption
      .flatMap(code => LanguageCode.normalize(code.asInstanceOf[JString].s))
      .map(iso639_2 => LanguageCode(Some(iso639_2)))
      .orElse({
        (document \ "content").toOption
          .map(_.asInstanceOf[JString].s)
          .map(text => identifyLocale(text, CLD2.DocumentMode.PlainText))
      }).getOrElse(LanguageCode(None))
  }

  /** Uses language detection to identify the correct locale for a string.
    *
    * Returns None if language detection fails for any reason, including native
    * library initialization errors and failed (unknown) detection.
    */
  private[language] def identifyLocale(content: String,
    documentMode: CLD2.DocumentMode) = Try({
      /* if CLD2 initialization fails, or detection throws an exception for
       * some reason, return None. fall-through cases are handled by the
       * thrown MatchNotFound exception */
      LanguageCode(LanguageCode.normalize(CLD2.detect(content, documentMode).code))
    }).getOrElse(LanguageCode(None))
}
