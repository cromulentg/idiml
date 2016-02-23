package com.idibon.ml.feature.language

import scala.util.Try
import com.idibon.ml.feature._
import com.idibon.ml.alloy.Codec
import com.ibm.icu.util.ULocale

/** Feature representing an ISO 639-2 3-letter language code
  *
  * @param iso_639_2 - an ISO 639-2 3-letter language code, or None
  *   if the language is unknown
  */
case class LanguageCode(iso_639_2: Option[String])
    extends Feature[LanguageCode]
    with Buildable[LanguageCode, LanguageCodeBuilder] {

  def get = this

  /** Saves this feature to an output stream
    *
    * Writes the ISO 639-2 3-letter language code to the stream as a string,
    * or an empty string for an unknown language */
  def save(output: FeatureOutputStream) {
    Codec.String.write(output, iso_639_2.getOrElse(""))
  }

  def getAsString: Option[String] = ???

  /** An ICU4J ULocale representation of the language code */
  val icuLocale: Option[ULocale] = iso_639_2.map(code => {
    val locale = ULocale.forLanguageTag(code)

    if (locale.getISO3Language != code)
      throw new IllegalArgumentException(s"Invalid ISO 639-2 sequence $code")

    locale
  })
}

object LanguageCode {
  /** Returns a normalized ISO 639-2 language code from various inputs
    *
    * Converts ISO 639-1 2-letter sequences, ISO 639-2 3-letter sequences,
    * and ISO 639-1 + country sequences into ISO 639-2 3-letter sequences.
    * Returns None if the input sequence could not be mapped to an ISO 639-2
    * 3-letter code.
    */
  def normalize(code: String): Option[String] = {
    Try(ULocale.forLanguageTag(code).getISO3Language).toOption
      .flatMap(iso3 => if (iso3.isEmpty) None else Some(iso3))
  }
}

/** Paired builder class for LanguageCode */
class LanguageCodeBuilder extends Builder[LanguageCode] {

  def build(input: FeatureInputStream): LanguageCode = {
    val code = Codec.String.read(input)
    if (code.isEmpty) new LanguageCode(None) else new LanguageCode(Some(code))
  }
}
